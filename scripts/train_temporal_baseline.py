#!/usr/bin/env python
"""Original TGN/TAMI modules with annual forecasting supervision and replay."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from torch.nn import functional as F
from patentflow.data import ROOT, atomic_json
from patentflow.dataset import DatasetView
from patentflow.evaluate import evaluate_queries
from patentflow.resources import ResourceMonitor
from patentflow.event_stream import EventStream


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(.3, torch.device(args.device))
    data = DatasetView(args.dataset, args.level)
    stream = EventStream(data)
    candidates = data.candidate_ids.astype(np.int64) + stream.org_count
    out = ROOT / 'results/deep' / args.dataset / args.method / args.tag / ('s%d' % args.seed)
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    config_path = out / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in cfg:
            if key not in ['device', 'threads', 'resume', 'evaluate_test'] and cfg[key] != previous.get(key):
                raise ValueError('Configuration differs: ' + key)
    atomic_json(config_path, cfg)
    if args.method == 'EDGPAT':
        from patentflow.edgpat_baseline import EDGPATForecast
        model = EDGPATForecast(stream, args.dim, args.time_dim, args.device)
    elif args.method == 'IDGP':
        from patentflow.idgp_baseline import IDGPForecast
        model = IDGPForecast(stream, args.dim, args.time_dim, args.device)
    else:
        from patentflow.temporal_baselines import TemporalForecast
        model = TemporalForecast(stream, args.method, args.dim, args.time_dim,
                                 args.neighbors, args.device, args.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    atomic_json(out / 'original_source_hashes.json', model.provenance())
    atomic_json(out / 'codes.json', {'codes': data.codes, 'candidate_ids': data.candidate_ids.tolist()})
    train_years = data.split_years('train')
    queries = {y: data.queries(y)[0] for y in train_years}
    labels = {y: data.counts(y)[:, data.candidate_ids].sign().tocsr() for y in train_years}
    positives = sum(np.asarray(labels[y][queries[y]].sum(axis=0)).ravel() for y in train_years)
    query_count = sum(len(q) for q in queries.values())
    weights = torch.as_tensor(np.clip((query_count-positives)/(positives+1), 1, 30),
                              dtype=torch.float32, device=args.device)
    checkpoint = out / 'best.pt'
    best, best_epoch, first_epoch, stale = -float('inf'), -1, 0, 0
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError('Use --resume or a new --tag')
        saved = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(saved['model'], strict=False)
        optimizer.load_state_dict(saved['optimizer'])
        best, best_epoch = saved['validation_ndcg20'], saved['epoch']
        first_epoch = best_epoch + 1
        if (out / 'validation.json').exists():
            first_epoch = args.epochs

    def optimize(nodes, year, pending=None, scale=1.):
        target = torch.as_tensor(labels[year][nodes].toarray(), device=args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.score(nodes, candidates, pending)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite temporal baseline loss')
        (scale * loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        return float(loss.detach())

    def evaluate(year, path=None):
        model.eval()
        replay_start = time.monotonic()
        model.replay(year-1)
        replay_seconds = time.monotonic() - replay_start
        if args.method in ['EDGPAT', 'IDGP']:
            def score(nodes):
                with torch.no_grad():
                    return model.score(nodes, candidates).cpu().numpy()
            metrics = evaluate_queries(data, year, score, path, batch_size=args.batch_size)
            metrics['replay_seconds'] = replay_seconds
            return metrics
        orgs = data.queries(year)[0]
        all_nodes = np.r_[orgs, candidates]
        embeddings = []
        with torch.no_grad():
            for offset in range(0, len(all_nodes), 1024):
                embeddings.append(model.encode(torch.as_tensor(all_nodes[offset:offset+1024], device=args.device)).cpu())
        embeddings = torch.cat(embeddings)
        dst = embeddings[len(orgs):].to(args.device)
        def score(nodes):
            src = embeddings[np.searchsorted(orgs, nodes)].to(args.device)
            with torch.no_grad():
                return model.score_embeddings(nodes, candidates, src, dst).cpu().numpy()
        metrics = evaluate_queries(data, year, score, path, batch_size=args.batch_size)
        metrics['replay_seconds'] = replay_seconds
        return metrics

    start = time.monotonic()
    with ResourceMonitor(out / 'resources.jsonl') as monitor:
        for epoch in range(first_epoch, args.epochs):
            model.train(); model.reset()
            epoch_start = time.monotonic()
            rng = np.random.RandomState(args.seed + epoch)
            loss_sum, n_annual, n_daily = 0., 0, 0
            current_year = None
            def annual(year):
                if year+1 not in train_years:
                    return 0., 0
                order = rng.permutation(queries[year+1])[:args.max_queries_per_year]
                total = 0.
                for offset in range(0, len(order), args.batch_size):
                    nodes = order[offset:offset+args.batch_size]
                    total += optimize(nodes, year+1) * len(nodes)
                return total, len(order)
            for left, right, year in stream.groups(data.spec['train_end_year']-1):
                if current_year is not None and year != current_year:
                    value, count = annual(current_year)
                    loss_sum += value; n_annual += count
                current_year = year
                # Auxiliary forecasting supervision updates the actual GRU,
                # through this complete date, with history truncated beforehand.
                eligible = year+1 in train_years and args.daily_queries > 0
                with torch.set_grad_enabled(eligible):
                    pending = model.propose(left, right)
                    if eligible:
                        orgs = np.unique(stream.src[left:right])
                        nodes = rng.permutation(orgs)[:args.daily_queries]
                        loss_sum += optimize(nodes, year+1, pending, args.daily_weight) * len(nodes)
                        n_daily += len(nodes)
                    model.commit(pending, left, right)
            if current_year is not None:
                value, count = annual(current_year)
                loss_sum += value; n_annual += count
            metrics = evaluate(data.spec['validation_year'])
            score = metrics['active']['ndcg_20']
            record = {'epoch': epoch, 'loss': loss_sum/max(n_annual+n_daily, 1),
                      'annual_queries': n_annual, 'auxiliary_daily_queries': n_daily,
                      'seconds': time.monotonic()-epoch_start, 'validation': metrics}
            with (out / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(args.dataset, args.method, args.seed, 'epoch', epoch, 'val_ndcg20', score,
                  'seconds', record['seconds'], flush=True)
            if score is not None and score > best:
                best, best_epoch, stale = score, epoch, 0
                # Memories and TAMI pair histories are reconstructed by exact
                # chronological replay at selected weights, not carried across splits.
                state = {k: v for k, v in model.state_dict().items()
                         if k not in ('memory.memory', 'memory.last_update', 'memory._assoc')
                         and not k.startswith('original.memory.')
                         and not k.startswith('original.memory_updater.memory.')
                         and not k.startswith('original.memory_bank.')
                         and not k.startswith('original.memory_updater.memory_bank.')
                         and k not in ['original.fluxon_bank.states','original.item_flux_hist',
                                       'original.item_time_hist','ipc_cache']}
                temp = out / 'best.tmp.pt'
                torch.save({'model': state, 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                            'validation_ndcg20': score, 'config': cfg,
                            'source_sha256': data.manifest['source_sha256'],
                            'protocol_sha256': data.manifest['protocol_sha256']}, temp)
                temp.replace(checkpoint)
            else:
                stale += 1
            if stale >= args.patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=args.device)['model'], strict=False)
        splits = {}
        for split in (['validation', 'test'] if args.evaluate_test else ['validation']):
            splits[split] = evaluate(data.split_years(split)[0], out / (split+'_queries.npz'))
        result = {'dataset': args.dataset, 'method': args.method, 'seed': args.seed,
                  'best_epoch': best_epoch, 'best_validation_ndcg20': best, 'config': cfg,
                  'parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad), 'splits': splits,
                  'wall_seconds': time.monotonic()-start, 'peak_process_rss_bytes': monitor.peak_rss,
                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else 0,
                  'source_sha256': data.manifest['source_sha256'],
                  'protocol_sha256': data.manifest['protocol_sha256'], 'test_evaluated': args.evaluate_test}
        atomic_json(out / ('metrics.json' if args.evaluate_test else 'validation.json'), result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--method', choices=['TGN', 'TGN_edgetype', 'TAMI', 'EDGPAT', 'IDGP'], default='TGN')
    p.add_argument('--tag', default='v2_default')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--level', type=int, default=3)
    p.add_argument('--dim', type=int, default=100); p.add_argument('--time-dim', type=int, default=32)
    p.add_argument('--neighbors', type=int, default=10); p.add_argument('--gamma', type=float, default=.9)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--patience', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-queries-per-year', type=int, default=20000)
    p.add_argument('--daily-queries', type=int, default=8)
    p.add_argument('--daily-weight', type=float, default=.25)
    p.add_argument('--resume', action='store_true'); p.add_argument('--evaluate-test', action='store_true')
    main(p.parse_args())
