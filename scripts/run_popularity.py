#!/usr/bin/env python
"""Exact TOP and Personal TOP baselines, with common full-catalog evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from patentflow.data import ROOT, atomic_json, file_hash
from patentflow.dataset import DatasetView
from patentflow.metrics import ranking_metrics
from patentflow.resources import ResourceMonitor


def run(name, method, seed, level):
    out = ROOT / 'results' / 'baselines' / name / ('%s_l%d_s%d' % (method, level, seed))
    if (out / 'metrics.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with ResourceMonitor(out / 'resources.jsonl') as monitor:
        data = DatasetView(name, level)
        results = {'dataset': name, 'method': method, 'seed': seed, 'level': level,
                   'deterministic_seed_has_no_effect': True, 'parameter_count': 0,
                   'candidate_count': len(data.candidate_ids), 'hyperparameters': {}, 'splits': {},
                   'label_codes':data.codes,'candidate_ids':data.candidate_ids.tolist(),
                   'source_sha256': data.manifest['source_sha256'],
                   'protocol_sha256': file_hash(ROOT / 'configs/protocol.json')}
        for split in ['validation', 'test']:
            year = data.split_years(split)[0]
            history = data.history(year-1)
            orgs, truth = data.queries(year)
            global_counts = np.asarray(history.sum(axis=0)).ravel()[data.candidate_ids]
            scores_all, truth_all, query_values, ranking_all = [], [], {}, []
            totals = None
            for offset in range(0, len(orgs), 512):
                nodes = orgs[offset:offset+512]
                if method == 'TOP':
                    scores = np.broadcast_to(global_counts, (len(nodes),len(global_counts)))
                else:
                    # Integer personal frequency first, global frequency only breaks ties.
                    scores = history[nodes][:,data.candidate_ids].toarray().astype(np.float64)
                    scores += (global_counts / max(float(global_counts.max()), 1.)) * .5
                metric = ranking_metrics(scores, truth[offset:offset+512], candidate_ids=data.candidate_ids)
                rankings = data.candidate_ids[np.argsort(-scores, axis=1, kind='stable')[:,:min(40,len(data.candidate_ids))]]
                ranking_all.append(rankings)
                for k,v in metric.pop('per_query').items():
                    query_values.setdefault(k, []).append(v)
                if totals is None:
                    totals = metric
                else:
                    for k in ['n_queries','n_active','n_empty','oov_positive_count','positive_count']:
                        totals[k] += metric[k]
            if totals is None:
                raise RuntimeError('No historical organizations to evaluate')
            active = np.asarray([bool(len(t)) for t in truth])
            arrays = {k:np.concatenate(v) for k,v in query_values.items()}
            for k,v in arrays.items():
                totals['active'][k] = float(v[active].mean()) if active.any() else None
                totals['all_zero_filled'][k] = float(v.mean())
            np.savez_compressed(out/(split+'_queries.npz'), org=orgs, active=active,
                                ranked_ids=np.concatenate(ranking_all), **arrays)
            results['splits'][split] = totals
        results['wall_seconds'] = time.monotonic()-start
        results['peak_process_rss_bytes'] = max(monitor.peak_rss, monitor.process.memory_info().rss)
        atomic_json(out / 'metrics.json', results)
    print('COMPLETE', name, method, seed, 'seconds', round(results['wall_seconds'],2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--methods', nargs='+', choices=['TOP','PersonalTOP'], default=['TOP','PersonalTOP'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42,2026,3407])
    p.add_argument('--level', type=int, default=3)
    args=p.parse_args()
    names=args.datasets or [p.stem for p in sorted((ROOT/'data/processed/views').glob('*.json'))]
    if not names:
        raise RuntimeError('No prepared datasets')
    for name in names:
        for method in args.methods:
            for seed in args.seeds:
                run(name,method,seed,args.level)
