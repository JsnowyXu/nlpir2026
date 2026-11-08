#!/usr/bin/env python
"""Validation-only model selection; test/export is an explicit final-stage flag."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .io import ROOT, RESULTS_ROOT, atomic_json, file_hash, record_run_provenance
from .dataset import DatasetView
from .evaluate import evaluate_queries
from .model import PatentFlow
from .neural_data import MemoryFeatures, anchors_for
from .resources import ResourceMonitor


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(0.3, torch.device(args.device))
    data = DatasetView(args.dataset, args.level)
    config_name = args.tag
    out = RESULTS_ROOT / args.dataset / config_name / ("s%d" % args.seed)
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    config_file = out / "config.json"
    if config_file.exists():
        old = json.loads(config_file.read_text())
        comparable = {
            k: v
            for k, v in cfg.items()
            if k not in ["evaluate_test", "export", "resume", "device", "threads"]
        }
        if any(old.get(k) != v for k, v in comparable.items()):
            raise ValueError("Run directory has a different configuration")
    atomic_json(config_file, cfg)
    record_run_provenance(out, data, cfg)
    atomic_json(
        out / "codes.json",
        {"codes": data.codes, "candidate_ids": data.candidate_ids.tolist()},
    )
    model = PatentFlow(
        anchors_for(data),
        dim=args.dim,
        communities=args.communities,
        dropout=args.dropout,
        disabled=args.disable,
        temperature=args.temperature,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    year_features = {}

    def feature(year):
        if year not in year_features:
            year_features[year] = MemoryFeatures(
                data, year, args.device, args.fast, args.slow, args.novelty
            )
        return year_features[year]

    def score(features, nodes):
        with torch.no_grad():
            return model(*features.batch(nodes)).cpu().numpy()

    start = time.monotonic()
    best = -float("inf")
    best_epoch = -1
    initial_epoch = 0
    stale = 0
    checkpoint = out / "best.pt"
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError("Existing checkpoint; use --resume or a new tag")
        loaded = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        best, best_epoch, initial_epoch = (
            loaded["validation_ndcg20"],
            loaded["epoch"],
            loaded["epoch"] + 1,
        )
        if (out / "validation.json").exists():
            initial_epoch = args.epochs
    train_years = data.split_years("train")
    queries = {y: data.queries(y) for y in train_years}
    # Imbalance weights derive strictly from supervised training labels.
    positive = np.zeros(len(data.candidate_ids), dtype=np.float64)
    nquery = 0
    for y, (orgs, _) in queries.items():
        positive += np.asarray(
            data.counts(y)[orgs][:, data.candidate_ids].sign().sum(axis=0)
        ).ravel()
        nquery += len(orgs)
    pos_weight = torch.as_tensor(
        np.clip((nquery - positive) / (positive + 1), 1, 30),
        dtype=torch.float32,
        device=args.device,
    )
    with ResourceMonitor(out / "resources.jsonl") as monitor:
        for epoch in range(initial_epoch, args.epochs):
            model.train()
            loss_total = 0.0
            n = 0
            epoch_start = time.monotonic()
            rng = np.random.RandomState(args.seed + epoch)
            for target_year in train_years:
                feat = feature(target_year - 1)
                feat.refresh(model)
                orgs, _ = queries[target_year]
                order = rng.permutation(orgs)
                if args.max_queries_per_year:
                    order = order[: args.max_queries_per_year]
                for start_index in range(0, len(order), args.batch_size):
                    nodes = order[start_index : start_index + args.batch_size]
                    y = (
                        data.counts(target_year)[nodes][:, data.candidate_ids]
                        .sign()
                        .toarray()
                    )
                    targets = torch.as_tensor(y, device=args.device)
                    optimizer.zero_grad(set_to_none=True)
                    inputs = feat.batch(nodes)
                    logits = model(*inputs)
                    bce = F.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=pos_weight
                    )
                    active = targets.sum(dim=1) > 0
                    rank = (
                        -(
                            targets[active]
                            / targets[active].sum(dim=1, keepdim=True)
                            * F.log_softmax(logits[active], dim=1)
                        )
                        .sum(dim=1)
                        .mean()
                        if active.any()
                        else logits.sum() * 0
                    )
                    alignment = model.anchor_regularization(feat.ipc_raw)
                    loss = bce + args.rank_loss * rank + args.alignment * alignment
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    loss_total += float(loss.detach()) * len(nodes)
                    n += len(nodes)
            model.eval()
            val = feature(data.spec["validation_year"] - 1)
            val.refresh(model)
            metrics = evaluate_queries(
                data,
                data.spec["validation_year"],
                lambda nodes: score(val, nodes),
                None,
                batch_size=args.batch_size,
            )
            value = metrics["active"]["ndcg_20"]
            record = {
                "epoch": epoch,
                "loss": loss_total / max(n, 1),
                "queries": n,
                "seconds": time.monotonic() - epoch_start,
                "validation": metrics,
                "learning_rate": args.lr,
            }
            with (out / "epochs.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                args.dataset,
                args.seed,
                "epoch",
                epoch,
                "loss",
                record["loss"],
                "val_ndcg20",
                value,
                "seconds",
                record["seconds"],
                flush=True,
            )
            if value is not None and value > best:
                best, best_epoch, stale = value, epoch, 0
                temp = out / "best.tmp.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "validation_ndcg20": value,
                        "config": cfg,
                        "source_sha256": data.manifest["source_sha256"],
                        "protocol_sha256": data.manifest["protocol_sha256"],
                    },
                    temp,
                )
                temp.replace(checkpoint)
            else:
                stale += 1
            if stale >= args.patience:
                break
        if not checkpoint.exists():
            raise RuntimeError("No selected checkpoint")
        model.load_state_dict(torch.load(checkpoint, map_location=args.device)["model"])
        model.eval()
        splits = {}
        for split in ["validation", "test"] if args.evaluate_test else ["validation"]:
            year = data.split_years(split)[0]
            feat = feature(year - 1)
            feat.refresh(model)
            splits[split] = evaluate_queries(
                data,
                year,
                lambda nodes: score(feat, nodes),
                out / (split + "_queries.npz"),
                batch_size=args.batch_size,
            )
        if args.export:
            export = out / "temporal_embeddings"
            export.mkdir(exist_ok=True)
            for year in data.manifest["years"]:
                if year >= data.spec["test_year"]:
                    continue
                feat = feature(year)
                known = np.flatnonzero(data.first_year <= year)
                embeddings = []
                with torch.no_grad():
                    for left in range(0, len(known), 1024):
                        nodes = known[left : left + 1024]
                        org, ipc = model.export_embeddings(
                            torch.as_tensor(feat.org_raw[nodes], device=args.device),
                            feat.ipc_raw,
                        )
                        embeddings.append(org.cpu().numpy())
                np.savez_compressed(
                    export / ("%d.npz" % year),
                    org_ids=known,
                    org=np.concatenate(embeddings),
                    ipc_ids=data.candidate_ids,
                    ipc=ipc.cpu().numpy(),
                    cutoff_year=year,
                )
        result = {
            "dataset": args.dataset,
            "method": "PatentFlow",
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_ndcg20": best,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "splits": splits,
            "wall_seconds": time.monotonic() - start,
            "peak_process_rss_bytes": monitor.peak_rss,
            "gpu_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(args.device)
                if args.device.startswith("cuda")
                else 0
            ),
            "source_sha256": data.manifest["source_sha256"],
            "protocol_sha256": data.manifest["protocol_sha256"],
            "test_evaluated": args.evaluate_test,
            "config": cfg,
        }
        atomic_json(
            out / ("metrics.json" if args.evaluate_test else "validation.json"), result
        )


def cli():
    p = argparse.ArgumentParser(
        epilog="Storage options (handled by train.py): --workspace PATH, --results-root PATH."
    )
    p.add_argument(
        "--config",
        type=Path,
        help="JSON defaults; explicit command-line options take precedence",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tag", default="default")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--communities", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--rank-loss", type=float, default=0.5)
    p.add_argument("--alignment", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-queries-per-year", type=int, default=20000)
    p.add_argument("--fast", type=int, default=180)
    p.add_argument("--slow", type=int, default=1095)
    p.add_argument("--novelty", type=float, default=1.0)
    p.add_argument(
        "--disable",
        nargs="*",
        default=[],
        choices=[
            "text",
            "structure",
            "hierarchy",
            "cooperation",
            "community",
            "repeat",
            "anchor",
            "alignment",
        ],
    )
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--export", action="store_true")
    p.add_argument("--resume", action="store_true")
    preliminary, _ = p.parse_known_args()
    if preliminary.config:
        defaults = json.loads(preliminary.config.read_text())
        allowed = {action.dest for action in p._actions} - {"help", "config"}
        unknown = set(defaults) - allowed
        if unknown:
            raise ValueError("Unknown configuration keys: " + str(sorted(unknown)))
        p.set_defaults(**defaults)
    args = p.parse_args()
    del args.config
    main(args)
