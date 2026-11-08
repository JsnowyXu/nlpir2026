"""Read-only checkpoint replay with identical full-catalog metrics."""

import argparse
from pathlib import Path
import time
import torch
from .dataset import DatasetView
from .evaluate import evaluate_queries
from .io import atomic_json, file_hash
from .model import PatentFlow
from .neural_data import MemoryFeatures, anchors_for


def cli():
    parser = argparse.ArgumentParser(epilog="Storage option: --workspace PATH.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    data = DatasetView(config["dataset"], config["level"])
    for key in ["source_sha256", "protocol_sha256"]:
        if checkpoint[key] != data.manifest[key]:
            raise ValueError("Checkpoint/prepared data mismatch: " + key)
    model = PatentFlow(
        anchors_for(data),
        dim=config["dim"],
        communities=config["communities"],
        dropout=config["dropout"],
        disabled=config["disable"],
        temperature=config["temperature"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    year = data.split_years(args.split)[0]
    features = MemoryFeatures(
        data, year - 1, args.device, config["fast"], config["slow"], config["novelty"]
    )
    features.refresh(model)

    def score(nodes):
        with torch.no_grad():
            return model(*features.batch(nodes)).cpu().numpy()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == args.checkpoint.resolve().parent:
        raise ValueError(
            "Use a separate evaluation directory to preserve the training record"
        )
    begin = time.monotonic()
    metrics = evaluate_queries(
        data,
        year,
        score,
        args.output / (args.split + "_queries.npz"),
        batch_size=config["batch_size"],
    )
    result = {
        "dataset": config["dataset"],
        "seed": config["seed"],
        "split": args.split,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "protocol_sha256": checkpoint["protocol_sha256"],
        "metrics": metrics,
        "evaluation_seconds": time.monotonic() - begin,
    }
    atomic_json(args.output / "evaluation.json", result)
    print(
        config["dataset"],
        args.split,
        "NDCG@20",
        metrics["active"]["ndcg_20"],
        flush=True,
    )
