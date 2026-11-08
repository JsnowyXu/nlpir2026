#!/usr/bin/env python3
"""Audit official training artifacts and redraw resource/quality comparisons.

Usage: .venv-evolution/bin/python scripts/analyze_training_resources.py
       .venv-evolution/bin/python scripts/analyze_training_resources.py --datasets CN_All US_All IN_All

Only main/deep validation.json is a training run. Their metrics.json is a
separate test invocation and must not be used as a training duration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/resource_comparison_20261005"
SEEDS = (42, 2026, 3407)
DATASETS = tuple(f"{country}_{part}" for country in ("CN", "US", "IN")
                 for part in ("All", "B", "F", "C"))
METHODS = ("PatentFlow", "EDGPAT", "MENCP", "RepeatMixer", "TGAT",
           "DyGFormer", "TAMI", "IDGP", "TGN", "TGN_edgetype",
           "TOP", "PersonalTOP", "SVM", "XGBoost", "RF")
CLASSICAL = {"TOP", "PersonalTOP", "SVM", "XGBoost", "RF"}
GIB = 1024 ** 3

RUN_COLUMNS = (
    "dataset", "method", "family", "seed", "scope", "device", "protocol_sha256",
    "validation_active_ndcg20", "wall_seconds", "wall_hours", "peak_process_rss_gib",
    "peak_torch_gpu_allocated_gib", "cpu_mean_sampled_percent", "cpu_p95_sampled_percent",
    "cpu_max_sampled_percent", "estimated_cpu_core_seconds", "resource_samples",
    "resource_coverage_fraction", "epochs_completed", "epoch_seconds_sum",
    "train_seconds_explicit", "classical_fit_seconds_explicit",
    "validation_seconds_explicit", "validation_replay_seconds",
    "configured_epochs", "batch_size", "max_queries_per_year", "training_pair_count",
    "parameter_count", "best_epoch", "source_json", "resource_jsonl", "epoch_jsonl",
    "source_sha256", "resource_sha256", "epoch_sha256",
)
SUMMARY_COLUMNS = ("dataset", "method", "family", "n_seeds", "seed_list", "scope",
                   "validation_active_ndcg20_median", "wall_seconds_median",
                   "wall_seconds_min", "wall_seconds_max", "peak_process_rss_gib_median",
                   "peak_torch_gpu_allocated_gib_median", "cpu_mean_sampled_percent_median",
                   "estimated_cpu_core_seconds_median",
                   "epochs_completed_median", "train_seconds_explicit_median",
                   "classical_fit_seconds_explicit_median", "validation_seconds_explicit_median")


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def path_for(dataset, method, seed):
    if method == "PatentFlow":
        base = ROOT / "results/main" / dataset / "v2_default" / f"s{seed}"
        return base, base / "validation.json"
    if method in CLASSICAL:
        base = ROOT / "results/baselines" / dataset / f"{method}_l3_s{seed}"
        return base, base / "metrics.json"
    base = ROOT / "results/deep" / dataset / method / "v2_default" / f"s{seed}"
    return base, base / "validation.json"


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def resource_segments(path: Path):
    if not path.exists():
        return []
    segments = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (not segments or record.get("pid") != segments[-1][-1].get("pid")
                or record["elapsed_s"] < segments[-1][-1]["elapsed_s"]):
            segments.append([])
        segments[-1].append(record)
    return segments


def sampled_cpu(path: Path, wall: float):
    chunks = resource_segments(path)
    if not chunks:
        return {}
    # An appended test run has a new PID and an elapsed-time reset. Pick the
    # segment whose observed duration matches the training artifact.
    chunk = min(chunks, key=lambda c: abs(c[-1]["elapsed_s"] - wall))
    coverage = chunk[-1]["elapsed_s"] / wall if wall > 0 else 0
    if abs(coverage - 1) > 0.15:
        return {"resource_samples": len(chunk), "resource_coverage_fraction": coverage}
    values = [as_float(item.get("cpu_percent")) for item in chunk[1:]]
    values = [v for v in values if v is not None]
    weighted_seconds = 0.0
    for previous, current in zip(chunk, chunk[1:]):
        dt = current["elapsed_s"] - previous["elapsed_s"]
        cpu = as_float(current.get("cpu_percent"))
        if dt > 0 and cpu is not None:
            weighted_seconds += dt * cpu / 100
    return {
        "resource_samples": len(chunk),
        "resource_coverage_fraction": coverage,
        "cpu_mean_sampled_percent": mean(values) if values else None,
        "cpu_p95_sampled_percent": percentile(values, 0.95),
        "cpu_max_sampled_percent": max(values) if values else None,
        "estimated_cpu_core_seconds": weighted_seconds,
    }


def epoch_stats(path: Path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    result = {
        "epochs_completed": len(rows),
        "epoch_seconds_sum": sum(float(row.get("seconds", 0)) for row in rows),
    }
    if any("train_seconds" in row for row in rows):
        result["train_seconds_explicit"] = sum(row.get("train_seconds", 0) for row in rows)
    if any("full_catalog_evaluation_seconds" in row.get("validation", {}) for row in rows):
        result["validation_seconds_explicit"] = sum(
            row.get("validation", {}).get("full_catalog_evaluation_seconds", 0) for row in rows)
    if any("replay_seconds" in row.get("validation", {}) for row in rows):
        result["validation_replay_seconds"] = sum(
            row.get("validation", {}).get("replay_seconds", 0) for row in rows)
    return result


def collect_run(dataset, method, seed):
    base, source = path_for(dataset, method, seed)
    if not source.exists():
        return None
    obj = read_json(source)
    wall = float(obj["wall_seconds"])
    resource = base / "resources.jsonl"
    epochs = base / "epochs.jsonl"
    config = obj.get("config", {})
    # The baseline artifact includes test evaluation; the neural validation
    # artifact includes training and post-selection validation, but no test.
    scope = ("fit_tune_validation_test" if method in {"SVM", "RF", "XGBoost"}
             else "validation_test_no_fit" if method in {"TOP", "PersonalTOP"}
             else "train_validation_no_test")
    row = {
        "dataset": dataset, "method": method,
        "family": "main" if method == "PatentFlow" else "classical" if method in CLASSICAL else "deep",
        "seed": seed, "scope": scope,
        "device": config.get("device", "cpu" if method in CLASSICAL else ""),
        "protocol_sha256": obj.get("protocol_sha256", ""),
        "validation_active_ndcg20": obj["splits"]["validation"]["active"]["ndcg_20"],
        "wall_seconds": wall, "wall_hours": wall / 3600,
        "peak_process_rss_gib": obj["peak_process_rss_bytes"] / GIB if obj.get("peak_process_rss_bytes") is not None else None,
        "peak_torch_gpu_allocated_gib": obj["gpu_peak_allocated_bytes"] / GIB if obj.get("gpu_peak_allocated_bytes") is not None else None,
        "configured_epochs": config.get("epochs"), "batch_size": config.get("batch_size"),
        "max_queries_per_year": config.get("max_queries_per_year"),
        "training_pair_count": obj.get("training_pair_count"),
        "classical_fit_seconds_explicit": (
            sum(t["fit_seconds"] for t in obj.get("trials", []) if "fit_seconds" in t)
            if method in {"SVM", "RF", "XGBoost"} else None
        ),
        "parameter_count": obj.get("parameter_count"), "best_epoch": obj.get("best_epoch"),
        "source_json": str(source.relative_to(ROOT)),
        "resource_jsonl": str(resource.relative_to(ROOT)) if resource.exists() else "",
        "epoch_jsonl": str(epochs.relative_to(ROOT)) if epochs.exists() else "",
        "source_sha256": digest(source), "resource_sha256": digest(resource),
        "epoch_sha256": digest(epochs),
    }
    row.update(sampled_cpu(resource, wall))
    row.update(epoch_stats(epochs))
    return row


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    output = []
    for (dataset, method), group in sorted(groups.items()):
        summary = {"dataset": dataset, "method": method, "family": group[0]["family"],
                   "n_seeds": len(group), "seed_list": ";".join(str(r["seed"]) for r in group),
                   "scope": group[0]["scope"]}
        for name in ("validation_active_ndcg20", "wall_seconds", "peak_process_rss_gib",
                     "peak_torch_gpu_allocated_gib", "cpu_mean_sampled_percent",
                     "estimated_cpu_core_seconds",
                     "epochs_completed", "train_seconds_explicit",
                     "classical_fit_seconds_explicit", "validation_seconds_explicit"):
            values = [as_float(r.get(name)) for r in group]
            values = [v for v in values if v is not None]
            summary[f"{name}_median"] = median(values) if values else None
            if name == "wall_seconds":
                summary["wall_seconds_min"] = min(values) if values else None
                summary["wall_seconds_max"] = max(values) if values else None
        output.append(summary)
    return output


def offline_costs():
    rows = []
    for dataset in DATASETS:
        country, part = dataset.split("_")
        slug = country.lower() if part == "All" else f"{country.lower()}_{part}"
        textpath = ROOT / "data/text" / country.lower() / "manifest.json"
        memorypath = ROOT / "data/memory" / slug / "f180_s1095_n1" / "manifest.json"
        if not textpath.exists() or not memorypath.exists():
            continue
        text, memory = read_json(textpath), read_json(memorypath)
        text_resource = textpath.parent / "resources.jsonl"
        text_cpu = sampled_cpu(text_resource, float(text["elapsed_seconds"]))
        rows.append({"dataset": dataset, "country": country,
                     "word2vec_shared_country_seconds": text["elapsed_seconds"],
                     "word2vec_shared_country_peak_rss_gib": text["peak_process_rss_bytes"] / GIB,
                     "word2vec_shared_country_cpu_mean_sampled_percent": text_cpu.get("cpu_mean_sampled_percent"),
                     "word2vec_shared_country_cpu_p95_sampled_percent": text_cpu.get("cpu_p95_sampled_percent"),
                     "word2vec_shared_country_estimated_cpu_core_seconds": text_cpu.get("estimated_cpu_core_seconds"),
                     "memory_view_seconds": memory["elapsed_seconds"],
                     "memory_view_peak_rss_gib": None,
                     "memory_gpu_allocated_gib": None,
                     "word2vec_manifest": str(textpath.relative_to(ROOT)),
                     "memory_manifest": str(memorypath.relative_to(ROOT)),
                     "word2vec_resource_jsonl": str(text_resource.relative_to(ROOT)) if text_resource.exists() else "",
                     "word2vec_sha256": digest(textpath),
                     "word2vec_resource_sha256": digest(text_resource),
                     "memory_sha256": digest(memorypath)})
    return rows


def make_charts(summaries, datasets, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    palette = plt.get_cmap("tab20")
    colors = {method: palette(i) for i, method in enumerate(METHODS)}
    shapes = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "8", "p", "*", "d", "H"]
    markers = dict(zip(METHODS, shapes))
    def size_for_seconds(wall):
        return 45 + 35 * math.log10(max(wall, 1))
    charts = [
        ("wall_seconds_median", "wall_time", "Run wall time (seconds; log scale)"),
        ("peak_torch_gpu_allocated_gib_median", "gpu_allocated", "Peak PyTorch GPU allocated (GiB)"),
        ("peak_process_rss_gib_median", "process_rss", "Peak process RSS (GiB)"),
        ("cpu_mean_sampled_percent_median", "cpu_percent", "Mean sampled process CPU (%)"),
        ("estimated_cpu_core_seconds_median", "cpu_core_seconds", "Estimated CPU core-seconds (sampled; log scale)"),
    ]
    for dataset in datasets:
        members = [r for r in summaries if r["dataset"] == dataset]
        if not members:
            continue
        for xfield, suffix, xlabel in charts:
            fig, ax = plt.subplots(figsize=(11.2, 6.1))
            fig.subplots_adjust(left=0.13, right=0.72, bottom=0.19, top=0.90)
            plotted = []
            for method in METHODS:
                row = next((r for r in members if r["method"] == method), None)
                if row is None:
                    continue
                x = as_float(row.get(xfield))
                y = as_float(row.get("validation_active_ndcg20_median"))
                wall = as_float(row.get("wall_seconds_median"))
                if x is None or y is None or wall is None:
                    continue
                size = size_for_seconds(wall)
                ax.scatter(x, y, s=size, marker=markers[method], color=colors[method],
                           edgecolors="#222222", linewidths=0.55, alpha=0.87, zorder=3)
                plotted.append(method)
            if suffix in {"wall_time", "cpu_core_seconds"}:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Validation active NDCG@20")
            ax.set_title(f"{dataset}: quality and {suffix.replace('_', ' ')}")
            ax.grid(alpha=0.24)
            handles = [Line2D([0], [0], marker=markers[m], linestyle="", color="w",
                              markerfacecolor=colors[m], markeredgecolor="#222222",
                              markersize=8, label=m) for m in plotted]
            method_legend = ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                                      fontsize=8, frameon=False, title="Method")
            ax.add_artist(method_legend)
            size_handles = [ax.scatter([], [], s=size_for_seconds(w), facecolors="none",
                                       edgecolors="#444444", linewidths=0.8,
                                       label=label) for w, label in
                            [(60, "1 min"), (3600, "1 h"), (36000, "10 h")]]
            ax.legend(handles=size_handles, bbox_to_anchor=(1.01, 0), loc="lower left",
                      fontsize=8, frameon=False, title="Wall time / bubble area")
            ax.text(0, -0.22, "Marker area encodes log10(wall seconds). Classical wall time includes test; GPU excludes classical (unmeasured).",
                    transform=ax.transAxes, fontsize=8)
            fig.savefig(out / f"{dataset}_{suffix}.png", dpi=180, bbox_inches="tight")
            fig.savefig(out / f"{dataset}_{suffix}.svg", bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="+", default=["CN_All", "US_All", "IN_All"],
                        help="Datasets to plot; CSV always covers all 12 views")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = [row for dataset in DATASETS for method in METHODS for seed in SEEDS
            if (row := collect_run(dataset, method, seed)) is not None]
    summary = aggregate(runs)
    protocols = {r["protocol_sha256"] for r in runs}
    if len(protocols) != 1:
        raise ValueError(f"Mixed protocols: {protocols}")
    write_csv(args.out / "run_level.csv", runs, RUN_COLUMNS)
    write_csv(args.out / "view_method_median.csv", summary, SUMMARY_COLUMNS)
    offline = offline_costs()
    write_csv(args.out / "offline_costs.csv", offline,
              ("dataset", "country", "word2vec_shared_country_seconds",
               "word2vec_shared_country_peak_rss_gib",
               "word2vec_shared_country_cpu_mean_sampled_percent",
               "word2vec_shared_country_cpu_p95_sampled_percent",
               "word2vec_shared_country_estimated_cpu_core_seconds",
               "memory_view_seconds",
               "memory_view_peak_rss_gib", "memory_gpu_allocated_gib",
               "word2vec_manifest", "memory_manifest", "word2vec_resource_jsonl",
               "word2vec_sha256", "word2vec_resource_sha256", "memory_sha256"))
    missing = []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                _, source = path_for(dataset, method, seed)
                if not source.exists():
                    missing.append({"dataset": dataset, "method": method,
                                    "seed": seed, "expected_training_artifact": str(source.relative_to(ROOT)),
                                    "reason": "No comparable completed training-run artifact"})
    write_csv(args.out / "missing_training_runs.csv", missing,
              ("dataset", "method", "seed", "expected_training_artifact", "reason"))
    make_charts(summary, args.datasets, args.out)
    print(f"{len(runs)} runs, {len(summary)} method/view cells, {len(offline)} offline views")
    print(f"{len(missing)} missing training artifacts")
    print(f"protocol: {next(iter(protocols))}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
