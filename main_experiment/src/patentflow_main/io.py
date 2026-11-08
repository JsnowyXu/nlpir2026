"""Prepared-artifact locations and reproducibility records; no preprocessing."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = (
    Path(os.environ.get("PATENTFLOW_WORKSPACE", PACKAGE_ROOT.parent))
    .expanduser()
    .resolve()
)
RESULTS_ROOT = (
    Path(os.environ.get("PATENTFLOW_RESULTS_ROOT", ROOT / "results/main"))
    .expanduser()
    .resolve()
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    temporary.replace(path)


def record_run_provenance(out, data, config):
    """Preserve each invocation's exact code without changing model RNG state."""
    source_paths = list(sorted((PACKAGE_ROOT / "src/patentflow_main").glob("*.py")))
    source_paths += [
        PACKAGE_ROOT / name for name in ["train.py", "evaluate.py", "_bootstrap.py"]
    ]
    contents = {
        str(path.relative_to(PACKAGE_ROOT)): path.read_bytes() for path in source_paths
    }
    sources = {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    snapshot = out / "source" / digest
    for name in sources:
        target = snapshot / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[name])
    slug = data.spec["country"] + (
        "" if data.spec["section"] == "All" else "_" + data.spec["section"]
    )
    memory = (
        ROOT
        / "data/memory"
        / slug
        / ("f%d_s%d_n%g" % (config["fast"], config["slow"], config["novelty"]))
    )
    paths = [
        data.path / "manifest.json",
        ROOT / "data/processed/views" / (config["dataset"] + ".json"),
        ROOT / "data/text" / data.spec["country"] / "manifest.json",
        memory / "manifest.json",
    ]
    provenance = {
        "timestamp_unix": time.time(),
        "entrypoint": str(PACKAGE_ROOT / "train.py"),
        "workspace": str(ROOT),
        "results_root": str(RESULTS_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "source_sha256": digest,
        "source_files": sources,
        "prepared_manifests": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
        "protocol_sha256": data.manifest["protocol_sha256"],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "scipy", "psutil"]
        },
    }
    with (out / "invocations.jsonl").open("a") as handle:
        handle.write(json.dumps(provenance) + "\n")
    atomic_json(out / "source_manifest.json", provenance)
