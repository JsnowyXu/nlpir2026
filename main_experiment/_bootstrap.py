"""Resolve storage options before importing the standalone package."""

import argparse
import os
from pathlib import Path
import sys


def configure():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace")
    parser.add_argument("--results-root")
    known, remaining = parser.parse_known_args()
    if known.workspace:
        os.environ["PATENTFLOW_WORKSPACE"] = str(
            Path(known.workspace).expanduser().resolve()
        )
    if known.results_root:
        os.environ["PATENTFLOW_RESULTS_ROOT"] = str(
            Path(known.results_root).expanduser().resolve()
        )
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    sys.argv = [sys.argv[0]] + remaining
