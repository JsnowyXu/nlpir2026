#!/usr/bin/env python
"""Canonical entry point for all main, ablation and sensitivity training."""
from _bootstrap import configure

if __name__ == "__main__":
    configure()
    from patentflow_main.train import cli

    cli()
