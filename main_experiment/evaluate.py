#!/usr/bin/env python
"""Evaluate a selected checkpoint without training or modifying its run."""
from _bootstrap import configure

if __name__ == "__main__":
    configure()
    from patentflow_main.checkpoint_evaluation import cli

    cli()
