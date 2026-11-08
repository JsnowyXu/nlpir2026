#!/usr/bin/env python
"""Compatibility entry point; all main training lives in main_experiment/."""
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    entry = Path(__file__).resolve().parents[1] / 'main_experiment/train.py'
    os.execv(sys.executable, [sys.executable, str(entry)] + sys.argv[1:])
