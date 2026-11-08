"""Compatibility import; edit main_experiment/src/patentflow_main/model.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'main_experiment/src'))
from patentflow_main.model import PatentFlow, projection
