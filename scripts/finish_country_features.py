#!/usr/bin/env python
"""Wait for complete text manifest, then build four causal memory views."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from patentflow.data import ROOT

p=argparse.ArgumentParser();p.add_argument('--country',required=True,choices=['cn','us','in'])
a=p.parse_args()
lock=(ROOT/'state'/('finish_features_'+a.country+'.lock')).open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (ROOT/'data/text'/a.country/'manifest.json').exists():
    time.sleep(15)
for section in ['All','B','F','C']:
    subprocess.run([sys.executable,'-u','scripts/build_memory.py','--languages',a.country,
                    '--section',section],cwd=str(ROOT),check=True)
subprocess.run([sys.executable,'scripts/project_status.py'],cwd=str(ROOT),check=True)
