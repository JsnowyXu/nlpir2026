#!/usr/bin/env python
"""Refresh the progress report while experiment supervisors are active."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
lock = (ROOT / 'state/progress_watcher.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
while True:
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/project_status.py')],
                            cwd=str(ROOT), stdout=subprocess.DEVNULL)
    if result.returncode:
        print('Progress refresh failed:', result.returncode, flush=True)
    else:
        progress = json.loads((ROOT / 'state/progress.json').read_text())
        campaigns = progress['campaigns']
        pending = sum(value['jobs'] - value['states'].get('complete', 0) - value['states'].get('failed', 0)
                      for value in campaigns.values())
        print(progress['updated'], 'pending_jobs', pending, flush=True)
        if not pending:
            break
    time.sleep(60)
