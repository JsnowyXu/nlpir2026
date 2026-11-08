#!/usr/bin/env python
"""Wait for all raw inputs to close and settle; a receipt is not a CSV audit."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from patentflow.data import ROOT, atomic_json, taxonomy_paths
from patentflow.input_gate import snapshot, writable_handles

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--settle-seconds', type=int, default=180)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    paths = [ROOT / 'datasets' / (lang.upper() + '-all.csv') for lang in ['cn', 'us', 'in']]
    for lang in ['cn', 'us', 'in']:
        paths.extend(taxonomy_paths(lang))
    previous, unchanged_since = None, time.monotonic()
    while True:
        current, writers = snapshot(paths), writable_handles(paths)
        if current != previous or writers or any(v is None or not v['size'] for v in current.values()):
            unchanged_since = time.monotonic()
        stable_for = time.monotonic() - unchanged_since
        status = {'observed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': current,
                  'writers': writers, 'stable_seconds': stable_for,
                  'ready_for_full_scan': stable_for >= args.settle_seconds}
        atomic_json(ROOT / 'state/input_transfer.json', status)
        print(status['observed_at'], 'stable_seconds', int(stable_for), 'writers', len(writers),
              'raw_sizes', [current[str(p)]['size'] if current[str(p)] else None for p in paths[:3]], flush=True)
        if status['ready_for_full_scan'] or args.once:
            break
        previous = current
        time.sleep(30)
