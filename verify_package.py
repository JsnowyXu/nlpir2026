#!/usr/bin/env python3
"""Check the local package's essential raw, prepared and selected model files."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
selection = json.loads((root / 'checkpoints/selection.json').read_text())
missing = []
for country in ('CN', 'US', 'IN'):
    for path in [root / 'datasets' / (country + '-all.csv'), root / 'data/processed' / country.lower() / 'manifest.json', root / 'data/text' / country.lower() / 'taxonomy_anchors.npz']:
        if not path.is_file(): missing.append(str(path.relative_to(root)))
for name, record in selection.items():
    view_path = root / 'data/processed/views' / (name + '.json')
    for path in [view_path, root / record['checkpoint'], root / 'checkpoints' / name / 'config.json']:
        if not path.is_file(): missing.append(str(path.relative_to(root)))
    if view_path.is_file():
        view = json.loads(view_path.read_text())
        slug = view['country'] if view['section'] == 'All' else view['country'] + '_' + view['section']
        cutoff = view['test_year'] - 1
        path = root / 'data/memory' / slug / 'f180_s1095_n1' / (str(cutoff) + '.npz')
        if not path.is_file(): missing.append(str(path.relative_to(root)))
if missing:
    print('Missing files:\n' + '\n'.join(missing))
    raise SystemExit(1)
print('Package has raw data, prepared test snapshots and selected checkpoints for', len(selection), 'views')
