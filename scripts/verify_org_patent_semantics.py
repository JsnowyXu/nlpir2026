#!/usr/bin/env python
"""Verify Top-5 organization/patent semantic figures and stored coordinates."""
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['PATENTFLOW_WORKSPACE'] = str(ROOT)
sys.path.insert(0, str(ROOT / 'main_experiment/src'))
import numpy as np
from patentflow_main.dataset import DatasetView

DATASETS = [c + '_' + s for c in ['CN', 'IN', 'US'] for s in ['All', 'B', 'C', 'F']]


def main():
    report = ROOT / 'reports/memory_evolution'
    summary = list(csv.DictReader((report / 'org_patent_semantics_summary.csv').open(encoding='utf-8-sig')))
    assert len(summary) == 60
    checks = []
    for dataset in DATASETS:
        folder = report / dataset / 'org_patent_semantics'
        for suffix in ['png', 'pdf', 'npz']:
            assert len(list(folder.glob('*.' + suffix))) == 5, (dataset, suffix)
        selection = [r for r in csv.DictReader((report / dataset / 'selection.csv').open(encoding='utf-8-sig'))
                     if r['kind'] == 'org']
        data = DatasetView(dataset)
        slug = data.spec['country'] if data.spec['section'] == 'All' else data.spec['country'] + '_' + data.spec['section']
        patent_vectors = np.load(ROOT / 'data/text' / data.spec['country'] / 'patent_vectors.npy', mmap_mode='r')
        for selected, path in zip(selection, sorted(folder.glob('*.npz'))):
            assert path.name.startswith('%02d_id%s' % (int(selected['rank']), selected['id']))
            with np.load(path, allow_pickle=False) as result:
                assert result['org'].shape == (len(result['years']), 100)
                assert result['patent_xy'].shape == (len(result['patent_ids']), 2)
                assert result['patent_offsets'][0] == 0 and result['patent_offsets'][-1] == len(result['patent_ids'])
                assert len(result['patent_offsets']) == len(result['years']) + 1
                np.testing.assert_allclose(result['pca_basis'] @ result['pca_basis'].T, np.eye(2), atol=1e-6)
                np.testing.assert_allclose(np.linalg.norm(result['org'], axis=1), 1, atol=1e-6)
                # Independently check first company vector against raw slow-text memory.
                year = int(result['years'][0]); org_id = int(selected['id'])
                with np.load(ROOT / 'data/memory' / slug / 'f180_s1095_n1' / ('%d.npz' % year)) as snap:
                    raw = snap['org_slow'][org_id, :100]
                raw = raw / np.linalg.norm(raw)
                np.testing.assert_allclose(result['org'][0], raw, atol=1e-6)
                # Projection of one raw patent must match stored PCA coordinates.
                pid = int(result['patent_ids'][0]); raw_patent = patent_vectors[pid].astype(np.float64)
                raw_patent /= np.linalg.norm(raw_patent)
                projected = (raw_patent - result['pca_mean']) @ result['pca_basis'].T
                np.testing.assert_allclose(result['patent_xy'][0], projected, atol=1e-6)
        checks.append({'dataset': dataset, 'figures': 5, 'status': 'passed'})
        print('PASS', dataset)
    (report / 'org_patent_semantics_verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
