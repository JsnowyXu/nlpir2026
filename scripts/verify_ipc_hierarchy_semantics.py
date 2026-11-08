#!/usr/bin/env python
"""Verify the CN_All IPC hierarchy semantic artifact."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'reports/memory_evolution/CN_All/ipc_hierarchy_semantics/ipc_hierarchy_original_vs_2019'
with np.load(base.with_suffix('.npz')) as data:
    assert len(data['codes']) == 747
    assert data['raw'].shape == data['final_aligned'].shape == (747, 100)
    assert data['raw_xy'].shape == data['final_xy'].shape == (747, 2)
    assert {1: 8, 2: 123, 3: 616} == {i: int((data['levels'] == i).sum()) for i in [1, 2, 3]}
    assert all(len(code) == {1: 1, 2: 3, 3: 4}[int(level)] for code, level in zip(data['codes'], data['levels']))
    np.testing.assert_allclose(np.linalg.norm(data['raw'], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(data['final_aligned'], axis=1), 1, atol=1e-6)
metrics = json.loads(base.with_name('metrics.json').read_text())
assert metrics['final_year'] == 2019 and metrics['seed'] == 42
assert 0 <= metrics['raw_knn_section_purity'] <= 1
assert 0 <= metrics['final_knn_section_purity'] <= 1
for suffix in ['.png', '.pdf', '.npz']:
    assert base.with_suffix(suffix).stat().st_size > 10000
print('PASS CN_All IPC hierarchy original vs 2019')
