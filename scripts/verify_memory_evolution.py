#!/usr/bin/env python
"""Validate generated artifacts and independently check checkpoint replay."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from visualize_memory_evolution import ROOT, DatasetView, PatentFlow, anchors_for, metrics, unit, digest
from patentflow_main.neural_data import MemoryFeatures


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'reports/memory_evolution')
    args = p.parse_args()
    torch.set_num_threads(2)
    checks = []
    for audit_path in sorted(args.output.glob('*/audit.json')):
        out = audit_path.parent
        audit = json.loads(audit_path.read_text())
        name = audit['dataset']
        arrays = {s: np.load(out / ('vectors/%s.npz' % s)) for s in ['memory', 'random', 'trained']}
        for state, data in arrays.items():
            for kind in ['org', 'ipc']:
                assert data[kind].shape[:2] == (len(data['years']), len(data[kind + '_ids']))
                assert np.isfinite(data[kind]).all()
                if state != 'memory':
                    np.testing.assert_allclose(np.linalg.norm(data[kind], axis=-1), 1, atol=1e-5)
        for path in out.glob('*/*.npz'):
            if path.parent.name == 'vectors':
                continue
            with np.load(path) as projection:
                assert np.isfinite(projection['xy']).all()
                np.testing.assert_allclose(projection['pca_basis'] @ projection['pca_basis'].T, np.eye(2), atol=1e-6)
        assert len(list((out / 'memory_org').glob('*.png'))) == 5
        assert len(list((out / 'memory_ipc').glob('*.png'))) == 10
        assert len(list((out / 'comparison_org').glob('*.png'))) == 3
        assert len(list((out / 'comparison_ipc').glob('*.png'))) == 3
        for png in out.rglob('*.png'):
            assert png.with_suffix('.pdf').is_file()
            assert png.stat().st_size > 10000
        # Independent full MemoryFeatures construction catches catalog/order errors.
        data = DatasetView(name)
        cfg = audit['config']
        year = data.spec['validation_year']
        features = MemoryFeatures(data, year, 'cpu', cfg['fast'], cfg['slow'], cfg['novelty'])
        model = PatentFlow(anchors_for(data), dim=cfg['dim'], communities=cfg['communities'],
                           dropout=cfg['dropout'], disabled=cfg['disable'], temperature=cfg['temperature']).eval()
        checkpoint = ROOT / 'results/main' / name / cfg['tag'] / ('s%d' % cfg['seed']) / 'best.pt'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
        stored = arrays['trained']
        t = np.flatnonzero(stored['years'] == year)[0]
        nodes = stored['org_ids']
        with torch.no_grad():
            org, ipc = model.export_embeddings(torch.tensor(features.org_raw[nodes]), features.ipc_raw)
        candidate_leaves = [data.all_leaf_codes.index(data.codes[i]) for i in data.candidate_ids]
        positions = [candidate_leaves.index(i) for i in stored['ipc_ids']]
        np.testing.assert_allclose(stored['org'][t], org.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(stored['ipc'][t], ipc.numpy()[positions], rtol=1e-5, atol=1e-6)
        checks.append(dict(dataset=name, png_count=len(list(out.rglob('*.png'))),
                           replay_year=year, checkpoint_replay='passed'))
        print('PASS', name, flush=True)
    assert checks, 'No dataset artifacts found'
    # Known geometric examples, including stationary vectors excluded from deltas.
    x = np.array([[1., 0], [1., 1], [0., 1]])
    result = metrics(x, x, np.array([2010, 2011, 2012]))
    np.testing.assert_allclose(result[:2], [1, 1]); assert result[2] == 2
    result = metrics(np.ones((3, 2)), np.ones((3, 2)), np.arange(3))
    assert result[1] is None and result[2] == 0
    (args.output / 'verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    from collections import Counter
    inventory = dict(main_npz_filenames=dict(Counter(path.name for path in (ROOT / 'results/main').rglob('*.npz'))))
    existing = ROOT / 'draft/patentflow/data/case_embeddings.npz'
    if existing.exists():
        meta = json.loads(existing.with_name('case_metadata.json').read_text())
        with np.load(existing) as case:
            inventory['existing_draft_case'] = dict(path=str(existing.relative_to(ROOT)), sha256=digest(existing),
                years=case['years'].tolist(), shapes={k: list(case[k].shape) for k in case.files}, metadata=meta)
        checkpoint = ROOT / 'results/main/US_All/v2_default/s42/best.pt'
        inventory['existing_draft_case']['checkpoint_hash_matches'] = digest(checkpoint) == meta['checkpoint_sha256']
    (args.output / 'saved_vectors_audit.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
    report = args.output / 'README.md'
    if report.exists():
        heading = '\n## 补充：已保存的论文案例与独立核验\n'
        text = report.read_text().split(heading)[0]
        text += heading + '\n主实验 `results/main` 下只有 validation_queries.npz，没有批量年度节点向量。'
        if 'existing_draft_case' in inventory:
            text += ('但论文草稿已有 `draft/patentflow/data/case_embeddings.npz`：US_All 的 '
                     '`kinpo electronics, inc.`（org ID 12542），2010–2017 年，1 个 org 和 4 个 IPC '
                     '（B29C、B33Y、B65H、G06F），均为 100 维训练后向量。其配套脚本为 '
                     '`draft/patentflow/tools/build_case_study.py`。该案例按训练期组合变化幅度选取，并非本次 Top 活动量规则；'
                     '本次保留该文件不变，另行统一重建。来源及哈希见 `saved_vectors_audit.json`。\n')
        text += '\n独立核验已通过 %d 个数据集：图表完整性、有限值、PCA 正交性、模型向量单位范数，以及通过原生 MemoryFeatures 对完整候选集合重算的向量一致性。结果见 `verification.json`。\n' % len(checks)
        summary = args.output / 'cooperation_summary.csv'
        if summary.exists():
            rows = list(csv.DictReader(summary.open(encoding='utf-8-sig')))
            selected = [r for r in rows if r['state'] == 'trained' and r['kind'] == 'collaboration' and r['mean_delta_cosine']]
            positive = sum(float(r['mean_delta_cosine']) > 0 for r in selected)
            stronger, comparable = 0, 0
            for name in sorted(set(r['dataset'] for r in selected)):
                values = {}
                for kind in ['collaboration', 'control']:
                    values[kind] = [float(r['mean_delta_cosine']) for r in rows if r['dataset'] == name and r['state'] == 'trained' and r['kind'] == kind and r['mean_delta_cosine']]
                if all(values.values()):
                    comparable += 1
                    stronger += np.mean(values['collaboration']) > np.mean(values['control'])
            text += '\n### 对合作趋势的观察\n\n'
            text += '训练后选中案例中，%d/%d 组的平均年度位移余弦为正；在 %d/%d 个可比较视图中，合作案例均值高于匹配对照。各视图存在重叠，这些计数不是独立样本的显著性检验。\n' % (positive, len(selected), stronger, comparable)
            robust = sorted([r for r in selected if int(r['valid_transitions']) >= 3], key=lambda r: float(r['mean_delta_cosine']))
            if robust:
                text += '\n以下列举预选案例中方向最一致与最不一致的各两例（至少 3 个有效区间），展示差异而非重新挑选样本：\n\n| 数据集 | 公司 A | 公司 B | 训练后位移余弦 | 有效区间 |\n|---|---|---|---:|---:|\n'
                for r in robust[-2:][::-1] + robust[:2]:
                    text += '| %s | %s | %s | %.3f | %s |\n' % (r['dataset'], r['name_a'], r['name_b'], float(r['mean_delta_cosine']), r['valid_transitions'])
            text += '\n因此应区分“技术状态接近”与“逐年变化方向接近”，不能只凭共同专利或二维轨迹断言所有合作公司都会同步演化。\n'
        report.write_text(text)


if __name__ == '__main__':
    main()
