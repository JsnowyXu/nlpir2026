#!/usr/bin/env python
"""Reproducible annual memory, checkpoint and collaboration case studies.

Run with .venv-evolution/bin/python scripts/visualize_memory_evolution.py.
Only writes to --output; never modifies training runs or source data.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
os.environ['PATENTFLOW_WORKSPACE'] = str(ROOT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path[:0] = [str(ROOT / 'main_experiment/src'), str(ROOT / 'src')]
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from patentflow_main.dataset import DatasetView
from patentflow_main.neural_data import anchors_for
from patentflow_main.model import PatentFlow
from patentflow.entities import entity_type


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def csvwrite(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pca(x):
    """One deterministic PCA for all years/entities passed in; never per year."""
    shape = x.shape
    x = x.reshape(-1, shape[-1]).astype(np.float64)
    mean = x.mean(0)
    u, s, v = np.linalg.svd(x - mean, full_matrices=False)
    for i in range(min(2, len(v))):
        if v[i, np.argmax(abs(v[i]))] < 0:
            v[i] *= -1
    xy = (x - mean) @ v[:2].T
    return xy.reshape(*shape[:-1], 2), float((s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-20)), mean, v[:2]


def fireworks(ax, years, xy, title, variance, label=None, marker='o', linestyle='-'):
    colors = plt.colormaps['viridis'](np.linspace(.08, .92, len(years)))
    origin = xy[0]
    for i in range(1, len(years)):
        ax.annotate('', xy[i], origin, arrowprops=dict(arrowstyle='->', color=colors[i], alpha=.40, lw=1.2))
    ax.plot(xy[:, 0], xy[:, 1], linestyle, lw=1, alpha=.65, label=label)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, marker=marker, s=45, zorder=3)
    for i, y in enumerate(years):
        annotation = ax.annotate(str(y), xy[i], xytext=(4, 5), textcoords='offset points', fontsize=7,
                                 arrowprops=dict(arrowstyle='-', lw=.4, color='0.55'))
        annotation.set_gid('year')
    ax.scatter(*origin, marker='*', color='#dd5b35', s=120, zorder=4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.grid(alpha=.15); ax.set_aspect('equal', adjustable='datalim')
    if not any(t.get_gid() == 'variance' for t in ax.texts):
        ax.text(.02, .02, 'PCA解释率 %.1f%%' % (100 * variance), transform=ax.transAxes, fontsize=8, gid='variance')


def savefig(fig, base):
    # Place year labels greedily in display coordinates, with leader lines.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.text import Text
    for ax in fig.axes:
        occupied = []
        if ax.get_legend() is not None:
            occupied.append(ax.get_legend().get_window_extent(renderer))
        bounds = ax.get_window_extent(renderer)
        labels = [t for t in ax.texts if t.get_gid() == 'year']
        for label in labels:
            best, best_cost = (4, 5), float('inf')
            for radius in [8, 18, 30, 44, 60, 78]:
                for dx, dy in [(radius, radius / 2), (radius, -radius), (-radius - 20, radius / 2),
                               (-radius - 20, -radius), (-12, radius), (-12, -radius - 8)]:
                    label.set_position((dx, dy))
                    label.update_positions(renderer)
                    box = Text.get_window_extent(label, renderer).expanded(1.12, 1.25)
                    overlaps = sum(box.overlaps(other) for other in occupied)
                    outside = not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1))
                    cost = overlaps * 10000 + outside * 10000 + abs(dx) + abs(dy)
                    if cost < best_cost:
                        best, best_cost = (dx, dy), cost
                if best_cost < 10000:
                    break
            label.set_position(best); label.update_positions(renderer)
            occupied.append(Text.get_window_extent(label, renderer).expanded(1.12, 1.25))
    fig.savefig(base.with_suffix('.png'), dpi=160, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def metrics(a, b, years):
    valid = (np.linalg.norm(a, axis=1) > 1e-10) & (np.linalg.norm(b, axis=1) > 1e-10)
    a, b = unit(a), unit(b)
    state = (a * b).sum(1)
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    good = valid[1:] & valid[:-1] & (np.diff(years) == 1)
    good &= (np.linalg.norm(da, axis=1) > 1e-8) & (np.linalg.norm(db, axis=1) > 1e-8)
    trend = (unit(da) * unit(db)).sum(1)
    return (float(state[valid].mean()) if valid.any() else None,
            float(trend[good].mean()) if good.any() else None,
            int(good.sum()), state, trend, good)


def select_pairs(data, names, activity, train_end):
    """Exact shared patents inside the view, among top 200 active companies."""
    pool = [int(i) for i in np.argsort(-activity, kind='stable')
            if activity[i] > 0 and entity_type(names[i]) == 'company'][:200]
    with np.load(data.path / 'events.npz') as e:
        src, dst, year, pid = [e[k] for k in ['src', 'dst', 'year', 'patent_row']]
    keep = (year <= train_end) & np.isin(dst, data.spec['subclass_ids']) & np.isin(src, pool)
    lookup = np.full(len(names), -1, dtype=int); lookup[pool] = np.arange(len(pool))
    _, columns = np.unique(pid[keep], return_inverse=True)
    incidence = sparse.coo_matrix((np.ones(keep.sum(), dtype=np.int64),
                                  (lookup[src[keep]], columns)),
                                 shape=(len(pool), int(columns.max()) + 1)).tocsr()
    incidence.data[:] = 1  # patent with several IPCs counts only once per org
    shared = (incidence @ incidence.T).toarray()
    candidates = [(int(shared[i, j]), pool[i], pool[j]) for i in range(len(pool))
                  for j in range(i + 1, len(pool)) if shared[i, j] > 0]
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    selected, used = [], set()
    for count, a, b in candidates:
        if a in used or b in used:
            continue
        selected.append(dict(a=a, b=b, shared_patents=count, kind='collaboration'))
        used.update([a, b])
        if len(selected) == 3:
            break
    # Matched negative examples: retain first org, nearest activity nonpartner.
    for p in list(selected):
        i = pool.index(p['a'])
        choices = [c for j, c in enumerate(pool) if c != p['a'] and shared[i, j] == 0]
        if choices:
            c = min(choices, key=lambda c: (abs(np.log1p(activity[c]) - np.log1p(activity[p['b']])), c))
            selected.append(dict(a=p['a'], b=c, shared_patents=0, kind='control'))
    return selected, len(pool), len(candidates)


def run_dataset(name, args):
    print('START', name, flush=True)
    data = DatasetView(name)
    country = data.spec['country']; section = data.spec['section']
    slug = country if section == 'All' else country + '_' + section
    memory = ROOT / 'data/memory' / slug / 'f180_s1095_n1'
    out = args.output / name; out.mkdir(parents=True, exist_ok=True)
    for folder in ['memory_org', 'memory_ipc', 'comparison_org', 'comparison_ipc', 'cooperation', 'vectors']:
        (out / folder).mkdir(exist_ok=True)
    names = json.loads((data.path / 'organizations.json').read_text())
    catalog = data.all_leaf_codes
    history = data.history(data.spec['train_end_year'])
    org_activity = np.asarray(history.sum(1)).ravel()
    ipc_activity = np.asarray(history.sum(0)).ravel()
    top_org = np.argsort(-org_activity, kind='stable')[:5].tolist()
    candidate_order = sorted(data.candidate_ids, key=lambda i: (-ipc_activity[i], i))[:10]
    top_ipc = [catalog.index(data.codes[i]) for i in candidate_order]
    pairs, pool_size, pair_count = select_pairs(data, names, org_activity, data.spec['train_end_year'])
    selected_org = sorted(set(top_org + [p[k] for p in pairs for k in ['a', 'b']]))
    selected_ipc = sorted(top_ipc)
    oi = {v: i for i, v in enumerate(selected_org)}; ii = {v: i for i, v in enumerate(selected_ipc)}
    years = np.array(sorted(int(p.stem) for p in memory.glob('[0-9]*.npz')))
    raw_org, raw_ipc = [], []
    source_hashes = {}
    for year in years:
        path = memory / ('%d.npz' % year)
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
        with np.load(path) as snap:
            raw_org.append(np.concatenate([snap['org_fast'][selected_org], snap['org_slow'][selected_org]], 1))
            raw_ipc.append(np.concatenate([snap['ipc_fast'][selected_ipc], snap['ipc_slow'][selected_ipc]], 1))
    raw_org, raw_ipc = np.stack(raw_org), np.stack(raw_ipc)
    np.savez_compressed(out / 'vectors/memory.npz', years=years, org_ids=selected_org,
                        ipc_ids=selected_ipc, org=raw_org, ipc=raw_ipc)
    selections = []
    for kind, ids, arr, index, labels in [('org', top_org, raw_org, oi, names), ('ipc', top_ipc, raw_ipc, ii, catalog)]:
        for rank, entity in enumerate(ids, 1):
            x = arr[:, index[entity]]
            valid = np.linalg.norm(x, axis=1) > 1e-10
            xy, var, mean, basis = pca(unit(x[valid]))
            base = out / ('memory_' + kind) / ('%02d_id%d' % (rank, entity))
            fig, ax = plt.subplots(figsize=(7.4, 6))
            fireworks(ax, years[valid], xy, '%s | %s Top%d\n%s' % (name, kind, rank, '\n'.join(textwrap.wrap(labels[entity], 48))), var)
            fig.text(.5, .01, '原始 fast+slow memory；星号为首个有效年份；射线从首年指向各年，折线按年连接', ha='center', fontsize=8)
            savefig(fig, base)
            np.savez_compressed(base.with_suffix('.npz'), years=years[valid], xy=xy, pca_mean=mean, pca_basis=basis)
            score = org_activity[entity] if kind == 'org' else ipc_activity[data.codes.index(catalog[entity])]
            selections.append(dict(dataset=name, kind=kind, rank=rank, id=entity, label=labels[entity],
                                   entity_type=entity_type(labels[entity]) if kind == 'org' else 'IPC subclass',
                                   train_activity=float(score), first_valid_year=int(years[valid][0])))
    csvwrite(out / 'selection.csv', selections)
    run = ROOT / 'results/main' / name / args.tag / ('s%d' % args.seed)
    cfg = json.loads((run / 'config.json').read_text())
    assert (cfg['level'], cfg['fast'], cfg['slow'], cfg['novelty']) == (3, 180, 1095, 1.0)
    torch.manual_seed(cfg['seed'])
    untrained = PatentFlow(anchors_for(data), dim=cfg['dim'], communities=cfg['communities'],
                          dropout=cfg['dropout'], disabled=cfg['disable'], temperature=cfg['temperature']).eval()
    import copy
    trained = copy.deepcopy(untrained)
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    trained.load_state_dict(checkpoint['model']); trained.eval()
    candidates_global = [catalog.index(data.codes[i]) for i in data.candidate_ids]
    ipc_positions = [candidates_global.index(i) for i in selected_ipc]
    model_org, model_ipc = {}, {}
    for label, model in [('random', untrained), ('trained', trained)]:
        orgs, ipcs = [], []
        with torch.no_grad():
            for t in range(len(years)):
                # Dynamic IPC projection is rowwise, but anchors have full candidate shape.
                full_ipc = np.zeros((len(candidates_global), 400), dtype=np.float32)
                full_ipc[ipc_positions] = raw_ipc[t]
                org, ipc = model.export_embeddings(torch.tensor(raw_org[t]), torch.tensor(full_ipc))
                orgs.append(org.numpy()); ipcs.append(ipc.numpy()[ipc_positions])
        model_org[label], model_ipc[label] = np.stack(orgs), np.stack(ipcs)
        np.savez_compressed(out / ('vectors/%s.npz' % label), years=years, org_ids=selected_org,
                            ipc_ids=selected_ipc, org=model_org[label], ipc=model_ipc[label])
    trajectory_rows = []
    for kind, ids, raw, models, index, labels in [('org', top_org[:3], raw_org, model_org, oi, names), ('ipc', top_ipc[:3], raw_ipc, model_ipc, ii, catalog)]:
        for rank, entity in enumerate(ids, 1):
            valid = (np.linalg.norm(raw[:, index[entity]], axis=1) > 1e-10) & (years < data.spec['test_year'])
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax, state, title in zip(axes, ['memory', 'random', 'trained'], ['原始 memory', '未训练模型（随机初始化）', '已训练模型（最佳 checkpoint）']):
                x = unit((raw if state == 'memory' else models[state])[valid, index[entity]])
                xy, var, mean, basis = pca(x)
                fireworks(ax, years[valid], xy, title, var)
                np.savez_compressed(out / ('comparison_%s/%02d_id%d_%s.npz' % (kind, rank, entity, state)), years=years[valid], xy=xy, pca_mean=mean, pca_basis=basis)
                for j in range(1, len(x)):
                    trajectory_rows.append(dict(dataset=name, kind=kind, id=entity, label=labels[entity], state=state,
                                                year=int(years[valid][j]), cosine_drift=float(1 - x[j] @ x[j-1])))
            fig.suptitle('%s | %s Top%d | %s' % (name, kind, rank, labels[entity]), fontsize=11)
            fig.text(.5, .005, '三栏分别拟合 PCA，方向和尺度不能跨栏比较；定量变化见 trajectory_metrics.csv。固定最佳模型的历史回放，不是逐年重训。', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .06, 1, .93))
            savefig(fig, out / ('comparison_%s/%02d_id%d' % (kind, rank, entity)))
    csvwrite(out / 'trajectory_metrics.csv', trajectory_rows)
    pair_rows, annual_rows = [], []
    for rank, pair in enumerate(pairs, 1):
        a, b = pair['a'], pair['b']
        for state, arr in [('memory', raw_org), ('random', model_org['random']), ('trained', model_org['trained'])]:
            # Same chronological mask for every representation; exclude test snapshots.
            mask = (years < data.spec['test_year']) & (np.linalg.norm(raw_org[:, oi[a]], axis=1) > 1e-10) & (np.linalg.norm(raw_org[:, oi[b]], axis=1) > 1e-10)
            yr = years[mask]; va, vb = arr[mask, oi[a]], arr[mask, oi[b]]
            state_sim, trend_sim, n, states, trends, good = metrics(va, vb, yr)
            row = dict(dataset=name, kind=pair['kind'], a=a, b=b, name_a=names[a], name_b=names[b],
                       shared_train_patents=pair['shared_patents'], state=state, mean_state_cosine=state_sim,
                       mean_delta_cosine=trend_sim, valid_transitions=n)
            pair_rows.append(row)
            for j, year in enumerate(yr):
                annual_rows.append(dict(dataset=name, kind=pair['kind'], a=a, b=b, state=state, year=int(year),
                                        state_cosine=float(states[j]), delta_cosine=float(trends[j-1]) if j and good[j-1] else None))
            if pair['kind'] == 'control':
                continue
            xy, var, mean, basis = pca(unit(np.stack([va, vb], 1)))
            # Align both starting positions for a direct comparison of displacement direction.
            relative = xy - xy[0:1]
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            for k, (entity, marker, line) in enumerate([(a, 'o', '-'), (b, '^', '--')]):
                fireworks(axes[0], yr, xy[:, k], '共享 PCA：绝对位置与轨迹', var, names[entity], marker, line)
                fireworks(axes[1], yr, relative[:, k], '共享 PCA：各自起点平移到原点', var, names[entity], marker, line)
            for ax in axes:
                ax.legend(fontsize=7, loc='upper left')
            trend_text = 'NA' if trend_sim is None else '%.3f' % trend_sim
            fig.suptitle('%s | 合作案例 %d | %s\n%s\n%s' % (name, rank, state, names[a], names[b]), fontsize=10)
            fig.text(.5, .005, '训练期共同专利 %d；高维状态余弦 %.3f；年度位移余弦 %s（%d 个有效区间）' % (pair['shared_patents'], state_sim, trend_text, n), ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, .88))
            base = out / ('cooperation/%02d_%s' % (rank, state))
            savefig(fig, base)
            np.savez_compressed(base.with_suffix('.npz'), years=yr, xy=xy, relative_xy=relative, pca_mean=mean, pca_basis=basis)
    csvwrite(out / 'cooperation_metrics.csv', pair_rows)
    csvwrite(out / 'cooperation_annual.csv', annual_rows)
    audit = dict(dataset=name, config=cfg, checkpoint_epoch=checkpoint['epoch'],
                 checkpoint_sha256=digest(run / 'best.pt'), existing_exports=[str(p.relative_to(ROOT)) for p in run.glob('temporal_embeddings/*.npz')],
                 memory_hashes=source_hashes, years=years.tolist(), train_end=data.spec['train_end_year'], test_year=data.spec['test_year'],
                 cooperation_company_pool=pool_size, positive_pairs_in_pool=pair_count, pairs=pairs,
                 memory_manifest=json.loads((memory / 'manifest.json').read_text()))
    for path in [data.path / 'events.npz', data.path / 'organizations.json', data.path / 'subclasses.json', ROOT / 'data/processed/views' / (name + '.json'), run / 'config.json', ROOT / 'main_experiment/src/patentflow_main/model.py', ROOT / 'data/text' / country / 'taxonomy_anchors.npz'] + list(data.path.glob('counts_*.npz')):
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    dump(out / 'audit.json', audit)
    print('DONE', name, 'figures', len(list(out.rglob('*.png'))), flush=True)
    return pair_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=[c + '_' + s for c in ['CN', 'IN', 'US'] for s in ['All', 'B', 'C', 'F']])
    p.add_argument('--output', type=Path, default=ROOT / 'reports/memory_evolution')
    p.add_argument('--tag', default='v2_default'); p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    font = ROOT / 'reports/presentation_20260919/assets/NotoSansSC-Regular.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    all_pairs = []
    for name in args.datasets:
        all_pairs.extend(run_dataset(name, args))
    csvwrite(args.output / 'cooperation_summary.csv', all_pairs)
    dump(args.output / 'environment.json', dict(python=sys.version, numpy=np.__version__, scipy=__import__('scipy').__version__,
         torch=torch.__version__, matplotlib=matplotlib.__version__, command=sys.argv, script_sha256=digest(Path(__file__))))
    sections = []
    for name in args.datasets:
        chunks = ['<h2 id="%s">%s</h2>' % (name, name)]
        for folder, title in [('memory_org', 'Top5 org：各自独立绘图'), ('memory_ipc', 'Top10 IPC：各自独立绘图'),
                              ('comparison_org', 'Top3 org：原始 / 随机初始化 / 训练后'), ('comparison_ipc', 'Top3 IPC：原始 / 随机初始化 / 训练后'),
                              ('cooperation', '合作公司：每组单独展示（memory / random / trained）')]:
            chunks.append('<h3>%s</h3><div class="grid">' % title)
            for image in sorted((args.output / name / folder).glob('*.png')):
                rel = image.relative_to(args.output).as_posix()
                chunks.append('<a href="%s"><img loading="lazy" src="%s"><span>%s</span></a>' % (rel, rel, html.escape(image.stem)))
            chunks.append('</div>')
        sections.append('\n'.join(chunks))
    page = '''<!doctype html><meta charset="utf-8"><title>Memory 年度演化</title>
<style>body{font-family:sans-serif;max-width:1600px;margin:30px auto;padding:20px;background:#f4f6fa;color:#233044}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}img{width:100%}a{color:#275e92}span{display:block;padding:8px}h2{border-top:2px solid #aac;padding-top:25px}nav{position:sticky;top:0;background:white;padding:15px;z-index:10}nav a{margin-right:16px}</style>
<h1>12 数据集：技术与 IPC 语义年度演化</h1>
<p>Top 排名固定使用训练期累计加权 org–IPC 活动量。每个 org / IPC 单独绘图。原始 memory 并非完全未学习：文本通道已使用 Word2Vec；random 才是未训练主模型。星号为起点，颜色从紫到黄表示年份。PCA 只在同一面板内解释。</p>
<p>合作代表从训练期活动量最高的 200 家名称规则识别公司中，按共同专利数选择 3 组不重复公司；不按相似度选样。对照仅表示该数据视图训练期未观察到共同专利，不能证明现实中无合作。详细规则与限制见 README.md。</p><nav>'''
    page += ''.join('<a href="#%s">%s</a>' % (n, n) for n in args.datasets) + '</nav>' + '\n'.join(sections)
    (args.output / 'index.html').write_text(page)
    report = ['# 年度向量烟花与合作演化分析', '', '[打开全部图表](index.html)', '',
              '## 复现', '', '```bash', 'cd ' + str(ROOT),
              '.venv-evolution/bin/python scripts/visualize_memory_evolution.py', '```', '',
              '环境依赖见 requirements.txt；首次部署使用 Python 3.13 创建 venv 后运行 `pip install -r reports/memory_evolution/requirements.txt`。支持 `--datasets CN_All` 单独重跑。独立核验：`.venv-evolution/bin/python scripts/verify_memory_evolution.py`。', '',
              '## 方法与边界', '',
              '- 默认数据为 `f180_s1095_n1`，org/IPC 各将 fast 与 slow 拼接为 400 维后做 L2 归一化。原始未归一化向量也已保存。',
              '- Top 按截至 train_end_year 的累计加权 org–IPC 活动量排序（不是去重专利数），同分按 ID。IPC 限制为训练候选三级 subclass。org 保留项目实体范围，可能包含高校/研究机构，selection.csv 标注类别。',
              '- 每个实体单独绘图，首个非零年份为烟花中心，射线连向以后各年，折线连接相邻年份；所有年份共同拟合一次确定性 PCA。每图保存 PNG、PDF、二维坐标及 PCA 参数。二维距离只是投影近似，解释率见图。',
              '- Top3 对比为同一实体的原始 memory、同配置同种子初始化主模型、最佳 checkpoint 三栏；每栏独立 PCA，不能比较跨栏角度或尺度。定量变化用原维度单位向量的相邻年份余弦距离。',
              '- memory 文本输入已经过 Word2Vec；random 表示没有进行主实验监督训练。默认只分析 s42，不代表跨种子的统计结论。',
              '- 原始 memory 图包含所有现存年份；模型比较和合作分析遵循项目导出范围，仅用 test_year 之前的年份。保存的模型向量还包含末年供诊断，不用于这些对比。',
              '- 最佳 checkpoint 固定用于全部年份，是事后历史回放，不是每年独立训练，也不能宣称该参数在早年可用。org 导出为 semantic+structural，IPC 导出为 anchor+dynamic；不包含 forward 中合作门控、社区和重复分支的完整预测分数。',
              '- 合作代表限定训练期活动量 Top200、项目名称规则识别为公司的候选池，按当前数据视图内去重共同专利数排序，贪心选取 3 组不重复公司；不按图形或相似度筛选。不是全体公司共同专利数量的全局 Top3。',
              '- 每组对照固定公司 A，为 B 匹配活动量最接近且训练期无共同专利的公司。不保证其他协变量匹配；未观测共同专利不等于现实无合作。',
              '- 状态余弦衡量位置接近；年度位移余弦衡量变化方向。后者只计算双方非零、连续年份且均发生变化的区间。零位移不当作零相似度；缺失记空值。',
              '- 合作两家公司共同拟合 PCA；左图绝对位置，右图将各自起点平移至原点但保留同一投影基。高维指标才是趋势判断依据。相关性不能证明合作导致技术趋同。', '',
              '## 主实验文件审计', '',
              '各数据集 audit.json 记录既有 temporal_embeddings、checkpoint epoch/SHA256、配置、源快照及输入哈希。本次从 best.pt 只读重建选中实体向量，没有重新训练或修改实验目录。', '',
              '## 合作案例结果（各数据集最多 3 组的年度位移余弦均值）', '',
              '| 数据集 | 原始 memory 合作 / 对照 | 训练后合作 / 对照 | 训练后合作正向组数 |',
              '|---|---:|---:|---:|']
    def avg(rows):
        values = [r['mean_delta_cosine'] for r in rows if r['mean_delta_cosine'] is not None]
        return 'NA' if not values else '%.3f' % np.mean(values)
    for name in args.datasets:
        subset = [r for r in all_pairs if r['dataset'] == name]
        cells = []
        for state in ['memory', 'trained']:
            cells.append(' / '.join(avg([r for r in subset if r['state'] == state and r['kind'] == kind]) for kind in ['collaboration', 'control']))
        trained_pairs = [r for r in subset if r['state'] == 'trained' and r['kind'] == 'collaboration' and r['mean_delta_cosine'] is not None]
        report.append('| %s | %s | %s | %d/%d |' % (name, *cells, sum(r['mean_delta_cosine'] > 0 for r in trained_pairs), len(trained_pairs)))
    report += ['', '正值表示平均变化方向一致，接近 0 表示缺少一致方向，负值表示相反；应结合逐年值与有效区间数判断。以上为按共同专利数预选的描述性案例，样本量小，不能据此作总体显著性或因果结论。', '',
               '## 文件', '', '- `index.html`：全量图册，点击图片可放大。', '- `<数据集>/selection.csv`：排名、名称、ID、训练活动量。',
               '- `<数据集>/vectors/{memory,random,trained}.npz`：选中实体逐年原始高维向量与 ID。',
               '- `<数据集>/trajectory_metrics.csv`：Top3 各状态年度变化。',
               '- `cooperation_summary.csv` 与 `<数据集>/cooperation_annual.csv`：合作及匹配对照的高维量化结果。',
               '- `<数据集>/audit.json`、`environment.json`、`requirements.txt`：输入来源、哈希、配置与软件环境。']
    (args.output / 'README.md').write_text('\n'.join(report) + '\n')
    import importlib.metadata
    versions = sorted(d.metadata['Name'] + '==' + d.version for d in importlib.metadata.distributions() if d.metadata['Name'].lower() != 'pip')
    (args.output / 'requirements.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n' + '\n'.join(versions) + '\n')
    print('Gallery:', args.output / 'index.html', flush=True)


if __name__ == '__main__':
    main()
