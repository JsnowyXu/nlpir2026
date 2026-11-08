#!/usr/bin/env python
"""CN_All IPC levels 1/2/3: original anchors versus aligned 2019 embeddings.

Independent level checkpoints have arbitrary latent rotations.  Each 2019 level
is therefore orthogonally aligned to its matching original taxonomy anchors,
then original and 2019 vectors are embedded together by one deterministic
t-SNE. Both panels consequently use exactly the same 2-D coordinate system.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['PATENTFLOW_WORKSPACE'] = str(ROOT)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path[:0] = [str(ROOT / 'main_experiment/src'), str(ROOT / 'src')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch
from patentflow_main.dataset import DatasetView
from patentflow_main.model import PatentFlow
from patentflow_main.neural_data import MemoryFeatures, anchors_for


COLORS = dict(zip('ABCDEFGH', ['#e76f51', '#f4a261', '#2a9d8f', '#8e6c8a',
                                '#457b9d', '#e9c46a', '#7b61a8', '#d62828']))
MARKERS = {1: '*', 2: 'o', 3: 'D'}
SIZES = {1: 190, 2: 35, 3: 15}
TAGS = {1: 'level1_default', 2: 'level2_default', 3: 'v2_default'}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def level_vectors(level, year=2019):
    data = DatasetView('CN_All', level=level)
    run = ROOT / 'results/main/CN_All' / TAGS[level] / 's42'
    cfg = json.loads((run / 'config.json').read_text())
    model = PatentFlow(anchors_for(data), dim=cfg['dim'], communities=cfg['communities'],
                       dropout=cfg['dropout'], disabled=cfg['disable'],
                       temperature=cfg['temperature']).eval()
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    features = MemoryFeatures(data, year, 'cpu', cfg['fast'], cfg['slow'], cfg['novelty'])
    with torch.no_grad():
        # org input is irrelevant to IPC export; pass one real row.
        _, learned = model.export_embeddings(torch.tensor(features.org_raw[:1]), features.ipc_raw)
    codes = np.asarray([data.codes[i] for i in data.candidate_ids])
    archive = np.load(ROOT / 'data/text/cn/taxonomy_anchors.npz')
    lookup = dict(zip(archive['codes'].tolist(), archive['anchors']))
    original = unit(np.stack([lookup[code] for code in codes]).astype(np.float64))
    learned = unit(learned.numpy().astype(np.float64))
    rotation, _ = orthogonal_procrustes(learned, original)
    aligned = unit(learned @ rotation)
    residual = float(np.sqrt(np.mean((aligned - original) ** 2)))
    return codes, original, aligned, residual, int(checkpoint['epoch']), str(run / 'best.pt')


def purity(xy, sections, neighbors=10):
    indices = NearestNeighbors(n_neighbors=min(neighbors + 1, len(xy))).fit(xy).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean(sections[indices] == sections[:, None]))


def plot_panel(ax, xy, codes, levels, title):
    # Fixed draw order: subclasses, classes, sections, so parent stars remain visible.
    for section in 'ABCDEFGH':
        for level in [3, 2, 1]:
            mask = (np.char.startswith(codes, section)) & (levels == level)
            if not mask.any():
                continue
            ax.scatter(xy[mask, 0], xy[mask, 1], marker=MARKERS[level], s=SIZES[level],
                       c=COLORS[section], alpha={1: 1, 2: .76, 3: .48}[level],
                       edgecolors='white' if level < 3 else 'none', linewidths=.45,
                       label=section if level == 1 else None, rasterized=level == 3)
    for i in np.flatnonzero(levels == 1):
        ax.annotate(codes[i], xy[i], xytext=(5, 5), textcoords='offset points', fontsize=8, weight='bold')
    ax.set_title(title); ax.set_xlabel('共享 t-SNE 1'); ax.set_ylabel('共享 t-SNE 2')
    ax.grid(alpha=.10); ax.set_aspect('equal', adjustable='box')


def main():
    output = ROOT / 'reports/memory_evolution/CN_All/ipc_hierarchy_semantics'
    output.mkdir(parents=True, exist_ok=True)
    font = ROOT / 'reports/presentation_20260919/assets/NotoSansSC-Regular.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    codes, levels, raw, final, residuals, sources = [], [], [], [], {}, []
    for level in [1, 2, 3]:
        c, r, f, residual, epoch, checkpoint = level_vectors(level)
        codes.extend(c.tolist()); levels.extend([level] * len(c)); raw.append(r); final.append(f)
        residuals[str(level)] = residual
        sources.append(dict(level=level, tag=TAGS[level], best_epoch=epoch,
                            checkpoint=str(Path(checkpoint).relative_to(ROOT))))
    codes = np.asarray(codes); levels = np.asarray(levels)
    raw, final = np.concatenate(raw), np.concatenate(final)
    both = np.concatenate([raw, final])
    embedding = TSNE(n_components=2, perplexity=35, init='pca', learning_rate='auto',
                     max_iter=2000, random_state=42, metric='cosine').fit_transform(both)
    raw_xy, final_xy = embedding[:len(raw)], embedding[len(raw):]
    sections = np.asarray([ord(code[0]) - ord('A') for code in codes])
    metrics = dict(dataset='CN_All', final_year=2019, seed=42, reducer='joint t-SNE',
                   perplexity=35, iterations=2000, node_count=len(codes),
                   counts_by_level={str(i): int((levels == i).sum()) for i in [1, 2, 3]},
                   procrustes_rmse=residuals,
                   raw_section_silhouette=float(silhouette_score(raw_xy, sections)),
                   final_section_silhouette=float(silhouette_score(final_xy, sections)),
                   raw_knn_section_purity=purity(raw_xy, sections),
                   final_knn_section_purity=purity(final_xy, sections), sources=sources,
                   limitation='Joint t-SNE preserves local neighborhoods, not global distance; checkpoints are retrospective best-validation models.')
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7))
    plot_panel(axes[0], raw_xy, codes, levels, '原始 IPC 一/二/三级语义锚点')
    plot_panel(axes[1], final_xy, codes, levels, '2019 IPC 一/二/三级语义（对齐后）')
    xlim = (embedding[:, 0].min() - 2, embedding[:, 0].max() + 2)
    ylim = (embedding[:, 1].min() - 2, embedding[:, 1].max() + 2)
    for ax in axes: ax.set_xlim(xlim); ax.set_ylim(ylim)
    section_handles = [plt.Line2D([0], [0], marker='s', linestyle='', color=COLORS[s], label=s, markersize=8) for s in 'ABCDEFGH']
    level_handles = [plt.Line2D([0], [0], marker=MARKERS[i], linestyle='', color='#555', label='Level %d' % i,
                                markersize={1: 12, 2: 7, 3: 6}[i]) for i in [1, 2, 3]]
    fig.legend(handles=section_handles + level_handles, loc='lower center', ncol=11,
               bbox_to_anchor=(.5, .015), frameon=True, fontsize=9)
    fig.suptitle('CN_All IPC 层级语义：八个 section 固定颜色、层级固定形状、两图共享坐标', fontsize=13)
    fig.text(.5, .075, '颜色：A–H；星形=一级 section，圆形=二级 class，菱形=三级 subclass。2019 三个独立潜空间先以原始锚点正交对齐。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    base = output / 'ipc_hierarchy_original_vs_2019'
    fig.savefig(base.with_suffix('.png'), dpi=190, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), dpi=190, bbox_inches='tight')
    plt.close(fig)
    np.savez_compressed(base.with_suffix('.npz'), codes=codes, levels=levels, sections=sections,
                        raw=raw, final_aligned=final, raw_xy=raw_xy, final_xy=final_xy)
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (output / 'README.md').write_text('''# IPC 层级语义图说明

左图是 taxonomy 描述经 Word2Vec 得到的原始 100 维锚点；右图是三个层级各自 seed 42 最佳 checkpoint 对 2019 memory 的输出。一级、二级、三级分别来自 `level1_default`、`level2_default`、`v2_default`。

三个 checkpoint 的潜空间存在任意旋转，不能直接叠加。代码先以对应原始锚点做正交 Procrustes 对齐，再将原始与2019全部节点共同进行一次固定参数 t-SNE，因此两幅图的坐标轴、范围、颜色和节点顺序完全一致。星形、圆形、菱形分别表示 section、class、subclass；颜色由首字母 A–H 决定。

t-SNE 只适合观察局部邻域，簇间远近和二维位移大小不能直接解释为高维语义距离。八个主簇是待检验预期，不是绘图程序强制产生的结果；定量 section silhouette 与10近邻纯度见 `metrics.json`。

复现：

```bash
cd ''' + str(ROOT) + '''
.venv-evolution/bin/python scripts/visualize_ipc_hierarchy_semantics.py
```
''')
    print(base.with_suffix('.png'))


if __name__ == '__main__':
    main()
