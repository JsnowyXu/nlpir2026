#!/usr/bin/env python
"""Plot each Top-5 organization's annual semantic memory and every annual patent.

The organization point is the 100-D slow text channel, which is in the same
Word2Vec space as patent_vectors.npy.  This makes org-to-patent cosine values
meaningful; the 400-D mixed memory and trained 100-D model output do not share
the raw patent-vector coordinate system.

Run:
  .venv-evolution/bin/python scripts/visualize_org_patent_semantics.py
"""
import argparse
import csv
import html
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
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import linregress, spearmanr
from patentflow_main.dataset import DatasetView


DATASETS = [c + '_' + s for c in ['CN', 'IN', 'US'] for s in ['All', 'B', 'C', 'F']]


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def shared_pca(x):
    """Deterministic two-component PCA, fitted once to all displayed vectors."""
    mean = x.mean(axis=0)
    _, singular, basis = np.linalg.svd(x - mean, full_matrices=False)
    basis = basis[:2]
    for i in range(2):
        if basis[i, np.argmax(abs(basis[i]))] < 0:
            basis[i] *= -1
    xy = (x - mean) @ basis.T
    variance = float((singular[:2] ** 2).sum() / max((singular ** 2).sum(), 1e-20))
    return xy, mean, basis, variance


def top_orgs(report_dir, dataset):
    path = report_dir / dataset / 'selection.csv'
    with path.open(encoding='utf-8-sig') as handle:
        rows = [r for r in csv.DictReader(handle) if r['kind'] == 'org']
    rows.sort(key=lambda r: int(r['rank']))
    if len(rows) != 5:
        raise ValueError('%s: expected five selected orgs, found %d' % (dataset, len(rows)))
    return rows


def annual_patents(data, org_id):
    """Unique patents by year for org, restricted to the current dataset view."""
    with np.load(data.path / 'events.npz') as events:
        keep = (events['src'] == org_id) & np.isin(events['dst'], data.spec['subclass_ids'])
        pairs = np.unique(np.stack([events['year'][keep], events['patent_row'][keep]], axis=1), axis=0)
    return {int(year): pairs[pairs[:, 0] == year, 1].astype(np.int64) for year in np.unique(pairs[:, 0])}


def select_increasing_similarity_band(similarities, points=20):
    """Select an explicitly increasing upper-tail similarity band.

    For every year, form all contiguous ``points``-patent windows within the
    upper 30% of cosine similarities. A common feasible interval is found and
    target medians increase linearly through it. The selected window is the
    first whose median reaches that year's target and the previous median.
    This guarantees a non-decreasing descriptive trajectory when feasible.
    """
    annual = []
    for values in similarities:
        order = np.argsort(values, kind='stable')
        count = min(points, len(order))
        start = min(int(.70 * len(order)), len(order) - count)
        windows = []
        for left in range(max(0, start), len(order) - count + 1):
            chosen = order[left:left + count]
            windows.append((float(np.median(values[chosen])), chosen))
        annual.append(windows)
    low = max(windows[0][0] for windows in annual)
    high = min(windows[-1][0] for windows in annual)
    if high < low:
        # Expand to the full distribution; failure is still disclosed below.
        annual = []
        for values in similarities:
            order = np.argsort(values, kind='stable'); count = min(points, len(order))
            annual.append([(float(np.median(values[order[left:left + count]])), order[left:left + count])
                           for left in range(0, len(order) - count + 1)])
        low = max(windows[0][0] for windows in annual)
        high = min(windows[-1][0] for windows in annual)
    if high < low:
        return None, None
    targets = np.linspace(low, high, len(annual))
    chosen_all, achieved, previous = [], [], -np.inf
    for target, windows in zip(targets, annual):
        medians = np.asarray([window[0] for window in windows])
        index = min(int(np.searchsorted(medians, max(target, previous), side='left')), len(windows) - 1)
        median, chosen = windows[index]
        if median + 1e-12 < previous:
            return None, targets
        chosen_all.append(chosen); achieved.append(median); previous = median
    return chosen_all, np.asarray(targets)


def plot_one(dataset, rank, org_id, name, years, org_vectors, patent_ids, patent_vectors, output,
             english_no_titles=False):
    valid_org = np.linalg.norm(org_vectors, axis=1) > 1e-10
    displayed_years, displayed_orgs, displayed_patents, displayed_ids = [], [], [], []
    invalid = 0
    for i, year in enumerate(years):
        ids = patent_ids.get(int(year), np.empty(0, dtype=np.int64))
        vectors = np.asarray(patent_vectors[ids]) if len(ids) else np.empty((0, 100), dtype=np.float32)
        good = np.linalg.norm(vectors, axis=1) > 1e-10
        invalid += int((~good).sum())
        if valid_org[i] and good.any():
            displayed_years.append(int(year))
            displayed_orgs.append(org_vectors[i])
            displayed_patents.append(vectors[good])
            displayed_ids.append(ids[good])
    if not displayed_years or not any(len(x) for x in displayed_patents):
        raise ValueError('%s org %d has no valid vectors' % (dataset, org_id))

    org_u = unit(np.stack(displayed_orgs))
    all_patent_u = [unit(x) for x in displayed_patents]
    all_similarities = [vectors @ org_u[i] for i, vectors in enumerate(all_patent_u)]
    chosen_indices, selection_targets = select_increasing_similarity_band(all_similarities)
    monotone_selection = chosen_indices is not None
    if chosen_indices is None:
        # Rare infeasible case: retain the 20 nearest and disclose non-monotonicity.
        chosen_indices = [np.argsort(-values, kind='stable')[:min(20, len(values))]
                          for values in all_similarities]
    selected_ranks = []
    for values, chosen in zip(all_similarities, chosen_indices):
        rank_map = np.empty(len(values), dtype=np.int64)
        rank_map[np.argsort(-values, kind='stable')] = np.arange(1, len(values) + 1)
        selected_ranks.append(rank_map[chosen])
    patent_u = [vectors[index] for vectors, index in zip(all_patent_u, chosen_indices)]
    displayed_ids = [ids[index] for ids, index in zip(displayed_ids, chosen_indices)]
    all_vectors = np.concatenate([org_u] + [x for x in patent_u if len(x)], axis=0)
    xy, pca_mean, pca_basis, explained = shared_pca(all_vectors)
    org_xy = xy[:len(org_u)]
    patent_xy, cursor = [], len(org_u)
    for vectors in patent_u:
        patent_xy.append(xy[cursor:cursor + len(vectors)])
        cursor += len(vectors)
    residual_scale = .10
    display_patent_xy = [org_xy[i] + residual_scale * (points - org_xy[i])
                         for i, points in enumerate(patent_xy)]

    cmap = plt.colormaps['viridis']
    colors = cmap(np.linspace(.06, .94, len(displayed_years)))
    fig, (ax, trend) = plt.subplots(1, 2, figsize=(12.4, 5.4) if english_no_titles else (15, 6.2),
                                  gridspec_kw={'width_ratios': [1.25 if english_no_titles else 1.35, 1]})

    # Every patent is displayed. Rasterization keeps PDF size manageable without sampling.
    for i, (year, points) in enumerate(zip(displayed_years, display_patent_xy)):
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=15, color=colors[i], alpha=.52,
                       linewidths=0, rasterized=True)
    origin = org_xy[0]
    for i in range(1, len(org_xy)):
        ax.annotate('', org_xy[i], origin,
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.1, alpha=.42))
    ax.plot(org_xy[:, 0], org_xy[:, 1], color='#172c45', lw=1.6, alpha=.85, zorder=4)
    ax.scatter(org_xy[:, 0], org_xy[:, 1], c=colors, s=75, marker='D',
               edgecolor='white', linewidth=.7, zorder=5)
    ax.scatter(*origin, s=190, marker='*', color='#dd5b35', edgecolor='white', linewidth=.7, zorder=6)
    for label_index, (year, point) in enumerate(zip(displayed_years, org_xy)):
        distance = 8 + 10 * (label_index % 4)
        direction = 1 if label_index % 2 == 0 else -1
        label_offset = (direction * distance, 6 + 5 * (label_index % 2))
        if english_no_titles:
            label_offset = {2014: (-38, -17), 2015: (-38, 3), 2016: (12, -17),
                            2017: (-38, 23), 2019: (12, 18)}.get(year, label_offset)
        ax.annotate(str(year), point, xytext=label_offset, textcoords='offset points',
                    fontsize=11 if english_no_titles else 8,
                    bbox=dict(boxstyle='round,pad=.14', fc='white', ec='none', alpha=.72))
    ax.set(title='' if english_no_titles else '公司语义烟花与逐年最近专利',
           xlabel='Shared PC1' if english_no_titles else '共享 PC1',
           ylabel='Shared PC2' if english_no_titles else '共享 PC2')
    ax.grid(alpha=.15)
    if english_no_titles:
        handles = [
            Line2D([], [], marker='o', linestyle='None', markersize=5, color='#74a9b8', label='Selected patents'),
            Line2D([], [], marker='D', linestyle='None', markersize=7, color='#263d60', label='Organization state'),
            Line2D([], [], marker='*', linestyle='None', markersize=11, color='#dd5b35', label='First year'),
        ]
        ax.legend(handles=handles, fontsize=11, loc='lower center', bbox_to_anchor=(.5, 1.015),
                  ncol=3, frameon=False, borderaxespad=0, columnspacing=1.1, handletextpad=.35)

    rows, medians, priors = [], [], []
    for i, (year, patents, ids, ranks) in enumerate(zip(displayed_years, patent_u, displayed_ids, selected_ranks)):
        similarities = patents @ org_u[i] if len(patents) else np.empty(0)
        prior = patents @ org_u[i - 1] if i and len(patents) else np.full(len(patents), np.nan)
        if len(similarities):
            q10, q25, median, q75, q90 = np.quantile(similarities, [.1, .25, .5, .75, .9])
            medians.append(median)
            priors.append(float(np.nanmedian(prior)) if i else np.nan)
            trend.vlines(year, q10, q90, color=colors[i], lw=2, alpha=.65)
            trend.vlines(year, q25, q75, color=colors[i], lw=7, alpha=.75)
            trend.scatter(year, median, color=colors[i], edgecolor='white', linewidth=.5, s=50, zorder=3)
            for selected_rank, patent_id, similarity, prior_similarity in zip(ranks, ids, similarities, prior):
                rows.append(dict(dataset=dataset, rank=rank, org_id=org_id, organization=name,
                                 year=year, patent_row=int(patent_id),
                                 selected_similarity_rank=selected_rank,
                                 selected_k=len(ids), candidate_patents=len(displayed_patents[i]),
                                 cosine_same_year=float(similarity),
                                 cosine_prior_year=None if np.isnan(prior_similarity) else float(prior_similarity)))
        else:
            medians.append(np.nan); priors.append(np.nan)
    year_array = np.asarray(displayed_years)
    median_array = np.asarray(medians)
    prior_array = np.asarray(priors)
    good = np.isfinite(median_array)
    regression = linregress(year_array[good], median_array[good]) if good.sum() >= 2 else None
    rho = spearmanr(year_array[good], median_array[good]) if good.sum() >= 2 else None
    trend.plot(year_array[good], median_array[good], color='#172c45', lw=1.3,
               label='Same-year organization state: median' if english_no_titles else '同年公司向量：中位数')
    prior_good = np.isfinite(prior_array)
    prior_regression = linregress(year_array[prior_good], prior_array[prior_good]) if prior_good.sum() >= 2 else None
    prior_rho = spearmanr(year_array[prior_good], prior_array[prior_good]) if prior_good.sum() >= 2 else None
    if prior_good.any():
        trend.plot(year_array[prior_good], prior_array[prior_good], '--', color='#c05b42', lw=1.3,
                   label='Previous-year organization state: median' if english_no_titles else '上一年公司向量：中位数')
    trend.set(title='' if english_no_titles else '专利—公司余弦相似度随年份变化',
              xlabel='Year' if english_no_titles else '年份',
              ylabel='Cosine similarity' if english_no_titles else '余弦相似度')
    trend.grid(alpha=.15)
    if english_no_titles:
        trend.legend(fontsize=11, loc='lower center', bbox_to_anchor=(.5, 1.015),
                     ncol=1, frameon=False, borderaxespad=0)
    else:
        trend.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    trend.set_xticks(displayed_years); trend.tick_params(axis='x', rotation=45)

    if not english_no_titles:
        fig.suptitle('%s | Top%d org | %s' % (dataset, rank, name), fontsize=12)
    selection_text = ('每年选择20项上尾相似度带专利，目标中位数逐年递增' if monotone_selection else
                      '各年相似度范围不存在共同递增路径，回退为每年Top20最近专利')
    stats = ('PCA解释率 %.1f%%；%s；左图专利相对公司残差缩放为0.10。'
             % (100 * explained, selection_text))
    if regression:
        stats += ' 同年斜率 %+0.4f/年，Spearman ρ=%+.3f (p=%.3g)。' % (regression.slope, rho.statistic, rho.pvalue)
    if not english_no_titles:
        fig.text(.5, .035, stats, ha='center', fontsize=8)
        fig.text(.5, .012, '圆点=预选最近专利；菱形=公司 slow-text memory；选择条件本身会造成上升趋势，仅用于条件化案例展示。',
                 ha='center', fontsize=8, color='#8b3e2f')
    fig.tight_layout(rect=(0, 0 if english_no_titles else .075,
                           1 if english_no_titles else .92,
                           .85 if english_no_titles else .94),
                     w_pad=.6 if english_no_titles else None)
    base = output / ('%02d_id%d%s' % (rank, org_id, '_en' if english_no_titles else ''))
    fig.savefig(base.with_suffix('.png'), dpi=170, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    lengths = np.asarray([len(x) for x in displayed_ids], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    np.savez_compressed(base.with_suffix('.npz'), years=np.asarray(displayed_years), org=org_u,
                        org_xy=org_xy, patent_ids=np.concatenate(displayed_ids),
                        patent_xy=np.concatenate(patent_xy), display_patent_xy=np.concatenate(display_patent_xy),
                        patent_offsets=offsets, residual_display_scale=residual_scale, pca_mean=pca_mean,
                        pca_basis=pca_basis, pca_explained_variance=explained)
    summary = dict(dataset=dataset, rank=rank, org_id=org_id, organization=name,
                   years=len(displayed_years), patents=sum(len(x) for x in patent_u),
                   candidate_patents=sum(len(x) for x in displayed_patents),
                   monotone_selection=monotone_selection,
                   selection_target_start=None if selection_targets is None else float(selection_targets[0]),
                   selection_target_end=None if selection_targets is None else float(selection_targets[-1]),
                   invalid_patent_vectors=invalid, pca_explained_variance=explained,
                   same_year_median_slope=None if regression is None else float(regression.slope),
                   same_year_spearman_rho=None if rho is None else float(rho.statistic),
                   same_year_spearman_p=None if rho is None else float(rho.pvalue),
                   prior_year_median_slope=None if prior_regression is None else float(prior_regression.slope),
                   prior_year_spearman_rho=None if prior_rho is None else float(prior_rho.statistic),
                   prior_year_spearman_p=None if prior_rho is None else float(prior_rho.pvalue),
                   first_median=None if not good.any() else float(median_array[good][0]),
                   last_median=None if not good.any() else float(median_array[good][-1]),
                   change_last_minus_first=None if not good.any() else float(median_array[good][-1] - median_array[good][0]))
    return rows, summary


def update_gallery(report_dir, datasets):
    path = report_dir / 'index.html'
    text = path.read_text()
    start, end = '<!-- ORG_PATENT_SEMANTICS_START -->', '<!-- ORG_PATENT_SEMANTICS_END -->'
    if start in text:
        text = text[:text.index(start)] + text[text.index(end) + len(end):]
    blocks = [start, '<h1 id="org-patent-semantics">Top5 org：公司语义与逐年预选近邻专利</h1>',
              '<p>左图显示逐年条件筛选的20项近邻专利局部簇；右图在原100维计算真实余弦相似度。递增约束不可行时回退为Top20，详见说明文档。</p>']
    for dataset in datasets:
        blocks += ['<h2>%s</h2><div class="grid">' % dataset]
        for image in sorted((report_dir / dataset / 'org_patent_semantics').glob('*.png')):
            rel = image.relative_to(report_dir).as_posix()
            blocks.append('<a href="%s"><img loading="lazy" src="%s"><span>%s</span></a>' %
                          (rel, rel, html.escape(image.stem)))
        blocks.append('</div>')
    blocks.append(end)
    path.write_text(text + '\n' + '\n'.join(blocks) + '\n')


def write_report(report_dir, summaries):
    positive = sum(row['change_last_minus_first'] > 0 for row in summaries)
    significant_positive = sum(row['same_year_spearman_p'] < .05 and row['same_year_spearman_rho'] > 0 for row in summaries)
    significant_negative = sum(row['same_year_spearman_p'] < .05 and row['same_year_spearman_rho'] < 0 for row in summaries)
    prior_positive = sum(row['prior_year_median_slope'] > 0 for row in summaries)
    feasible = sum(row['monotone_selection'] for row in summaries)
    lines = ['# Top5 org 与逐年关联专利语义', '',
             '[打开图册](index.html#org-patent-semantics)', '', '## 复现', '', '```bash',
             'cd ' + str(ROOT),
             '.venv-evolution/bin/python scripts/visualize_org_patent_semantics.py',
             '.venv-evolution/bin/python scripts/verify_org_patent_semantics.py', '```', '',
             '## 结果', '',
             '共生成 12×5=60 张独立图片。按用户指定，本版优先构造条件化案例：每年在余弦相似度上尾区间构造连续20项专利窗口，在各年份共同可行的相似度区间内设置逐年递增的目标中位数，并选择首个达到目标且不低于上年中位数的窗口。该约束对%d/60个org可行；其余%d个org即使扩大至当年全部专利，也不存在覆盖所有年份的非递减相似度路径，因而回退为每年Top20最近专利，没有伪造上升。最终首末年份相似度%d/60上升；年度Spearman显著上升%d/60、显著下降%d/60（p<0.05）；上一年公司向量口径下%d/60的线性斜率为正。条件筛选所得上升只描述选中子集，不能作为模型自然产生技术聚集的证据。' %
             (feasible, 60 - feasible, positive, significant_positive, significant_negative, prior_positive), '',
             '## 口径', '',
             '- Top5 与原报告 `selection.csv` 完全一致，按训练期累计活动量固定选择；org 可能是公司或研究机构。',
             '- 候选专利是当前数据集视图中该 org 当年出现的去重 patent_row；同一专利有多个 IPC 时只计一次。图中只画按上述规则预选的最近专利。',
             '- 公司采用 memory 的 slow-text 前100维，专利采用 `patent_vectors.npy` 的100维 Word2Vec 向量，两者处于同一原始语义空间。400维混合 memory 和训练后模型输出不与原始专利向量直接混画。',
             '- 向量先做 L2 归一化，再对一家公司全部年份的公司与预选专利共同拟合一次 PCA。每个被选中的专利点都实际绘出，没有再抽样。为避免局部专利云淹没公司轨迹，左图将“专利坐标−同年公司坐标”的残差统一乘0.10后加回公司坐标；原始未缩放二维坐标同时保存在 NPZ。',
             '- 右图在原100维空间计算余弦相似度：细线为10%–90%分位数，粗线为25%–75%分位数，圆点为中位数。',
             '- 同年公司向量是当年年末的慢速 memory，已经吸收当年专利，因此同年相似度存在定义上的机械相关。红色虚线把当年专利与上一年公司向量比较，作为更严格的时间对照。',
             '- 预选算法优先限制在每年相似度最高的30%候选中；每年固定选择20项。共同递增区间不可行时扩大到全分布，再不可行则退化为每年 Top20，并在 summary 的 `monotone_selection` 标记为 false。',
             '- 这一图适合展示满足假设的近邻子集，不适合检验“所有专利是否越来越聚集”。严谨检验应同时保留上一版全量专利统计或另做预注册阈值分析。', '',
             '## 输出', '',
             '- `<dataset>/org_patent_semantics/*.png|pdf`：每个 org 一张图。',
             '- 同名 `.npz`：原始年度公司单位向量、全部专利 ID、二维坐标、年度 offsets、PCA 参数。',
             '- `org_patent_semantics_patents.csv`：每项专利的同年及上一年余弦相似度。',
             '- `org_patent_semantics_summary.csv`：60 个 org 的趋势统计。',
             '- `org_patent_semantics_verification.json`：独立核验结果。']
    lines += ['', '## 图片逐项解释', '',
              '“条件递增”表示存在覆盖全部有效年份的20专利相似度窗口；“回退Top20”表示原始相似度范围不允许构造该路径。斜率仍由原100维余弦计算。', '']
    for dataset in sorted({row['dataset'] for row in summaries}):
        lines += ['### ' + dataset, '', '| 排名 | org | 年份 | 预选/候选专利 | 筛选 | 同年斜率/年 | 图 |',
                  '|---:|---|---:|---:|---|---:|---|']
        for row in [x for x in summaries if x['dataset'] == dataset]:
            mode = '条件递增' if row['monotone_selection'] else '回退Top20（无可行递增路径）'
            image = '%s/org_patent_semantics/%02d_id%d.png' % (dataset, row['rank'], row['org_id'])
            lines.append('| %d | %s | %d | %d/%d | %s | %+.4f | [查看](%s) |' %
                         (row['rank'], row['organization'], row['years'], row['patents'],
                          row['candidate_patents'], mode, row['same_year_median_slope'], image))
        lines.append('')
    (report_dir / 'ORG_PATENT_SEMANTICS.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--report-dir', type=Path, default=ROOT / 'reports/memory_evolution')
    p.add_argument('--case-org-id', type=int, help='Render only this organization without rewriting aggregate reports')
    p.add_argument('--english-no-titles', action='store_true', help='Use English axis labels and legend; omit headings and footer')
    args = p.parse_args(); args.report_dir = args.report_dir.resolve()
    font = ROOT / 'reports/presentation_20260919/assets/NotoSansSC-Regular.ttf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    all_rows, all_summaries = [], []
    for dataset in args.datasets:
        print('START', dataset, flush=True)
        data = DatasetView(dataset)
        selection = top_orgs(args.report_dir, dataset)
        if args.case_org_id is not None:
            selection = [row for row in selection if int(row['id']) == args.case_org_id]
            if not selection:
                raise ValueError('%s: organization %d is not selected' % (dataset, args.case_org_id))
        slug = data.spec['country'] if data.spec['section'] == 'All' else data.spec['country'] + '_' + data.spec['section']
        memory = ROOT / 'data/memory' / slug / 'f180_s1095_n1'
        years = np.asarray(sorted(int(path.stem) for path in memory.glob('[0-9]*.npz')))
        ids = [int(row['id']) for row in selection]
        org_vectors = []
        for year in years:
            with np.load(memory / ('%d.npz' % year)) as snapshot:
                org_vectors.append(snapshot['org_slow'][ids, :100])
        org_vectors = np.stack(org_vectors)
        patent_vectors = np.load(ROOT / 'data/text' / data.spec['country'] / 'patent_vectors.npy', mmap_mode='r')
        output = args.report_dir / dataset / 'org_patent_semantics'; output.mkdir(exist_ok=True)
        for index, row in enumerate(selection):
            with plt.rc_context({'font.size': 11} if args.english_no_titles else {}):
                records, summary = plot_one(dataset, int(row['rank']), int(row['id']), row['label'], years,
                                            org_vectors[:, index], annual_patents(data, int(row['id'])),
                                            patent_vectors, output, args.english_no_titles)
            all_rows.extend(records); all_summaries.append(summary)
        if args.case_org_id is not None:
            print('DONE', dataset, flush=True)
            continue
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_summaries[-5]))
            writer.writeheader(); writer.writerows(all_summaries[-5:])
        print('DONE', dataset, flush=True)
    if args.case_org_id is not None:
        return
    with (args.report_dir / 'org_patent_semantics_summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summaries[0])); writer.writeheader(); writer.writerows(all_summaries)
    with (args.report_dir / 'org_patent_semantics_patents.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    update_gallery(args.report_dir, args.datasets)
    write_report(args.report_dir, all_summaries)
    print('Gallery:', args.report_dir / 'index.html', flush=True)


if __name__ == '__main__':
    main()
