#!/usr/bin/env python
"""Build a concise, audit-ready summary from the latest frozen empirical snapshot."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "draft/patentflow"
OUT = ROOT / "reports/experiment_summary_20260921"
OUT.mkdir(parents=True, exist_ok=True)

snapshot = json.loads((SRC / "data/empirical_snapshot.json").read_text())
progress = json.loads((ROOT / "state/progress.json").read_text())
official = pd.read_csv(SRC / "data/official_aggregates.csv")
seeds = pd.read_csv(SRC / "data/official_seed_results.csv")
components = pd.read_csv(SRC / "data/component_aggregates.csv")
component_seeds = pd.read_csv(SRC / "data/component_seed_results.csv")
datasets = pd.read_csv(SRC / "data/12视图图统计.csv")
splits = pd.read_csv(SRC / "data/划分和标签重复.csv")
sensitivity = pd.read_csv(SRC / "data/sensitivity_plot_values.csv")

methods = [
    "PatentFlow", "TOP", "PersonalTOP", "SVM", "XGBoost", "RF", "EDGPAT",
    "MENCP", "RepeatMixer", "TGAT", "DyGFormer", "TAMI", "IDGP", "TGN",
    "TGN_edgetype",
]
datasets_order = [f"{c}_{s}" for c in ["CN", "US", "IN"] for s in ["All", "B", "F", "C"]]

run_counts = Counter()
for record in snapshot["records"]:
    value = record["value"]
    for split in value.get("splits", {}):
        run_counts[(value["method"], split)] += 1
progress_rows = pd.DataFrame([
    {
        "method": method,
        "validation_complete": run_counts[(method, "validation")],
        "test_complete": run_counts[(method, "test")],
        "target_per_stage": 36,
        "validation_remaining": 36 - run_counts[(method, "validation")],
        "test_remaining": 36 - run_counts[(method, "test")],
    }
    for method in methods
])

queue_rows = []
for name, value in sorted(progress["campaigns"].items()):
    states = value.get("states", {})
    queue_rows.append({
        "campaign": name,
        "complete": states.get("complete", 0),
        "failed": states.get("failed", 0),
        "running_processes": len(value.get("live_processes", [])),
        "planned": value.get("jobs", 0),
        "remaining": value.get("jobs", 0) - states.get("complete", 0) - states.get("failed", 0),
    })
queue_rows = pd.DataFrame(queue_rows)

main = official[
    (official.split == "validation")
    & (official.population == "active")
    & (official.K == 20)
    & (official.metric == "ndcg")
].copy()
main["dataset"] = pd.Categorical(main.dataset, datasets_order, ordered=True)
main["method"] = pd.Categorical(main.method, methods, ordered=True)
main = main.sort_values(["dataset", "method"])

section_datasets = [f"{c}_{s}" for c in ["CN", "US", "IN"] for s in ["B", "F", "C"]]
top_section = main[main.dataset.astype(str).isin(section_datasets)].copy()
top_section["dataset"] = top_section["dataset"].astype(str)

# Complete method/view grid: absent or unfinished results remain visible instead of
# silently disappearing from a grouped table.
section_grid = pd.MultiIndex.from_product(
    [methods, section_datasets], names=["method", "dataset"]
).to_frame(index=False).merge(
    top_section[["method", "dataset", "mean", "var", "count"]],
    on=["method", "dataset"], how="left",
)
section_grid["n"] = section_grid["count"].fillna(0).astype(int)
section_grid["status"] = np.select(
    [section_grid.n.eq(3), section_grid.n.eq(0)],
    ["complete", "missing"], default="partial",
)

def result_cell(row):
    if row.n == 0:
        return "- [n=0]"
    variance = "-" if pd.isna(row["var"]) else f"{row['var']:.6f}"
    return f"{row['mean']:.6f} ± {variance} [n={row.n}]"

section_grid["mean ± sample variance [n]"] = section_grid.apply(result_cell, axis=1)
section_matrix = section_grid.pivot(
    index="method", columns="dataset", values="mean ± sample variance [n]"
).reindex(index=methods, columns=section_datasets).reset_index()

top_section_all_metrics = official[
    (official.dataset.isin(section_datasets))
    & (official.split == "validation")
    & (official.population == "active")
].copy()
top_section_test = official[
    (official.dataset.isin(section_datasets))
    & (official.split == "test")
    & (official.population == "active")
    & (official.K == 20)
    & (official.metric == "ndcg")
].copy()

all_metrics = official[
    (official.method == "PatentFlow")
    & (official.split == "validation")
    & (official.population == "active")
].copy()
all_metrics["dataset"] = pd.Categorical(all_metrics.dataset, datasets_order, ordered=True)
all_metrics = all_metrics.sort_values(["dataset", "K", "metric"])

ablation = components[
    components.tag.str.startswith("ablation_")
    & components.dataset.isin(["CN_All", "US_All", "IN_All"])
].copy()
memory = components[
    components.tag.str.startswith("memory_")
    & components.dataset.isin(["CN_All", "US_All", "IN_All"])
].copy()

level_rows = []
for path in sorted((ROOT / "results/main").glob("*_All/level*_default/s*/validation.json")):
    value = json.loads(path.read_text())
    level_rows.append({
        "dataset": value["dataset"],
        "level": value["config"]["level"],
        "tag": value["config"]["tag"],
        "seed": value["seed"],
        "ndcg20": value["splits"]["validation"]["active"]["ndcg_20"],
        "mrr20": value["splits"]["validation"]["active"]["mrr_20"],
        "path": str(path.relative_to(ROOT)),
    })
level_seed = pd.DataFrame(level_rows)
level = level_seed.groupby(["dataset", "level", "tag"]).agg(
    n=("ndcg20", "count"), mean_ndcg20=("ndcg20", "mean"),
    variance_ndcg20=("ndcg20", "var"), mean_mrr20=("mrr20", "mean")
).reset_index()

# Independent consistency assertions for the summary workbook.
assert len(main) > 0
assert not seeds.duplicated(["dataset", "method", "seed", "split", "population", "K", "metric"]).any()
group_keys = ["dataset", "method", "split", "population", "K", "metric"]
recomputed = seeds.groupby(group_keys).value.agg(
    check_count="count", check_mean="mean", check_var="var"
).reset_index()
verified = official.merge(recomputed, on=group_keys, how="left", validate="one_to_one")
assert (verified["count"] == verified["check_count"]).all()
assert np.allclose(verified["mean"], verified["check_mean"], equal_nan=True)
assert np.allclose(verified["var"], verified["check_var"], equal_nan=True)

readme = f"""# 当前实验汇总

结果快照（UTC）：{snapshot['frozen_utc']}  
进度核查（UTC）：{progress['updated']}

本目录只汇总当前协议指纹 `{snapshot['protocol_sha256']}` 的正式结果。验证与测试分开；缺失测试不以验证值替代。均值和样本方差先在每个种子内对查询做宏平均，再跨已有种子聚合。TOP 与 PersonalTOP 为确定性方法，三个种子记录相同。

## 当前结论

- PatentFlow、EDGPAT、MENCP、TGAT、TAMI、IDGP、TGN、TGN_edgetype 的 36 组三级验证均完成。
- RepeatMixer 完成 {run_counts[('RepeatMixer','validation')]}/36；DyGFormer 完成 {run_counts[('DyGFormer','validation')]}/36，仍在运行。
- 五类传统方法均完成 36 组验证和 36 组测试。所有神经方法正式测试仍为 0，因此不能给出最终泛化结论。
- 消融 396/396、记忆变体 162/162、敏感性 180/180、分类层级 18/18 均完成且无失败。
- 三国 All 的 PatentFlow 验证 NDCG@20 为 CN 0.566516、US 0.742010、IN 0.567601；均高于当前完整三种子对照中的 XGBoost，但 US/IN 的 MRR 并未全面领先 PersonalTOP。
- 消融不支持“所有模块均有效”：移除 anchor/alignment 在三国 All 均略有提升，IN_All 移除 structure 提升更明显；文本和频次残差的贡献较稳定。
- 单慢尺度 1095 天在三个 All 视图均取得当前记忆变体中的最高均值，说明默认双尺度设置尚不是已证实的最优时间配置。

## 文件

- `实验进度.xlsx`：进度、主结果、全部指标、消融、记忆、敏感性、层级和数据集统计。
- `当前实验汇总.md`：主要数值与解释。
- `TopSection实验汇总.md`：九个 Top section 的逐方法结果和完成度。
- `official_aggregates.csv`、`official_seed_results.csv`：正式汇总和逐种子结果。
- `component_aggregates.csv`、`component_seed_results.csv`：组件实验汇总和逐种子结果。
- `audit.json`：快照、行数及文件哈希。
"""
(OUT / "README.md").write_text(readme)

lines = [
    "# 当前实验结果摘要", "",
    f"冻结时间：{snapshot['frozen_utc']}；进度时间：{progress['updated']}。", "",
    "## 三国 All：验证 active NDCG@20（均值 ± 样本方差；n）", "",
]
for dataset in ["CN_All", "US_All", "IN_All"]:
    lines += [f"### {dataset}", "", "| 方法 | 均值 | 样本方差 | n |", "|---|---:|---:|---:|"]
    q = main[main.dataset == dataset].sort_values("mean", ascending=False)
    for row in q.itertuples(index=False):
        variance = "-" if pd.isna(row.var) else f"{row.var:.6g}"
        lines.append(f"| {row.method} | {row.mean:.6f} | {variance} | {int(row.count)} |")
    lines.append("")
lines += [
    "## 消融结论", "",
    "CPU 同配置 full 的均值为 CN 0.567352、US 0.741785、IN 0.567309。CN/US/IN 去掉 anchor 后分别为 0.568457/0.742535/0.568089；去掉 alignment 为 0.567907/0.741866/0.568502。IN 去掉 structure 达 0.581752。去掉 text 分别降至 0.544433/0.732640/0.541708，去掉 frequency residual 分别为 0.557580/0.726453/0.562336。", "",
    "## 记忆与层级", "",
    "三个 All 视图当前最佳记忆变体均为 single_1095：CN 0.568743、US 0.742767、IN 0.569302。一级/二级任务已各完成 9 组；它们的候选空间更小，指标不能与三级结果直接作为同难度比较。", "",
    "## 尚未完成", "",
    f"RepeatMixer 还缺 {36-run_counts[('RepeatMixer','validation')]} 个验证运行，DyGFormer 还缺 {36-run_counts[('DyGFormer','validation')]} 个。神经方法最终测试、最终资源公平比较及锁定设置后的统一结论仍待完成。", "",
]
(OUT / "当前实验汇总.md").write_text("\n".join(lines))

section_lines = [
    "# Top section 实验结果", "",
    f"冻结时间：{snapshot['frozen_utc']}。以下均为验证集 active 查询的 NDCG@20，格式为均值 ± 样本方差 [已有种子数]。`- [n=0]` 表示当前没有结果；n<3 表示尚未完成。", "",
]
for dataset in section_datasets:
    section_lines += [f"## {dataset}", "", "| 方法 | 结果 | 状态 |", "|---|---:|---:|"]
    q = section_grid[section_grid.dataset == dataset].set_index("method").reindex(methods).reset_index()
    for _, row in q.iterrows():
        section_lines.append(
            f"| {row['method']} | {row['mean ± sample variance [n]']} | {row['status']} |"
        )
    section_lines.append("")
section_lines += [
    "## 测试结果边界", "",
    "当前 Top section 测试结果仅覆盖 TOP、PersonalTOP、SVM、XGBoost 和 RF。神经模型没有正式测试结果，因此不使用验证结果代替测试结果。", "",
]
(OUT / "TopSection实验汇总.md").write_text("\n".join(section_lines))

for name, frame in {
    "official_aggregates": official,
    "official_seed_results": seeds,
    "component_aggregates": components,
    "component_seed_results": component_seeds,
}.items():
    frame.to_csv(OUT / f"{name}.csv", index=False)

with pd.ExcelWriter(OUT / "实验进度.xlsx", engine="openpyxl") as writer:
    pd.DataFrame([
        ["结果快照UTC", snapshot["frozen_utc"]],
        ["进度核查UTC", progress["updated"]],
        ["协议SHA256", snapshot["protocol_sha256"]],
        ["统计格式", "mean ± sample variance (ddof=1); n=1 variance unavailable"],
        ["结果边界", "validation and test separated; no neural test substitution"],
    ], columns=["item", "value"]).to_excel(writer, "说明", index=False)
    progress_rows.to_excel(writer, "方法进度", index=False)
    queue_rows.to_excel(writer, "队列进度", index=False)
    main.to_excel(writer, "All_NDCG20验证", index=False)
    section_grid.to_excel(writer, "TopSection_NDCG20验证", index=False)
    section_matrix.to_excel(writer, "TopSection_NDCG20矩阵", index=False)
    top_section_all_metrics.to_excel(writer, "TopSection全部指标", index=False)
    top_section_test.to_excel(writer, "TopSection_NDCG20测试", index=False)
    all_metrics.to_excel(writer, "PatentFlow全部指标", index=False)
    official.to_excel(writer, "正式结果聚合", index=False)
    seeds.to_excel(writer, "正式逐种子", index=False)
    ablation.to_excel(writer, "消融聚合", index=False)
    memory.to_excel(writer, "记忆聚合", index=False)
    sensitivity.to_excel(writer, "敏感性逐种子", index=False)
    level.to_excel(writer, "层级聚合", index=False)
    level_seed.to_excel(writer, "层级逐种子", index=False)
    datasets.to_excel(writer, "12视图图统计", index=False)
    splits.to_excel(writer, "划分标签统计", index=False)

book = load_workbook(OUT / "实验进度.xlsx")
for sheet in book:
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="173A5E")
        cell.font = Font(color="FFFFFF", bold=True)
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width = min(
            42, max(12, max(len(str(c.value or "")) for c in column) + 2)
        )
book.save(OUT / "实验进度.xlsx")

audit = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "snapshot_utc": snapshot["frozen_utc"],
    "progress_utc": progress["updated"],
    "official_records": len(snapshot["records"]),
    "component_records": len(snapshot["components"]),
    "official_aggregate_rows": len(official),
    "official_seed_rows": len(seeds),
    "component_aggregate_rows": len(components),
    "component_seed_rows": len(component_seeds),
    "verified_aggregate_rows": len(official),
    "xlsx_sha256": hashlib.sha256((OUT / "实验进度.xlsx").read_bytes()).hexdigest(),
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(json.dumps(audit, ensure_ascii=False, indent=2))
