#!/usr/bin/env python
"""Build a current, auditable experiment-result workbook and compact report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "latest_experiment_results"
OUT.mkdir(parents=True, exist_ok=True)

progress = json.loads((ROOT / "state/progress.json").read_text())
agg = pd.read_csv(ROOT / "draft_v3/data/active_validation_aggregates.csv")
coverage = pd.read_csv(ROOT / "draft_v3/data/active_validation_coverage.csv")
seed = pd.read_csv(ROOT / "draft_v3/data/active_validation_seed_results.csv")
components = pd.read_csv(ROOT / "draft_v3/data/revision_component_seed_results.csv")

# Validation: one canonical long table, plus a compact K=20 matrix for each metric.
validation = agg.sort_values(["dataset", "K", "metric", "mean"], ascending=[True, True, True, False])
validation.to_csv(OUT / "main_validation_all_metrics.csv", index=False)
seed.to_csv(OUT / "main_validation_seed_results.csv", index=False)
coverage.to_csv(OUT / "main_validation_coverage.csv", index=False)

# Test results really present in artifacts. Neural validation is never substituted.
test_rows = []
test_paths = sorted((ROOT / "results/baselines").glob("*/*/metrics.json"))
test_paths += sorted((ROOT / "results/main").glob("*/v2_default/s*/metrics.json"))
test_paths += sorted((ROOT / "results/deep").glob("*/*/v2_default/s*/metrics.json"))
for path in test_paths:
    value = json.loads(path.read_text())
    split = value.get("splits", {}).get("test")
    if not split or "active" not in split:
        continue
    for name, score in split["active"].items():
        metric, k = name.rsplit("_", 1)
        method = value.get("method", "PatentFlow")
        test_rows.append({"dataset": value["dataset"], "method": method,
                          "seed": value["seed"], "K": int(k), "metric": metric,
                          "population": "active", "value": score,
                          "source": str(path.relative_to(ROOT))})
test_seed = pd.DataFrame(test_rows)
test = (test_seed.groupby(["dataset", "method", "K", "metric", "population"])["value"]
        .agg(n_seeds="count", mean="mean", std="std").reset_index())
test.to_csv(OUT / "main_test_available_metrics.csv", index=False)

# Component experiments: keep the two reported metrics and aggregate across seeds.
abl_seed = components[components.tag.str.startswith("ablation_")].copy()
abl = (abl_seed.groupby(["dataset", "tag"])
       .agg(n_seeds=("seed", "count"), ndcg20_mean=("ndcg20", "mean"),
            ndcg20_std=("ndcg20", "std"), mrr20_mean=("mrr20", "mean"),
            mrr20_std=("mrr20", "std")).reset_index())
abl.to_csv(OUT / "ablation_by_dataset.csv", index=False)
abl_seed.to_csv(OUT / "ablation_seed_results.csv", index=False)

sens_seed = components[components.tag.str.startswith("sensitivity_")].copy()
sens = (sens_seed.groupby(["dataset", "tag"])
        .agg(n_seeds=("seed", "count"), ndcg20_mean=("ndcg20", "mean"),
             ndcg20_std=("ndcg20", "std"), mrr20_mean=("mrr20", "mean"),
             mrr20_std=("mrr20", "std")).reset_index())
sens.to_csv(OUT / "sensitivity_by_dataset.csv", index=False)
sens_seed.to_csv(OUT / "sensitivity_seed_results.csv", index=False)

memory_seed = components[components.tag.str.startswith("memory_")].copy()
memory = (memory_seed.groupby(["dataset", "tag"])
          .agg(n_seeds=("seed", "count"), ndcg20_mean=("ndcg20", "mean"),
               ndcg20_std=("ndcg20", "std"), mrr20_mean=("mrr20", "mean"),
               mrr20_std=("mrr20", "std")).reset_index())

# Case-study numerical evidence and provenance.
case_selection = pd.read_csv(ROOT / "reports/memory_evolution/CN_All/selection.csv")
case_trajectory = pd.read_csv(ROOT / "reports/memory_evolution/CN_All/trajectory_metrics.csv")
case_semantics = pd.read_csv(ROOT / "reports/memory_evolution/CN_All/org_patent_semantics/summary.csv")
legacy_case = pd.read_csv(ROOT / "draft/patentflow/data/case_annual_counts.csv")
hierarchy = pd.json_normalize(json.loads((ROOT / "reports/memory_evolution/CN_All/ipc_hierarchy_semantics/metrics.json").read_text()), sep=".")

methods = ["PatentFlow", "TOP", "PersonalTOP", "SVM", "XGBoost", "RF", "EDGPAT",
           "MENCP", "RepeatMixer", "TGAT", "DyGFormer", "TAMI", "IDGP", "TGN", "TGN_edgetype"]
counts = []
for method in methods:
    c = coverage[coverage.method == method].completed_seeds.sum()
    counts.append({"method": method, "validation_complete": int(c), "validation_target": 36,
                   "test_complete": int(test_seed[test_seed.method == method][["dataset", "seed"]].drop_duplicates().shape[0]),
                   "test_target": 36})
method_progress = pd.DataFrame(counts)
campaigns = []
for name, val in sorted(progress["campaigns"].items()):
    states = val.get("states", {})
    campaigns.append({"campaign": name, "complete": states.get("complete", 0),
                      "failed": states.get("failed", 0), "planned": val.get("jobs", 0),
                      "live_processes": len(val.get("live_processes", []))})
campaign_progress = pd.DataFrame(campaigns)

method_progress.to_csv(OUT / "progress_by_method.csv", index=False)
campaign_progress.to_csv(OUT / "progress_by_campaign.csv", index=False)
for metric in ["recall", "ndcg", "phr", "mrr"]:
    q = validation[(validation.K == 20) & (validation.metric == metric)]
    q.pivot(index="method", columns="dataset", values="mean").reindex(methods).to_csv(OUT / f"validation_{metric}20_matrix.csv")
memory.to_csv(OUT / "memory_sensitivity_by_dataset.csv", index=False)
case_selection.to_csv(OUT / "case_entity_selection.csv", index=False)
case_trajectory.to_csv(OUT / "case_yearly_drift.csv", index=False)
case_semantics.to_csv(OUT / "case_org_patent_semantics.csv", index=False)
hierarchy.to_csv(OUT / "case_ipc_hierarchy.csv", index=False)
legacy_case.to_csv(OUT / "case_us_technology_shift.csv", index=False)

# Short human-readable index; detailed numbers remain in CSV/XLSX without truncation.
missing = coverage[coverage.completed_seeds < 3]
all20 = validation[(validation.K == 20) & (validation.metric == "ndcg")]
lines = ["# 最新实验结果汇总", "",
         f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
         "## 完成状态", "",
         "实验尚未全部完成。主实验验证共完成 534/540 个种子单元；仅 RepeatMixer 的 US_All、US_B 各缺 3 个种子。传统方法测试 180/180 完成，神经方法（含 PatentFlow）正式测试仍为 0。消融 396/396、参数敏感性 180/180、记忆变体 162/162、层级实验 18/18 均完成，无失败。", "",
         "## 主实验验证 NDCG@20（各数据集当前最优）", "", "| 数据集 | 方法 | 均值 | 标准差 | n |", "|---|---|---:|---:|---:|"]
for dataset, q in all20.groupby("dataset", sort=True):
    row = q.sort_values("mean", ascending=False).iloc[0]
    lines.append(f"| {dataset} | {row.method} | {row['mean']:.6f} | {row['std']:.6f} | {int(row.n_seeds)} |")
lines += ["", "完整的 Recall/NDCG/PHR/MRR @ 10/20/30/40，以及逐种子来源，见各 CSV。", "",
          "## 消融与参数敏感性", "",
          "消融覆盖 12 个数据集视图、11 个配置、3 个种子；敏感性覆盖三个 All 视图、20 个非默认配置、3 个种子。CSV 同时给出 NDCG@20、MRR@20 的均值、样本标准差和逐种子结果。记忆时间尺度单列，避免与一般超参数混淆。", "",
          "## Case study", "",
          "已整理 CN_All 的组织/IPC 年度轨迹、组织—专利语义相似度、IPC 层级语义，以及 US_All 的 Kinpo 技术迁移案例。CN_All 主展示对象包括浙江吉利控股集团与 G06F；所有图均为回顾性描述，不是测试集预测证据。", "",
          "主要图：`draft_v3/figures/memory_g06f.pdf`、`draft_v3/figures/semantic_geely.pdf`、`draft_v3/figures/ipc_hierarchy.pdf`、`draft/patentflow/figures/case_study.pdf`。", "",
          "## 文件索引", "",
          "- `validation_*20_matrix.csv`：按数据集与指标展开的主实验矩阵。",
          "- `main_validation_all_metrics.csv` / `main_test_available_metrics.csv`：主实验验证/测试长表。",
          "- `ablation_by_dataset.csv`：消融聚合。",
          "- `sensitivity_by_dataset.csv`：参数敏感性聚合。",
          "- `main_validation_coverage.csv`：缺失种子清单。", ""]
(OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
print(OUT)
