# PatentFlow: temporal patent IPC forecasting

工作目录：`/home/xuxj/2026/workspace/nlpir2026`。Python：
`/home/xuxj/.conda/envs/pytorch/bin/python`（当前为 3.8）。

预测对象是公司的未来 IPC 类别集合，不是专利文献编号。原始参考实现保留在
`baseline/`；解压副本在 `baseline/vendor/`，不会覆盖原文件。

正式输入是 `datasets/{CN,US,IN}-all.csv`，等待文件传输完成后重新处理。
IPC 词典只读取对应 `datasets/patent_*` 下三个 IPC 文件（US/IN 的文件名
带 `_US` 后缀）。旧子集及其审计不属于正式实验数据。

## 续跑

先读 `state/PROJECT_MEMORY.md` 和 `state/progress.json`，再查
`reports/`、`results/`、`logs/`。统计与结果必须由代码生成；未完成实验不能
填入论文结果。已有根目录的两份 feasibility report 属于另一项目，不能作为
本任务统计。

## 目录

- `configs/`：任务协议、模型和实验清单。
- `src/patentflow/`：数据、指标、模型和实验实现。
- `scripts/`：可重跑入口。
- `data/processed/`：本项目新生成数据，保留原始 `datasets/`。
- `reports/`：审计、统计及图表解释。
- `results/`：逐次实验指标、预测、资源记录和 checkpoint。
- `docs/`：方法、协议、参考实现差异及实验设计。
- `state/`：断点记录与关键决策。
- `draft/`：用户提供的模板；完整实验完成后开始事实驱动的论文初稿。

## 基本约束

1. 时间按公开日期处理，同一天所有交互共享更新前状态。
2. 不按验证/测试期活跃与否筛选公司，不使用未来合作关系。
3. Word2Vec 仅使用训练专利文本；IPC 词典单独作为外部静态知识。
4. 统一全候选排序指标，种子 42、2026、3407，验证集选参，测试集最终评估。
5. 不终止已有 GPU 任务；先测显存和耗时，再排队。
6. 原始 baseline 的真实适配与简化方法分开命名，不用简化模型冒充。
