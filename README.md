# NLPIR PatentFlow 完整复现包

本目录从原项目单独整理，包含三国原始专利数据、IPC 词典、预处理脚本、处理后的训练输入、主模型源码、12 个视图按验证集 NDCG@20 选择的最佳 checkpoint、推理入口及原始 baseline 实现。模型预测**已有公司下一年的 IPC 类别排序**，并非专利编号或新公司的冷启动预测。详细模型结构与基线见 [七模块与基线详细说明](reports/七模块与基线详细说明.md)。

## 环境与目录

建议 Python 3.8，安装 `python -m pip install -r requirements.txt`。主要目录：`datasets/` 为 CN/US/IN 原始 CSV 和词典；`scripts/`、`src/patentflow/` 为原始数据到特征的代码；`data/` 为本次实验处理结果；`main_experiment/` 为正式主模型训练、评估与配置；`checkpoints/` 为选优权重和配置；`baseline/` 为参考基线代码。仓库中大型数据和权重按 `.gitignore` 排除，因此 GitHub 克隆版须从本地完整包复制这些文件或按下述流程重建。不要将未含数据的 GitHub 克隆版视为即开即用的推理包。

## 从原始数据重建

在本目录执行，脚本会以当前目录为工作区。原始文件为 `datasets/{CN,US,IN}-all.csv` 及 `datasets/patent_{cn,us,in}/` 的 IPC 词典。依次运行：

```bash
python scripts/wait_for_inputs.py
python scripts/prepare_raw.py --languages cn us in
python scripts/build_datasets.py --languages cn us in
python scripts/train_word2vec.py --languages cn us in --workers 4 --epochs 5
python scripts/build_memory.py --languages cn us in --fast 180 --slow 1095 --novelty 1 --section All
# 对各国家 manifest.json 所列的 B、C、F 视图再逐一建立 section memory：
python scripts/build_memory.py --languages cn us in --fast 180 --slow 1095 --novelty 1 --section B
python scripts/build_memory.py --languages cn us in --fast 180 --slow 1095 --novelty 1 --section C
python scripts/build_memory.py --languages cn us in --fast 180 --slow 1095 --novelty 1 --section F
```

各脚本的 `--help` 给出国家、并行数和 memory 参数；正式模型默认 memory 为 fast=180、slow=1095、novelty=1。第一次处理前请确认 `state/input_transfer.json` 由当前目录的 `wait_for_inputs.py` 生成。处理好的本次运行输入已放在 `data/`，复现训练时可直接使用，无须再次处理。其他数据分支的生成入口有 `scripts/prepare_text_sequences.py`、`scripts/prepare_idgp.py`，供基线实验使用。协议在 `configs/protocol.json`，主模型参数在 `main_experiment/configs/v2_default.json`，每个最优权重的实际参数在 `checkpoints/<视图>/config.json`。

## 主模型训练与评估

```bash
python main_experiment/train.py --workspace . --dataset IN_All --seed 42 --tag v2_default --device cpu
python main_experiment/evaluate.py --workspace . --checkpoint checkpoints/IN_All/best.pt --output /tmp/nlpir-evaluation --split test --device cpu
```

训练输出默认在 `results/main/`；原有选优权重在 `checkpoints/`。12 个视图的种子、最佳 epoch、验证 NDCG@20 见 `checkpoints/selection.json`。选择规则为每个视图的 `v2_default` 三个种子中验证集 active NDCG@20 最大者。测试集没有参与选参。完整训练选项和消融计划见 [主实验说明](main_experiment/README.md)。

## Agent 工具调用

`infer.py` 提供可导入的 `predict(dataset, organization, top_k=20, device='cpu', checkpoint=None)`，返回 JSON 可序列化字典。`dataset` 是 `CN_All`、`US_C` 等 12 个视图之一；`organization` 是 `data/processed/<country>/organizations.json` 中的**已观察公司名称**，按空白和大小写规范化后精确匹配。示例：

```bash
python infer.py --dataset IN_All --organization 'bajaj auto limited' --top-k 10 --device cpu
```

```python
from infer import predict
result = predict('IN_All', 'bajaj auto limited', top_k=10)
# 将 result 作为 agent 工具的结构化返回值；predictions 是 rank/ipc_code/score 列表
```

输出的 `target_year` 是该数据视图的测试年份；输入仅用到 `history_cutoff_year` 年末之前的状态。`score` 是排序 logits，**不是校准概率**。当前 checkpoint 只支持其已建好的历史快照，不能直接对新传入的任意专利文本或新的未来年份实时更新；这些场景须重新执行数据处理并训练或设计在线更新流程。推理会校验 checkpoint 与处理数据指纹，不匹配会报错。Agent 应把未知公司、未知视图和缺少快照的错误返回给调用方。

## 自检

```bash
python verify_package.py
python -m pytest main_experiment/tests -q
python -m compileall -q infer.py main_experiment/src src/patentflow scripts
```

本目录采用源项目相同的相对路径布局。仓库忽略原始 CSV、派生数据、checkpoint、训练结果、缓存和压缩归档；本机完整包保留它们。
