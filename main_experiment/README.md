# PatentFlow 主实验

本目录是主模型的唯一维护入口，可独立复制。包含模型、训练、全候选评估、消融开关和实验配置；不包含 baseline、原始数据清洗或 Word2Vec/memory 预处理。

输入是实验已经生成的年度计数、合作索引、静态 IPC 语义锚点和因果 memory。重新训练时必须使用相同的输入文件、协议、配置和种子。模型代码不导入本目录外的 Python 文件。

## 目录

```text
main_experiment/
  train.py                    # 正式训练入口
  evaluate.py                 # 已选 checkpoint 的独立评估
  run_campaign.py             # 有资源限制的批量训练队列
  configs/                    # 默认参数、36 组主实验、消融、敏感性计划
  src/patentflow_main/
    model.py                  # PatentFlow 模型及模块开关
    train.py                  # 训练循环、验证选优、向量导出
    dataset.py                # 已处理年度数据读取
    neural_data.py            # 已处理 memory 读取与 batch 组装
    features.py               # 主模型频次分支所需的 16 维历史特征
    metrics.py / evaluate.py  # 统一排名指标与逐查询输出
    checkpoint_evaluation.py  # 独立 checkpoint 评估
    resources.py / io.py      # 资源采样、路径与源码记录
  docs/                       # 输入契约、方法、迁移核验
  tests/                      # 核心行为测试
  pyproject.toml / requirements.txt
```

## 运行

本机使用已安装的 `pytorch` 环境，不需要重新安装依赖：

```bash
conda activate pytorch
cd main_experiment
python train.py --dataset IN_All --seed 42 --tag v2_default --device cuda:0 --resume
```

默认读取上一级项目的 `data/`，输出到该项目 `results/main/`。已有完成记录不重复训练。复制到其他位置后，显式指定包含已处理 `data/` 的工作区：

```bash
python train.py --workspace /path/to/prepared_workspace \
  --results-root /path/to/main_results \
  --config configs/v2_default.json \
  --dataset CN_All --seed 2026 --tag v2_default --device cuda:0
```

`--workspace` 与 `--results-root` 也可分别通过环境变量 `PATENTFLOW_WORKSPACE`、`PATENTFLOW_RESULTS_ROOT` 设置。CLI 覆盖 JSON 默认参数。训练参数以每次运行的 `config.json` 为准。

批量运行同一套 12 数据集 × 3 种子：

```bash
python run_campaign.py --plan configs/main_campaign.json --gpus 0 1
python run_campaign.py --plan configs/ablation_campaign.json --cpu-slots 4
python run_campaign.py --plan configs/sensitivity_campaign.json --cpu-slots 2
python run_campaign.py --plan configs/memory_campaign.json --cpu-slots 2
python run_campaign.py --plan configs/level_campaign.json --cpu-slots 2
```

同一个计划只启动一个 supervisor；已有运行中的队列无需重复启动。批量入口共享工作区的完成标记、日志和状态文件。若需要全新复现实验，应使用新的工作区结果目录或新的实验 tag；训练和单次评估支持独立 `--results-root`，当前批量计划的完成标记使用工作区内 `results/main/`。

## 评估已有模型

独立评估不会修改原训练目录：

```bash
python evaluate.py --workspace /path/to/prepared_workspace \
  --checkpoint /path/to/main_results/IN_All/v2_default/s42/best.pt \
  --output /path/to/replay_validation --split validation --device cpu
```

正式流程只按验证 NDCG@20 选择 epoch 和超参数。参数锁定后才用 `--split test` 评估测试。训练入口的 `--evaluate-test --resume` 可生成正式 `metrics.json`，`--export --resume` 可导出最佳 epoch 下、按年份排列的节点向量。输出有源数据和协议指纹，不能将不同协议结果混入同一表。

## 模块和层级

例如 `--disable cooperation` 去掉合作模块，`--disable text hierarchy anchor alignment` 去掉文本和全局语义相关路径。支持 `text`、`structure`、`hierarchy`、`cooperation`、`community`、`repeat`、`anchor`、`alignment`。默认保持当前正式实验计算方式。

`--level 1/2/3` 控制 section/class/subclass 输出层级，默认 3。一级候选数量通常小于 40，指标按 `K_eff=min(K,候选数)` 处理，不能与三级指标难度等同。

`--fast`、`--slow`、`--novelty` 读取对应预先构建的 memory 缓存，不在本目录重新处理原始数据。调整这些参数之前，须准备匹配的缓存，见[输入契约](docs/input_contract.md)。

## 复现范围

训练循环、损失、候选集合、排序规则和模型张量名称均保持迁移前版本。CPU 同种子一轮训练的全部模型张量、排名和 24 个逐查询指标已核验完全相同；已有最佳 checkpoint 的全候选评估也完全相同，详见[迁移核验](docs/migration_validation.md)。这不表示不同 GPU、CUDA 版本或中断位置下均保证逐位一致；已有 `--resume` 从最佳 checkpoint 恢复的语义保持不变。

每次训练额外保存 `source_manifest.json`、`invocations.jsonl` 和 `source/<hash>/` 代码快照，便于追踪后续修改。计时与资源文件来自真实运行，不是固定复现目标。

测试：`python -m pytest tests -q`。若在新环境安装，可用 `python -m pip install -r requirements.txt`；本机核验环境为 Python 3.8.13、PyTorch 2.2.2，依赖版本见 requirements。
