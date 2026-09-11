# AFVLM-CM

面向**任务异构异步联邦视觉语言模型（VLM）指令微调**的最小机制验证框架。项目用尽可能少的训练回答两个问题：

1. 上传侧：从旧模型训练出的任务更新，在相同接收状态下是否比 fresh twin 对离线期间新获得的能力造成更大干扰；相同标量陈旧度 `tau` 经过不同任务路径时，影响是否不同。
2. 下载侧：即使每个客户端最终上传次数相同，任务相关延迟造成的局部快任务更新突发，是否会让慢任务客户端从更不匹配的中间全局状态开始有限预算适配。

默认配置使用 `tiny_mock` 和确定性合成样本，在普通 CPU 上完成 S0 级流程验收。正式结论应把模型切换为 `Qwen/Qwen2.5-VL-7B-Instruct`，并接入真实 Hateful-Memes 与 VQA-RAD（或 SLAKE）数据。

> `tiny_mock` 只证明调度、配对、版本、日志和指标公式实现正确；其数值不能作为论文实验结论。

## 已实现能力

- 单一、完整、严格校验的 `configs/run.yaml`；模型、任务、客户端、延迟、方法、实验、种子和输出均在此处控制。
- 确定性 plan-then-replay 虚拟时钟，不使用 `sleep`，单 GPU 可顺序回放多客户端异步时间线。
- 统一 `Sample` 数据结构与 FedMLLM JSON/JSONL 适配器；训练数据按任务内打乱后等量、不重叠划分。
- `tiny_mock` CPU 适配器与 Qwen2.5-VL LoRA/QLoRA 适配器；联邦层只快照和分支 LoRA 可训练状态。
- 完整 `Method` 生命周期：下载起点、本地 loss、上传变换、到达/缓冲、结束刷新、方法状态保存/恢复。
- baseline：固定权重异步加法、陈旧度衰减、同步 FedAvg、FedAsync 模型插值、FedBuff 缓冲平均。
- E1 stale/fresh twin、同 `tau` 不同任务路径；E2 `FFFF`/`FSFS` 受控起点；等上传配额端到端轨迹与 task-delay-shuffled 控制。
- 分支前后模型、方法和随机状态恢复；哈希校验防止诊断分支污染主训练。
- 每次运行输出解析配置、调度、事件、probe、更新、分支、JSON/CSV 汇总和可选状态快照。
- 自动拒绝覆盖已有非空运行目录。

## 快速开始

要求 Python 3.10 或更高版本。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m scripts.validate_config --config configs/run.yaml
python -m scripts.run_experiment --config configs/run.yaml --dry-run
python -m scripts.run_experiment --config configs/run.yaml
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m scripts.validate_config --config configs/run.yaml
python -m scripts.run_experiment --config configs/run.yaml --dry-run
python -m scripts.run_experiment --config configs/run.yaml
```

默认 `run.yaml` 启用 additive 方法和 E1/E2，各运行一个 seed。`--dry-run` 只校验配置并展开运行矩阵/事件计划，不加载数据或 7B 模型。

若输出目录已存在，请修改 `run.output_root`。只有明确希望覆盖本地结果时才设置 `output.overwrite: true`。
临时验证也可使用 `--output-root runs/another-run`；命令行 `--overwrite` 与 YAML 开关一样需要显式给出。

## 一份 YAML 控制完整流程

[`configs/run.yaml`](configs/run.yaml) 是唯一必需的用户配置：

| 配置段 | 作用 |
|---|---|
| `run` | seeds 和总输出目录 |
| `model` | 适配器、HF 模型、精度、量化、LoRA 与 mock 参数 |
| `tasks` | fast/slow 数据后端、任务名、每客户端样本数、数据文件 |
| `clients` | 客户端数量、统一上传配额、任务分配、基础虚拟训练时长 |
| `training` | 本地步数、batch、梯度累积、学习率、最大文本长度 |
| `timing` | 虚拟训练/网络延迟；均支持 fixed、per-task、per-client、scripted |
| `methods` | 方法注册名、开关与各自参数；`server_lr` 仅在方法参数中定义 |
| `experiments` | E1、E2、trace 开关、方法选择与机制参数 |
| `evaluation` | probe/final 样本数、局部窗口及评估开关 |
| `output` | 状态保存与覆盖策略 |

runner 展开：

```text
启用实验 × 该实验选中的启用方法 × seeds × 场景
```

每个组合重新初始化模型和方法。划分、训练、评估与时间分别使用派生随机流，因此方法排列不会改变另一方法的外生随机性。

## 切换到 Qwen2.5-VL 7B

正式训练推荐在带 CUDA 的 Linux 环境运行：

```bash
python -m pip install -e '.[dev,qwen]'
```

只修改 `configs/run.yaml`：

```yaml
model:
  adapter: qwen25_vl
  hf_id: Qwen/Qwen2.5-VL-7B-Instruct
  dtype: bf16
  quantization: nf4
  gradient_checkpointing: true
  device_map: auto
  lora:
    r: 8
    alpha: 16
    dropout: 0.05
    include_visual: false
    target_modules: [q_proj, k_proj, v_proj, o_proj]
```

NF4 依赖 CUDA/bitsandbytes。不支持时将 `quantization` 改为 `none`，保留 BF16 LoRA。默认只训练语言层投影；如需视觉塔 LoRA，必须显式列出经模型核验的视觉模块名。

Qwen 模式禁止使用合成数据，并要求每个任务提供 `train/probe/final` 三个文件：

```yaml
tasks:
  fast:
    backend: fedmllm
    name: hateful_memes
    train_per_client: 128
    allow_synthetic: false
    data_files:
      train: data/hateful_memes/train.jsonl
      probe: data/hateful_memes/probe.jsonl
      final: data/hateful_memes/final.jsonl
  slow:
    backend: fedmllm
    name: vqa_rad       # 改为 slake 即可切换任务
    train_per_client: 128
    allow_synthetic: false
    data_files:
      train: data/vqa_rad/train.jsonl
      probe: data/vqa_rad/probe.jsonl
      final: data/vqa_rad/final.jsonl
```

训练 split 除客户端分区外还需至少保留 `local_steps × grad_accumulation` 条样本，供 held-out S 客户端反事实诊断。默认即每任务至少 `2 × 128 + 20 = 276` 条训练样本。

## 数据格式

内部统一格式为：

```text
Sample {id, task_name, image, instruction, answer, split, metadata}
```

FedMLLM 适配器支持 `.json` 数组、带 `data/samples/annotations` 容器的 JSON，以及一行一个对象的 `.jsonl`。可直接读取规范字段：

```json
{
  "id": "sample-001",
  "image": "images/001.png",
  "instruction": "Is this meme hateful? Answer yes or no.",
  "answer": "no",
  "metadata": {"label": 0}
}
```

也兼容常见 FedMLLM/MiniCPM-V 字段 `question`、`text`、`label`、`image_path`、`img` 和 `conversations/messages`。相对图片路径按数据文件所在目录解析。样本 ID 必须唯一；客户端分区如有交集会在训练前失败。

Hateful-Memes 原样保存 accuracy/AUC；VQA-RAD/SLAKE 保存 exact/official accuracy。框架不会把不同任务、不同单位的原始指标直接平均，机制判断使用同一任务的前后差值。

## 实验说明

### E1：上传侧 stale/fresh twin

对每条路径：

1. 在旧状态 `w_s` 用固定 S 样本、seed 和 5 步生成 `delta_stale`。
2. 服务器在线吸收两次路径任务更新，得到 `w_t`，因此 `tau=2`。
3. 在同一个 `w_t`、同一 S 批次、同一 seed 和步数生成 `delta_fresh`。
4. 从同一个 `w_t` 分别应用 stale/fresh delta，使用相同 `server_lr`。
5. 报告 `Damage_stale`、`Damage_fresh`、`ExtraHarm`、S 自身收益、delta norm、服务器漂移 norm 和 cosine。

默认比较 `F,F` 与 `S,S` 两条相同 `tau` 路径。fresh twin 是诊断更新，不计入真实客户端上传配额。

### E2：下载侧偏置起点

分别用在线本地训练构造 `FFFF` 的 `w_bias` 与 `FSFS` 的 `w_bal`。一个未参与构造的 held-out S 客户端从两个起点使用完全相同的样本、seed 和 5 步预算，比较：

- S 初始和训练后 loss/score；
- 第 5 步 loss 与平均损失曲线面积；
- 纠偏 delta norm；
- 以相同服务器权重应用纠偏后对 F 的回退。

### Trace：等上传配额端到端确认

将 `trace.enabled` 改为 `true` 后运行两条 4 客户端、每客户端 2 次上传的轨迹：

- `task_correlated`：F=`1.0/1.2`，S=`3.0/3.2`；
- `task_shuffled`：保持延迟多重集合完全不变，但让每个任务各含一个快/慢客户端。

每个到达事件记录最近窗口 `W=4` 的任务计数、`WindowMass_F`、下载/接收版本、staleness 和客户端下载时的分任务 probe。

## 方法语义与扩展

内置方法：

- `async_additive`：`w <- w + server_lr × delta`，E1/E2 的固定权重主机制基线。
- `staleness_decay`：加法更新权重乘 `(tau + 1)^(-p)`。
- `fedavg_sync`：同一 local round 的全部客户端到齐后平均更新；下一轮使用同步 barrier 时间线。
- `fedasync`：按论文执行 `w <- (1-alpha_t)w + alpha_t × w_local`，不是把 raw delta 当作模型插值。
- `fedbuff`：收满 `K` 个客户端 delta 后取均值并执行服务器步；可配置是否刷新尾部 buffer。

缓冲/同步方法默认只参加端到端比较。E1/E2 要求同一接收状态和固定相同应用权重，配置校验会拒绝不兼容方法，而不会静默改变方法语义。

新增方法只需：

1. 在 `code/afl_vlm/methods/` 下继承 `Method`；
2. 实现需要的 `prepare_download`、`local_loss`、`prepare_upload`、`on_arrival`、`on_finish`、状态保存/恢复 hook；
3. 使用 `@register_method("name")` 注册；
4. 在 `run.yaml` 新增并启用该方法。

`methods/custom/my_method.py` 是明确报错的模板，不会冒充有效方法。实现完成前启用它会在模型加载和训练前失败。

## 输出结构

每个运行隔离在：

```text
runs/validation/<experiment>/<scenario>/<method>/seed<seed>/
  config.resolved.yaml
  schedule.json
  events.jsonl
  probes.jsonl
  updates.jsonl
  branches.jsonl
  summary.json
  summary.csv
  adapter_state.json       # 可关闭
  method_state.json        # 可关闭
```

`events.jsonl` 包含事件 ID、虚拟时间、客户端/任务/本地轮次、下载/接收版本、staleness、虚拟时长、实际权重和最近窗口任务计数。`updates.jsonl` 包含 stale/fresh/normal 类型、delta/drift norm、cosine、样本 ID 哈希、seed、步数和损失曲线。`branches.jsonl` 保存接收状态哈希、路径、tau、分支及任务内前后指标。

跨 seed 汇总：

```bash
python -m scripts.summarize_results runs/validation
```

该命令只聚合顶层数值机制指标，输出 `aggregate_summary.json/csv`；不会把嵌套的不同任务原始指标混合平均。

## 验证

```bash
python -m pytest
ruff check .
ruff format --check .
```

测试覆盖：

- 同配置/seed 的调度完全一致；
- 四客户端各恰好两次真实上传；
- 延迟重排保持客户端数、上传数和延迟多重集合；
- stale/fresh 批次哈希、seed、步数和权重配对；
- 分支后主状态哈希恢复；
- 固定 delta/权重纯加法换序最终状态相同；
- FedAsync 使用模型插值，FedBuff 达到 buffer 大小时才应用均值；
- `tiny_mock` 完成 E1/E2 及两条 trace，并产出要求的日志字段；
- 未知字段、客户端数量不一致和不兼容实验/方法在训练前报错；
- 非空结果目录默认不可覆盖。

## 项目结构

```text
code/afl_vlm/
  models/          # ModelAdapter、状态代数、tiny_mock、Qwen2.5-VL
  data/            # Sample、FedMLLM 适配、划分、collator
  federation/      # Client、Server、版本状态库、记录类型
  scheduling/      # 虚拟事件与四类延迟模型
  aggregation/     # 加法、FedAsync、FedBuff 原语
  methods/         # 完整生命周期、baseline、自定义模板
  evaluation/      # 任务指标、probe、配对与分支评估
  experiments/     # E1、E2、端到端轨迹
  analysis/        # 跨 seed 汇总
configs/run.yaml   # 唯一用户配置
scripts/           # 配置校验、统一 runner、结果汇总
tests/             # 机制与验收测试
```

## 论断边界

- 等上传总配额只排除完整运行中快任务因总上传更多而长期主导，不排除局部到达不均和顺序反馈。
- 只有吸收另一任务更新后原任务性能可测回退，才可描述为类似遗忘；“持续到达”本身不等于持续学习遗忘。
- 固定 delta 与固定权重的纯加法最终结果与顺序无关；顺序证据必须来自中间状态、重新训练路径、状态相关权重或模型插值。
- Hateful-Memes + VQA-RAD 同时改变任务类型和领域，第一阶段只能证明“任务源/客户端目标条件的干扰”，不能分解类型与领域的独立贡献。
- 单 seed 先看方向；出现信号后再把 `run.seeds` 改为 `[42, 43, 44]`，报告每个配对以及均值/标准差。

## 参考

- [FedMLLM 论文](https://arxiv.org/abs/2411.14717) / [官方代码](https://github.com/1xbq1/FedMLLM)
- [FedAsync: Asynchronous Federated Optimization](https://arxiv.org/abs/1903.03934)
- [FedBuff: Federated Learning with Buffered Asynchronous Aggregation](https://arxiv.org/abs/2106.06639)
- [Qwen2.5-VL 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Transformers Qwen2.5-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
