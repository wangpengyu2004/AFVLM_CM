# 异步联邦 VLM 两项机制的最小验证项目规格

## 1. 项目目标

在任务异构的异步联邦 VLM 指令微调中，用尽可能少的训练验证两个机制：

1. **上传侧：任务条件的陈旧更新干扰。** 客户端基于旧模型训练出的更新，到达当前服务器时，是否比同任务的新鲜更新更容易破坏服务器在客户端离线期间获得的其他任务能力；同样的版本差经过不同任务路径时，影响是否不同。
2. **下载侧：局部任务偏置起点造成的负迁移。** 即使每个客户端最终上传次数相同，任务相关延迟是否仍会在局部时间窗内产生快任务更新突发，使随后开始训练的慢任务客户端从不匹配的全局起点出发，并增加有限预算适配的难度。

本阶段只验证问题是否存在及其作用链条，不设计复杂的新聚合算法。实验必须允许替换模型、数据任务、客户端数、延迟、聚合器和评估指标。

补充要求：项目必须预留完整方法插件，支持后续自定义方法与 baseline；用户只编辑一个 `configs/run.yaml` 即可控制完整过程。本文第 14 节是方法扩展与配置入口的最新约定，优先于前面的拆分配置示意。目前交付的是实现规格，新增接口仍需由项目实现。

## 2. 必须守住的论断边界

- 每个客户端总上传次数相同，只能排除“完整训练期间快任务因上传总数更多而长期主导”，不能排除局部时间窗内的到达不均、中间模型偏置和顺序反馈。
- “持续到达”本身不自动等于“持续学习遗忘”。只有当某任务性能在吸收另一任务更新后发生可测回退，才能称为类似遗忘的现象。
- 若服务器采用 `w <- w + eta * alpha_i * delta_i`，并且所有 `delta_i` 与 `alpha_i` 都预先固定，则同一更新集合最终相加的结果与顺序无关。顺序效应只能来自中间状态、客户端从不同中间状态重新训练、状态相关权重，或 FedAsync 式模型插值。因此本项目不使用“固定增量换顺序后的最终模型不同”作为证据。
- 第二项挑战的主表述应为“瞬时/局部窗口任务偏置”，不是“最终快任务一定主导”。若最终仍有偏置，需要进一步证明累计有效权重不相等或闭环路径产生了不可逆影响。

## 3. 最小公共实验设置

### 3.1 模型

- 正式模型：`Qwen/Qwen2.5-VL-7B-Instruct`。
- 微调方式：只训练 LoRA；默认不训练视觉塔。
- 推荐低成本参数：
  - 4-bit NF4 QLoRA，计算类型 `bfloat16`；若硬件或环境不支持，则改为 BF16 LoRA；
  - LoRA rank `r=8`，`alpha=16`，`dropout=0.05`；
  - 目标模块由模型适配器解析，默认语言层的 `q_proj/k_proj/v_proj/o_proj`；
  - 本地学习率 `2e-4`，batch size `1`，梯度累积 `4`；
  - 每次本地训练只做 `5` 个 optimizer steps，最大文本长度 `512`；
  - 固定图像最大像素或缩放到统一尺寸，避免图片分辨率改变训练耗时。
- 冒烟模型：`tiny_mock`，在 CPU 上验证调度、版本、日志和指标公式。
- 可选快速真模型：Qwen2.5-VL 3B，仅用于调试；最终结论仍用 7B 复核。

模型名称不得散落在客户端或服务器代码中，只允许出现在模型注册表与 YAML 配置中。

### 3.2 数据与任务

推荐从 FedMLLM 当前仓库已有的数据工具中选两个差异明显且规模较小的任务：

- 快任务 `F`：Hateful-Memes，多模态二分类；
- 慢任务 `S`：VQA-RAD，视觉问答。若 VQA-RAD 获取不便，可换成 SLAKE。

这样比 Hateful-Memes + CrisisMMD 两个分类任务更容易形成任务级差异。需要如实说明：该组合同时包含任务类型和领域差异，因此第一阶段证明的是“任务源/客户端目标条件的干扰”，不能仅凭它严格分解任务类型与领域的各自贡献。以后若需要强化论文结论，再增加同领域不同目标的控制，不放入最小验证。

每个任务：

- 两个训练客户端；任务内先打乱，再等量、不重叠划分；
- 每客户端最多准备 `128` 条训练样本，但每轮只取固定批次完成 5 步；
- 固定 `16` 条 probe 样本用于频繁 teacher-forced loss；
- 固定 `64` 条 final-eval 样本用于任务原始指标；
- Hateful-Memes 报 accuracy/AUC，VQA-RAD 或 SLAKE 报 exact/官方 accuracy；
- 不跨任务直接平均原始指标。主机制分析均采用“同一任务内、更新前后”的差值。

数据层必须支持 FedMLLM 原始格式到统一样本格式的转换：

```text
Sample {
  id, task_name, image, instruction, answer, split, metadata
}
```

### 3.3 客户端与上传配额

- 客户端数：4。
- `c0,c1 -> F`，`c2,c3 -> S`。
- 每客户端上传 `2` 次，所以完整运行共有 `8` 次真实联邦更新；所有客户端严格等配额。
- 每次上传后，若仍有配额，客户端立即下载服务器当前版本并开始下一轮。
- 正式机制实验使用不重叠客户端数据；诊断性 fresh twin 使用与对应 stale update 完全相同的本地批次和随机种子。

### 3.4 延迟与事件时钟

必须使用**确定性虚拟时钟**，不通过真实 `sleep` 或 GPU 抢占制造延迟：

```text
finish_time = start_time + virtual_train_time(client, task, local_round)
              + virtual_network_delay(client, local_round)
```

默认任务相关时长：

- `c0=1.0, c1=1.2`；
- `c2=3.0, c3=3.2`；
- 网络延迟先设为 0；
- 相同配置和 seed 必须产生完全相同的下载版本、到达顺序和服务器版本序列。

虚拟时间只决定事件关系，真实 GPU 训练按计划回放。因此一个 GPU 也能模拟四个客户端，不应让线程排队改变研究中的异步时间线。

### 3.5 聚合与评估

- 机制主实验采用立即应用、固定 `server_lr=0.5`、固定权重 `1.0`，不使用 staleness decay，以免把待研究因素提前消除。
- 若单个更新导致极端崩溃，则将 `server_lr` 调到 `0.2`；该值一经校准，所有配对分支必须一致。
- 每个真实事件只在 16 条固定 probe 上计算 teacher-forced loss。
- 生成式/任务原始指标只在初始、关键分支和最终检查点计算，降低评估耗时。
- 首先跑一个 seed（建议 42）；只有看到方向一致的信号后，再对关键配对跑 `42/43/44` 三个 seed。

## 4. 实验一：上传侧陈旧跨任务干扰

### 4.1 要回答的问题

在相同接收状态、相同任务、相同本地数据与相同服务器权重下，陈旧更新是否比新鲜更新对“客户端离线期间服务器学到的任务”造成更大损害？相同版本差经过不同任务路径时，损害是否不同？

### 4.2 最小配对设计

设待测客户端属于任务 `S`，它在旧检查点 `w_s` 上训练：

```text
delta_stale = LocalTrain(S, w_s; batches, seed, 5 steps) - w_s
```

客户端训练期间，服务器应用两个 `F` 任务更新，得到接收状态 `w_t`，版本差固定为 `tau=2`。随后从同一个 `w_t` 产生新鲜孪生更新：

```text
delta_fresh = LocalTrain(S, w_t; same batches, same seed, 5 steps) - w_t
```

把 `w_t` 克隆为两个只包含 LoRA 状态的分支，使用完全相同的服务器步长：

```text
w_stale = w_t + eta * delta_stale
w_fresh = w_t + eta * delta_fresh
```

主对照必须满足：接收状态相同、更新任务相同、本地样本顺序相同、本地步数相同、随机种子相同、聚合权重相同。fresh twin 是诊断性反事实分支，不计入四个客户端的上传配额。

### 4.3 路径控制

再构造同样 `tau=2` 的第二条路径：服务器在客户端离线期间应用两个 `S` 更新，而不是两个 `F` 更新。分别在各自接收状态上生成 fresh twin。

因此最小条件为：

| 条件 | 版本差 | 离线期间服务器路径 | 到达更新 |
|---|---:|---|---|
| P1-stale | 2 | F, F | 从旧状态训练的 S 更新 |
| P1-fresh | 0 | 同一个 P1 接收状态 | 从当前状态训练的 S 更新 |
| P2-stale | 2 | S, S | 从旧状态训练的 S 更新 |
| P2-fresh | 0 | 同一个 P2 接收状态 | 从当前状态训练的 S 更新 |

这能把三件事区分开：普通多任务冲突、标量陈旧度和离线期间的任务路径。

### 4.4 指标

若任务得分越高越好，对被服务器新吸收的任务 `F` 定义：

```text
Damage_stale(F) = Score_F(w_t) - Score_F(w_stale)
Damage_fresh(F) = Score_F(w_t) - Score_F(w_fresh)
ExtraHarm(F)    = Damage_stale(F) - Damage_fresh(F)
```

若使用 loss，则方向改为：

```text
ExtraLoss(F) = [Loss_F(w_stale)-Loss_F(w_t)]
             - [Loss_F(w_fresh)-Loss_F(w_t)]
```

同时记录：

- S 任务自己的更新收益；
- `||delta_stale||`、`||delta_fresh||`；
- 更新与服务器离线期间漂移 `w_t-w_s` 的 cosine；
- 接收前后所有任务的 probe loss。

可选的低成本排查：把 stale/fresh 两个增量缩放到相同 L2 norm 后重放一次。它不是主结果，只用于判断现象是否完全来自更新幅度。

### 4.5 支持与否定标准

- 支持：在 P1 中 `ExtraHarm(F)>0`，且三 seed 或至少 4 个独立配对中的大多数方向一致；P1 与 P2 的结果存在明显差别，说明相同 `tau` 不能充分描述影响。
- 部分支持：stale 只比 fresh 更大，但 P1/P2 无差异。此时只能声称“staleness 放大干扰”，不能声称“任务路径决定干扰”。
- 不支持：stale 与 fresh 基本相同，或差异完全由增量 norm 解释。此时观测到的更可能只是普通多任务冲突。

不必一开始做显著性检验。先报告每个配对差值和三 seed 均值/标准差；样本量扩大后再做 paired bootstrap 或配对检验。

## 5. 实验二：下载侧局部任务偏置起点

### 5.1 要回答的问题

当每个客户端总上传数相同，但快任务更新在局部窗口内成簇到达时，该中间全局状态是否会使慢任务的固定预算适配更困难？

### 5.2 受控起点实验（主实验）

从同一初始模型 `w0` 生成两个只含 4 次服务器事件的检查点：

- 偏置起点 `w_bias`：最近四次任务序列为 `F,F,F,F`；
- 平衡起点 `w_bal`：最近四次任务序列为 `F,S,F,S`。

这里比较的是局部窗口内不同任务组成造成的中间状态，不比较同一组固定增量换顺序后的最终状态。四次更新均通过在线本地训练生成，或直接从端到端轨迹中截取相应检查点。

使用一个不参与构造上述检查点的 held-out `S` 客户端进行诊断：

```text
delta_bias = LocalTrain(S, w_bias; fixed batches, seed, 5 steps) - w_bias
delta_bal  = LocalTrain(S, w_bal;  same batches, seed, 5 steps) - w_bal
```

比较：

1. 训练前 S 的 probe loss/score：起点是否更不匹配；
2. 同样 5 步后的 S 性能：有限预算是否仍然落后；
3. 训练损失曲线面积或第 5 步 loss：是否更难适配；
4. `||delta_bias||` 与 `||delta_bal||`：是否需要更强纠偏；
5. 把两个纠偏更新分别应用后，F 任务是否出现更大回退：强纠偏是否反馈影响全局。

held-out 客户端只用于反事实诊断，不计入真实联邦上传次数。

### 5.3 等上传配额的端到端确认

运行两条 4 客户端、每客户端 2 次上传的完整轨迹：

- 任务相关延迟：F 客户端延迟 `1.0/1.2`，S 客户端延迟 `3.0/3.2`；
- 延迟重排控制：保持延迟集合 `{1.0,1.2,3.0,3.2}` 不变，但令每个任务各有一个快客户端和一个慢客户端，例如 F=`1.0/3.2`，S=`1.2/3.0`。

两条轨迹客户端数、客户端数据、本地步数、总上传数和延迟总体分布全部相同，只改变“任务与速度是否相关”。记录最近窗口 `W=4` 的任务到达比例：

```text
WindowMass_F(t) = 最近 4 个已应用更新中 F 的数量 / 4
```

并在每个客户端下载时记录其起点对 F/S 的 probe loss。主要看中间曲线和 S 客户端的下载时刻，不把最终得分差异自动解释为长期频率偏置。

### 5.4 支持与否定标准

- 支持：相对 `w_bal`，`w_bias` 的 S 初始性能更差，并且“5 步后性能仍更差、纠偏 norm 更大、纠偏对 F 干扰更强”三项中至少一项成立；端到端轨迹同时显示任务相关延迟产生更高的局部 `WindowMass_F`。
- 仅效率问题：初始 S loss 更高，但 5 步后完全恢复，且更新幅度/外部任务损害无差异。此时应表述为“增加适配开销”，不表述为持续负迁移。
- 不支持：受控起点下 S 的初始和训练后表现均无稳定差异。

## 6. 最小运行矩阵与计算控制

| 阶段 | 模型 | 运行内容 | 目的 |
|---|---|---|---|
| S0 | tiny_mock | E1、E2 全流程各一次 | 检查事件、分支、日志、指标公式，分钟级 |
| S1 | Qwen 7B | E1 的 P1/P2，seed=42 | 判断陈旧与路径信号是否存在 |
| S2 | Qwen 7B | E2 受控起点，seed=42 | 判断偏置起点信号是否存在 |
| S3 | Qwen 7B | 只重复 S1/S2 的关键配对，seed=43/44 | 检查稳定性 |
| S4，可选 | Qwen 7B | 两条 8 更新端到端轨迹 | 展示机制在真实异步闭环中出现 |

不要先跑大量客户端、长轮次和复杂基线。正式单 seed 训练量应控制在约 80–120 个本地 optimizer steps 内；实际耗时取决于 GPU 和图像预处理。逐事件只跑小 probe，完整生成评估只跑关键检查点。

## 7. 可替换的软件结构

在现有 `code/afl_vlm` 原型上增量实现，不重写已有虚拟调度器和聚合器。

```text
code/
  afl_vlm/
    models/
      base.py
      registry.py
      qwen25_vl.py
      tiny_mock.py
    data/
      base.py
      registry.py
      fedmllm.py
      partition.py
      collators.py
    federation/
      types.py
      client.py
      server.py
      state_store.py
    scheduling/
      base.py
      virtual_event.py
      delay_models.py
    aggregation/
      base.py
      immediate.py
      fedasync.py
      fedbuff.py
    evaluation/
      task_metrics.py
      probes.py
      paired_metrics.py
    experiments/
      stale_twin.py
      biased_start.py
      end_to_end.py
    analysis/
      summarize.py
  configs/
    run.yaml                 # 唯一必需的用户配置，完整且无需继承其他 YAML
  scripts/
    run_experiment.py
    validate_config.py
  tests/
    test_schedule.py
    test_equal_quota.py
    test_branch_restore.py
    test_additive_commutativity.py
    test_paired_metrics.py
```

现有路径可以保留；上面是职责边界，不要求为了目录美观强行搬迁所有旧文件。

### 7.1 核心接口

`ModelAdapter`：

```text
load(config)
snapshot_trainable() -> LoRAState
load_trainable(state)
local_train(task_batch_stream, train_config) -> Update
evaluate(task_adapter, sample_ids, mode) -> MetricDict
```

`TaskAdapter`：

```text
load_split(split)
format_sample(sample) -> messages/image
collate(samples, model_adapter)
metric(predictions, references)
probe_loss(model, sample_ids)
```

`DelayModel`：

```text
duration(client_id, task, local_round, rng) -> float
```

至少支持 `fixed`、`per_task`、`per_client`、`scripted`。所有延迟只产生虚拟事件，不直接休眠。

`Aggregator`：

```text
apply(global_state, update, context) -> ApplyResult
```

`ApplyResult` 必须记录实际权重、接收前版本、训练版本和接收后版本。以后更换 FedAsync/FedBuff 时，实验代码不变。

`BranchEvaluator`：

```text
evaluate_branches(receiving_state, [delta_stale, delta_fresh], tasks)
```

只克隆 LoRA 可训练状态，不复制 7B 基座。每个分支结束后恢复状态，并校验哈希，保证诊断不会污染主训练。

## 8. 推荐配置示例

```yaml
experiment:
  name: e2_end_to_end_equal_quota
  seed: 42
  timing: virtual
  output_dir: runs/e2_end_to_end/seed42

model:
  adapter: qwen25_vl
  hf_id: Qwen/Qwen2.5-VL-7B-Instruct
  dtype: bf16
  quantization: nf4
  gradient_checkpointing: true
  lora:
    r: 8
    alpha: 16
    dropout: 0.05
    include_visual: false

tasks:
  fast:
    backend: fedmllm
    name: hateful_memes
    train_per_client: 128
    probe_samples: 16
    final_eval_samples: 64
  slow:
    backend: fedmllm
    name: vqa_rad
    train_per_client: 128
    probe_samples: 16
    final_eval_samples: 64

clients:
  upload_quota: 2
  local_steps: 5
  batch_size: 1
  grad_accumulation: 4
  client_lr: 2.0e-4
  data_partition: disjoint
  assignments:
    - {id: c0, task: fast, virtual_train_time: 1.0}
    - {id: c1, task: fast, virtual_train_time: 1.2}
    - {id: c2, task: slow, virtual_train_time: 3.0}
    - {id: c3, task: slow, virtual_train_time: 3.2}

server:
  aggregation: immediate
  server_lr: 0.5
  staleness_weight: none

evaluation:
  window_size: 4
  probe_on_selected_events: true
  generation_on_selected_checkpoints: true
```

配置加载后必须进行硬校验：客户端 ID 唯一、任务存在、每客户端 quota 相同、数据划分无交集、所有关键配对的 seed/batch/steps 一致、输出目录不覆盖旧运行。

## 9. 日志与结果文件

每次运行输出：

```text
runs/<experiment>/<seed>/
  config.resolved.yaml
  schedule.json
  events.jsonl
  probes.jsonl
  updates.jsonl
  branches.jsonl
  summary.json
  summary.csv
```

`events.jsonl` 每行至少包含：

```text
event_id, virtual_time, client_id, task, local_round,
download_version, receive_version, staleness,
virtual_duration, applied_weight, recent_window_task_counts
```

`updates.jsonl` 至少包含：

```text
pair_id, update_kind(stale/fresh/normal), delta_norm,
server_drift_norm, cosine_with_server_drift, sample_ids_hash, seed
```

`branches.jsonl` 至少包含：

```text
pair_id, receiving_state_id, path_tasks, tau, branch,
task_metrics_before, task_metrics_after, extra_harm
```

最终自动生成两个小表：

1. E1：每个配对的 `Damage_stale`、`Damage_fresh`、`ExtraHarm`、路径、tau；
2. E2：`w_bias/w_bal` 的 S 初始指标、5 步后指标、delta norm、对 F 的回退。

不需要复杂可视化；最多输出两张折线/柱状图：逐事件分任务 probe 曲线，以及配对差值图。

## 10. Codex 实现顺序

### P0：先让当前原型可运行

1. 建立缺失的 `afl_vlm/data` 包，实现统一 `Sample`、FedMLLM task adapter、划分和 collator。
2. 给 Qwen2.5-VL 增加正式模型注册项与 QLoRA 配置；保留 `tiny_mock`。
3. 让以下命令只解析数据、配置和事件表，不加载 7B 模型：

```text
python -m scripts.run_experiment --config configs/run.yaml --dry-run
```

### P1：完成可复现事件层

1. 复用现有 plan-then-replay 虚拟时钟。
2. 增加 `upload_quota` 强校验、最近窗口任务计数和逐客户端下载版本日志。
3. 增加 task-delay-shuffled 配置生成器，确保只重排任务与速度的对应关系。

### P2：实现两个诊断实验

1. `stale_twin.py`：缓存旧状态、本地批次和 stale delta；在接收状态生成 fresh twin；从同一 LoRA 状态做分支评估。
2. `biased_start.py`：构造/截取 `FFFF` 与 `FSFS` 检查点；从两个起点训练同一个 held-out S 客户端。
3. 分支评估不得写回主服务器状态。

### P3：汇总与复核

1. 自动计算 E1/E2 指标并输出 CSV/JSON。
2. tiny_mock 全流程通过后，先运行一个 7B seed。
3. 只对出现信号的关键条件追加两个 seed。

## 11. 验收测试

项目完成必须同时满足：

- 同一配置和 seed 的 `schedule.json` 完全一致；
- 四个客户端各出现恰好两次真实上传；
- task-delay-shuffled 控制保持客户端数、上传数和延迟多重集合一致；
- stale/fresh twin 的任务、样本 ID、步骤数、seed 和应用权重一致；
- 分支评估前后主全局 LoRA 状态哈希一致；
- 固定 delta、固定权重的纯加法换序测试得到相同最终状态；
- tiny_mock 能在 CPU 上完成 E1/E2 并产出所有日志字段；
- 模型从 tiny_mock 换成 Qwen2.5-VL 只需改 YAML；
- 数据从 VQA-RAD 换 SLAKE 只需改 task 配置；
- 分任务指标原样保存，不因单位不同直接求平均；
- 运行失败不会覆盖已有结果目录。

## 12. 最终论文层面的最小证据链

如果实验成功，工作可以形成以下简洁证据链：

1. **同一接收状态的 stale/fresh 配对**证明损害不是普通任务冲突的重复描述；
2. **同一 tau、不同离线任务路径**证明标量 staleness 不足以描述更新风险；
3. **同一慢任务预算、不同全局起点**证明局部快任务突发会改变客户端适配难度；
4. **等上传配额的完整轨迹**证明第二个现象不依赖快客户端最终上传更多；
5. 因而异步任务异构带来的关键问题不是简单的“旧了多久”，而是“离线期间全局模型沿什么任务路径变化，以及该路径如何塑造后续更新”。

## 13. 数据来源说明

- FedMLLM paper: https://arxiv.org/abs/2411.14717
- Official FedMLLM repository: https://github.com/1xbq1/FedMLLM

官方仓库当前提供 Hateful-Memes、CrisisMMD、VQA-RAD、MedAlpaca 和 SLAKE 等数据准备/评估入口；其原始框架主要面向 MiniCPM-V，因此本项目通过适配器接入 Qwen2.5-VL，并保持联邦事件层与模型实现解耦。

## 14. 自定义方法、baseline 与单文件配置

### 14.1 方法扩展范围

方法必须能同时影响下载起点、本地训练与服务器聚合，不能仅预留一个加权平均函数。在 `afl_vlm/methods/` 下建立：

```text
methods/
  base.py                  # Method 接口，所有可选 hook 默认原样返回
  registry.py              # 方法名到实现类的显式注册
  baselines/
    async_additive.py      # 当前固定权重加性机制基线
    staleness_decay.py     # 带陈旧度衰减的加性基线
    fedavg_sync.py         # 同步参照
    fedasync.py            # 按论文核验后实现
    fedbuff.py             # 按论文核验后实现
  custom/
    my_method.py           # 自定义方法模板，默认关闭
```

`Method` 提供以下生命周期接口：

- `prepare_download(global_state, client_context)`：返回客户端起点；支持任务适配、参数混合等下载侧方法。
- `local_loss(base_loss, model, batch, context)`：加入可选正则、蒸馏等训练项；默认返回原 loss。
- `prepare_upload(update, context)`：可选变换上传更新，保留原始下载版本与基点标识。
- `on_arrival(update, server_context)`：决定立即处理或缓冲，以及聚合动作。
- `on_finish(server_context)`：按方法定义处理尾部缓冲，记录实际应用数量。
- `state_dict()/load_state_dict()`：保存方法的历史统计和缓冲状态。

方法复用现有模型、训练器和聚合原语。增加一种方法只需新增实现、注册名称并在 YAML 中启用，不修改实验主循环。初版只实现默认训练器及可选 loss hook；需要完全不同训练流程时，再通过显式 trainer 插件扩展。

自定义模板不得冒充有效新方法：未实现时显式报错，或仅在明确的 `identity_template` 名称下运行。任何论文 baseline 必须核验公式与本地训练设定；现有文件名称不能作为正确复现的证据。尤其模型插值与增量相加要分别实现，上传记录须包含实际本地起点，避免把二者混用。

### 14.2 一个文件控制完整过程

`configs/run.yaml` 包含完整的 model、tasks、clients、training、timing、methods、experiments、evaluation、output 配置；不要求用户维护配置继承链。第 8 节配置中的所有公共参数合并进此文件，并增加下面的控制段：

```yaml
run:
  seeds: [42]
  output_root: runs/validation

methods:
  - id: additive
    implementation: async_additive
    enabled: true
    params: {server_lr: 0.5}
  - id: decay
    implementation: staleness_decay
    enabled: false
    params: {server_lr: 0.5, decay_exponent: 1.0}
  - id: sync
    implementation: fedavg_sync
    enabled: false
    params: {}
  - id: ours
    implementation: my_method
    enabled: false
    params:
      download_correction: true
      upload_correction: true

experiments:
  - id: e1
    type: stale_twin
    enabled: true
    methods: [additive]
    params: {tau: 2, paths: [[fast, fast], [slow, slow]]}
  - id: e2
    type: biased_start
    enabled: true
    methods: [additive]
    params: {biased_path: [fast, fast, fast, fast], balanced_path: [fast, slow, fast, slow]}
  - id: trace
    type: end_to_end
    enabled: false
    methods: all_enabled
    params: {delay_scenarios: [task_correlated, task_shuffled]}

evaluation:
  probe_samples_per_task: 16
  final_samples_per_task: 64
  window_size: 4
  save_update_norm: true

output:
  save_adapter: true
  save_method_state: true
  overwrite: false
```

以上是附加字段示意，Codex 必须交付一份包含全部公共参数的完整 `run.yaml`，不能让用户手工拼接两个示例。各参数只有一个权威位置：如 `server_lr` 放入各方法的 `params` 后，移除旧公共 `server.server_lr`；随机种子由 `run.seeds` 统一控制。按训练、划分、延迟、评估分开派生随机数流，方法执行次序不能影响其他运行的随机性。

统一命令：

```text
python -m scripts.run_experiment --config configs/run.yaml --dry-run
python -m scripts.run_experiment --config configs/run.yaml
```

runner 展开“启用实验 × 该实验所选启用方法 × seeds × 场景”，每个组合从相同初始化重新开始；每次保存完整解析配置，并生成总汇总表。默认只启用 additive 与两个机制实验，保证初次计算量小。以后启用 baseline 和 ours 只修改此文件。

### 14.3 公平比较与实验兼容性

所有方法共享数据划分、客户端配额、本地训练预算、初始权重和外生延迟规则；每种方法重新生成自己的训练更新，不复用另一方法产生的 delta。同步等待或缓冲会改变后续下载版本及事件关系，因此不能强迫所有方法沿用同一份包含权重和版本的完整计划。只共享外生持续时间/网络延迟，按各方法的真实语义推进事件。

若方法权重依赖更新内容或服务器性能，不能预先在元数据调度阶段算死权重；应在更新真正就绪时计算并记录。需额外数据的蒸馏/回放方法必须在 YAML 声明数据来源和预算，汇总时报告额外开销。

机制分支实验默认使用固定权重 additive。带缓冲或同步方法优先参加端到端对比；若不支持单更新反事实分支，配置校验必须清晰报错，不能悄悄改成立即应用。研究自定义方法时，同一次分支配对同时恢复模型状态、优化器状态、随机状态及方法内部状态，避免分支相互污染。

### 14.4 新增验收标准

- 仅编辑一个 `run.yaml` 即可切换模型、任务分配、客户端数量、延迟、方法、实验和 seeds。
- 配置客户端数量与显式列表不一致时报错；未知字段、未知方法、实验与方法不兼容均在训练前报错。
- 添加自定义方法无需修改 server/client/experiment 主循环。
- 每个方法独立初始化模型与方法状态，结果按实验/场景/方法/seed 隔离存储。
- 方法间对比同时输出任务效果、真实上传/应用数量、训练步数与方法额外开销。
