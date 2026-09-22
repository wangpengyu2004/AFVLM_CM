# 异步运行时、TrainPlan 与方法生命周期

## 采用的运行时结构

AFVLM-CM 使用主流异步联邦框架常见的职责分离：

```text
authoritative server/method (GPU aggregation in parent)
        │
        ├── virtual-clock scheduler + persisted TrainPlan
        ├── method.prepare_download() at logical start
        ├── Plan-arrival EDF dispatcher
        │       ├── visible GPU 0: persistent LLaVA worker
        │       ├── visible GPU 1: persistent LLaVA worker
        │       └── ... one worker per visible GPU
        └── method.prepare_upload() + on_arrival() at logical arrival
```

对于 `client_parallel`，父进程是唯一权威服务器，联邦可训练状态、服务器优化器、缓冲区、超网络和
方法张量状态默认保存在第一张可见 GPU。每个 worker 是一个由 `spawn` 创建的独立进程，
固定占用一张 GPU，并常驻一份完整 LLaVA-1.5-7B；worker 只执行客户端本地
训练钩子，不能直接修改服务器状态。冻结主干不会跨进程传输，每个作业只发送
LoRA/显式联邦可训练状态。worker 前向/反向使用 FP16，服务器聚合使用 GPU FP32。
进程队列边界会各进行一次小型 LoRA 状态的 CPU 中转，但不会在 CPU 上执行
FedAvg/FedAdam/MasFL/Pilot 等高频张量聚合。

该边界对应 APPFL 的 Server Agent / Scheduler / Aggregator / Trainer 分工，也对应
FLGo 在通信时先为客户端打包服务器消息、再把客户端回复交给虚拟时钟处理的
方式。当前实现没有复制两个项目的框架代码。

## 能力选择的两种物理执行器

当前可编辑配置使用 `runtime.executor_policy: capability`。`scripts/run_one.sh` 读取方法的
`MethodCapabilities.supports_client_ddp`，选择以下一种物理执行器：

- `client_ddp`：全部可见 GPU 同时训练一个客户端，每卡处理不同样本，最后一个累积
  micro-batch 同步梯度；适用于普通 local-epoch 方法和 `ours`。
- `client_parallel`：每张 GPU 训练一个不同客户端；FedCompass、FedASMU、MasFL、
  AdaMasFL 因算法依赖 local-step 分配、中途刷新或 optimizer-step 梯度轨迹而保留此路径。
  Pilot 也使用该路径，避免动态任务/客户端 adapter 的 unused-parameter 集合与重入式梯度
  检查点冲突。

执行器选择只影响物理训练。两条路径仍在逻辑 start 冻结下载状态，并按 TrainPlan 的
arrival 顺序聚合。DDP 不会因为物理完成更早而提前聚合。不过 DDP 的 `batch_size` 是每卡
batch，有效批量为 `batch_size * gradient_accumulation * world_size`，实际 optimizer steps
也按填充后的分布式数据分片计算；这些字段会写入运行输出。选择性使用不同执行器意味着
本地优化轨迹不同，论文对比必须明确披露。

## TrainPlan 不保存全局模型

持久化 TrainPlan 只包含与方法原则上独立的系统条件：

- 客户端、任务、数据集和本地轮次；
- `start_time`、`arrival_time`、网络延迟和客户端速度；
- 由 epoch 推导的预计 optimizer steps；
- 固有任务计算因子；
- FedCompass 专用的动态本地步数与 group。

源 TrainPlan 不保存 `base_version`、全局参数或“客户端必须下载哪个模型”。运行时
遇到 start 事件后，才读取当时服务器版本并调用方法的 `prepare_download`。输出
目录中的 `train_plan.json` 会追加实际 `base_version`、下载策略和（若存在）中途
刷新版本，作为审计记录；它不是下一次实验的输入 Plan。

因此，下列方法不会被错误地强制从普通全局模型开始：

| 方法 | start 时真正下发的状态 | worker 本地专属状态 | arrival 时服务器处理 |
|---|---|---|---|
| FedAvg/FedProx/FedAsync/FedBuff/FedAdam | 当前全局 LoRA | 无或损失钩子 | 各自聚合器 |
| Local-only | 该客户端上一次本地 LoRA | 无 | 只保存客户端状态 |
| Pilot | 服务器保存的客户端个性化状态 | Pilot 模型适配器 | task/text 自适应聚合 |
| UniFed-LoRA | 超网络按任务描述符条件化的 LoRA | 无 | 更新超网络并聚合 |
| MasFL/AdaMasFL | 当前全局 LoRA | dispatch 时冻结的控制变量/动量 | 更新服务器控制变量 |
| FedASMU | 当前全局 LoRA | 客户端自适应系数 | 动态陈旧更新和系数更新 |
| FedCompass | 当前全局 LoRA | 调度器分配的 local steps/group | 组内聚合 |

新增方法不应在 `main.py` 写分支。它应通过 `MethodCapabilities` 声明需求，并实现
适用的 `configure_server`（服务器 GPU 状态）、`configure_model`（worker 模型扩展）、
`prepare_download`、`client_runtime_state`、客户端训练钩子、
`prepare_upload`、`on_arrival` 或调度能力。

## 自定义方法的服务器状态

`FederatedServer.state` 和 `version` 是唯一权威全局 LoRA 状态与版本；方法在
`configure_server(initial_state, clients)` 初始化额外服务器对象，在每次
`on_arrival(update, server_context)` 中读取当前全局状态并返回一个或多个
`ServerMutation`。服务器优化器矩、缓冲区、任务原型、历史全局快照、超网络以及未来的
`server_memory/client_memory` 都应由父进程中的 Method 实例持有，而不是放到 GPU worker。

需要随最终 checkpoint 保存的额外状态必须进入 `state_dict()`，可恢复状态还必须实现
`load_state_dict()`。`ours` 已把按任务、按 LoRA module 的历史敏感性、任务权重和验证过的
LoRA rank/shape/scaling schema 作为方法状态保存；客户端只保存本次 Rank-Gate EMA。
当前项目保存最终服务器和方法状态，尚未提供从中途事件游标恢复未完成训练的入口；
实现断点续训时还需要恢复 pending jobs、完成缓存、虚拟时间游标和随机数状态。

## 逻辑时间与真实 GPU 时间

`runtime.arrival_policy: planned` 表示服务器按保存的虚拟到达顺序应用更新，而不是
按一次运行中偶然的真实 GPU 完成顺序应用。逻辑 start 事件仍立即冻结当时的
`base_version`、全局状态、方法下载状态和客户端运行时状态，但训练作业先进入候选堆；
在下一条计划 arrival 需要结果之前，运行器已经看见此前所有合法 start，再按照
`(arrival_time, start_time, event_id)` 将作业非抢占地分配给空闲 GPU。这样避免
start 顺序靠前但计划完成较晚的任务提前占满全部 GPU。

EDF 只决定真实 GPU 先算哪个已经合法启动的作业。服务器仍从
`completed_updates[event_id]` 按 TrainPlan 到达事件取结果，所以物理完成较早的更新只会
缓存，不会提前聚合；已经冻结的 `base_version` 和下载参数也不会因排队而重算。
未来 start 事件不能越过当前逻辑 arrival 提前训练，因为其下载版本可能依赖当前聚合。
因此该策略减少可避免的物理队头阻塞，但不会通过偷跑未来客户端改变论文实验语义。

`updates.jsonl` 同时记录 `worker_id`、物理 GPU、入队、分派、真实开始和完成时间，
以及 `physical_queue_wait_seconds`；
`system_stats.json` 同时报告虚拟完成时间和真实 wall-clock 秒数。worker 异常会使
整次运行失败，不会把失败客户端静默重试到另一张卡，也不会丢弃更新。

FedASMU 的中途新鲜模型请求是一个显式逻辑事件：刷新时刻为
`start_time + refresh_fraction * estimated_train_time`。服务器在该逻辑时刻冻结
新鲜状态；worker 到达指定本地步后请求并接收该快照。因此它既能使用更新后的
全局信息，又不会让物理 GPU 抖动改变方法语义。

## 不同任务每步耗时不同

客户端样本数已经通过 optimizer steps 影响预计训练量；caption、grounding 等任务
如果每个 optimizer step 仍显著更慢，应使用 `federation.task_compute_factors` 表示
任务固有计算成本。该量与 `speed_factor` 不同：前者属于任务 workload，后者属于
客户端设备能力，不能混在一起。

默认六个因子都是 `1.0`，避免在没有测量时虚构差异。完成一次 8 卡并行运行后可从
真实 worker 时间估计相对因子：

```bash
python tools/estimate_task_compute_factors.py \
  runs/llava/afvlm_cm/profiles/<profile>/2clients/fedasync/seed42
```

将输出的六个值写回 `configs/base.yaml`，然后创建一个**不复用旧 Plan**的新 profile。
生成器会把因子写入 system profile 和 TrainPlan；之后所有方法复用这一套系统条件。
若因子不同却使用 `--reuse_plans_from`，生成器会明确拒绝。

## V100 八卡 profile

当前可编辑基础配置使用 FP16，并通过 `runtime.devices: all` 自动使用全部可见 GPU。旧的
`default_e1_bs1_ga4_r10_s42` 是历史不可变单执行器/BF16 快照，不会因修改基础配置
自动变化。先基于相同 Plan 创建新的 V100 八卡快照：

```bash
python tools/generate_system_profiles.py \
  --profile v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42 \
  --reuse_plans_from default_e1_bs1_ga4_r10_s42

bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42
```

旧 profile 的 `worker_queue: fifo` 是不可变历史快照，仍会按原 FIFO 策略运行。使用新名称
创建 EDF profile 时可以复用旧 TrainPlan，从而只切换物理执行策略，不重新采样客户端
速度、网络延迟或计划到达顺序。

V100 没有原生 BF16 支持，因此新 profile 必须保持 `model.dtype: fp16`。普通运行不需要
写 GPU 编号；如果临时只允许程序看到部分卡，可以在命令前设置
`CUDA_VISIBLE_DEVICES`，程序仍会自动使用其中全部可见卡。需要将卡数作为正式实验
配置固定时，再修改 `runtime.devices` 并创建新 profile；不要改已有 profile。
