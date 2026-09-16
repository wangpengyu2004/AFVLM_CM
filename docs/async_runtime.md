# 异步运行时、TrainPlan 与方法生命周期

## 采用的运行时结构

AFVLM-CM 使用主流异步联邦框架常见的职责分离：

```text
authoritative server/method (GPU aggregation in parent)
        │
        ├── virtual-clock scheduler + persisted TrainPlan
        ├── method.prepare_download() at logical start
        ├── FIFO dispatcher
        │       ├── visible GPU 0: persistent LLaVA worker
        │       ├── visible GPU 1: persistent LLaVA worker
        │       └── ... one worker per visible GPU
        └── method.prepare_upload() + on_arrival() at logical arrival
```

父进程是唯一权威服务器，联邦可训练状态、服务器优化器、缓冲区、超网络和
方法张量状态默认保存在第一张可见 GPU。每个 worker 是一个由 `spawn` 创建的独立进程，
固定占用一张 GPU，并常驻一份完整 LLaVA-1.5-7B；worker 只执行客户端本地
训练钩子，不能直接修改服务器状态。冻结主干不会跨进程传输，每个作业只发送
LoRA/显式联邦可训练状态。worker 前向/反向使用 FP16，服务器聚合使用 GPU FP32。
进程队列边界会各进行一次小型 LoRA 状态的 CPU 中转，但不会在 CPU 上执行
FedAvg/FedAdam/MasFL/Pilot 等高频张量聚合。

该边界对应 APPFL 的 Server Agent / Scheduler / Aggregator / Trainer 分工，也对应
FLGo 在通信时先为客户端打包服务器消息、再把客户端回复交给虚拟时钟处理的
方式。当前实现没有复制两个项目的框架代码。

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

## 逻辑时间与真实 GPU 时间

`runtime.arrival_policy: planned` 表示服务器按保存的虚拟到达顺序应用更新，而不是
按一次运行中偶然的真实 GPU 完成顺序应用。物理上空闲 worker 以 FIFO 接收已触发
的客户端作业；最多八个客户端同时真实训练。某个任务真实训练较慢时，它会增加
总 wall-clock 时间，但不会悄悄改变不同方法的系统异构条件和更新顺序。

`updates.jsonl` 同时记录 `worker_id`、物理 GPU、真实开始和完成时间；
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
  --profile v100_fp16_8gpu_e1_bs1_ga4_r10_s42 \
  --reuse_plans_from default_e1_bs1_ga4_r10_s42

bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_e1_bs1_ga4_r10_s42
```

V100 没有原生 BF16 支持，因此新 profile 必须保持 `model.dtype: fp16`。普通运行不需要
写 GPU 编号；如果临时只允许程序看到部分卡，可以在命令前设置
`CUDA_VISIBLE_DEVICES`，程序仍会自动使用其中全部可见卡。需要将卡数作为正式实验
配置固定时，再修改 `runtime.devices` 并创建新 profile；不要改已有 profile。
