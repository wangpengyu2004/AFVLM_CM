# Baseline 方法内 DDP 适配（2026-10-18）

只修改相应方法及检查/说明，不改公共训练器、`models/llava15.py`、执行器选择、
Scheduler、评估、数据、`base.yaml` 或旧 profile。`ours`、OPCM 的历史 DDP 路径不改。

## 默认执行方式

| 方法 | 能力策略下的执行方式 | 方法内部处理 |
|---|---|---|
| fedasmu | 全部可见卡训练同一客户端（DDP） | 反馈归约、rank0 请求策略 |
| fedcompass | DDP | 原调度器分配的真实 optimizer steps 不变 |
| masfl / adamasfl | DDP | 原 SGD/归一化梯度钩子；初始化绑定真实 rank |
| aced | DDP | 初始化绑定真实 rank；缓存真实同步梯度均值 |
| unifed_lora | DDP | 同步本地 LoRA；每 rank 仍有完整服务器超网络 |
| pilot | 仍为每卡一个客户端 | 动态阶段/任务训练参数不兼容当前固定 DDP reducer |
| 其他已有 DDP 方法（含 ours、OPCM） | 原 DDP | 不修改 |

遵循 `CUDA_VISIBLE_DEVICES`，有八张可见卡就启动八个 ranks，不硬编码卡号。
`runtime.backend: client_parallel` 加 `executor_policy: capability` 是能力选择入口；
`select_runtime_backend.py` 返回 `client_ddp` 后，`run_one.sh` 自动使用 torchrun。
有意保留客户端并行时，在新的配置中设 `executor_policy: fixed`、`backend: client_parallel`。

## FedASMU：完整反馈与策略同步

新增 `methods/fedasmu_ddp.py`，仅在 FedASMU 自己的模型实例上安装反馈桥接，不修改模型类。
非 DDP 调用仍走原单卡函数。普通训练沿用历史 AdamW DDP，refresh 时刻、请求步骤、
base version、到达顺序与聚合公式不变。

额外元目标在底层未包装网络上临时 eval 前向，用 `autograd.grad` 求方向导数，然后在 GPU
显式归约损失和导数三个标量，取 rank 均值；不写参数 `.grad`，不执行 optimizer.step，
finally 恢复 LoRA 和全部模块训练模式。普通训练的 checkpoint、DDP 包装及 reducer 不改。
各 rank 因而得到相同 beta/reward、本地控制量和服务器控制量，不各自按不同目标漂移。
LSTM/Q 请求策略只由 rank0 选择、学习，广播小型请求元数据；rank0 的原有方法检查点保存
权威策略，其他 ranks 不重复更新策略。

这不依赖 DDP 自动同步 `autograd.grad`：额外目标不调用 DDP forward，并显式归约方向导数。
PyTorch 对 DDP 常规反向、`autograd.grad` 及重入 checkpoint 的限制见
[官方说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

本地反馈现在是各 rank 最近 micro-batch 损失的均值，而不是只看一张卡。
上传后的服务器目标 probe 在各 rank 使用同一组 seeded 私有训练记录，保留 `feedback_samples`
定义；重复物理计算单独记录。新增 `objective_reduction`、`objective_batch_scope`、
`objective_world_size`、`physical_objective_forwards`、`physical_objective_backwards`、
`physical_batch_records`。不伪造上传/optimizer steps，不引入服务器校准集。
私有训练目标依然是原文全局目标的已披露代理；本次 DDP 不消除这个理论差异。

### FedASMU 反馈阶段 OOM 修复

旧反馈虽然关闭 dropout，却也因为 `eval()` 绕过了 Transformers 的训练模式梯度
检查点。7B decoder 的额外求梯度前向因此保存全层激活；这不是把单卡 GA32 错当成
DDP batch32。`feedback_samples: 4` 的到达反馈原本就是每 rank 重复同4条记录，
与普通训练的 batch/梯度累积独立；刷新反馈则读取各 rank 最近的真实 micro-batch。

现在只在 FedASMU DDP 的额外反馈梯度阶段，为当前模型实例的每个 decoder layer
临时安装 `checkpoint(..., use_reentrant=False)`。它支持项目的 PyTorch2.0.1
`autograd.grad` 和关键字输入；wrapper 一直保留到求导/重计算结束，异常时也恢复
原 forward。全程保留 eval/dropout-off，不调用 DDP forward、不写 `.grad`，
不把模型、激活或运算迁到 CPU。损失前向显式关闭 KV cache、attention/hidden输出。
依据：[PyTorch2.0.1 checkpoint 源码及接口说明](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/utils/checkpoint.py)。

不减少 `feedback_samples`、不截短文本、不改变反馈目标、rank均值归约、普通
batch/累积/optimizer steps、Plan、聚合或其他方法。以额外逐层重计算换取激活显存，
不保证任意文本长度或批量下32GB卡都足够，也不保证指标逐位不变。新增日志
`objective_per_rank_batch_records`、`objective_activation_checkpointing`、
`objective_checkpointed_layers`；原 forwards/backwards 计数仍表示顶层目标调用次数，
不是每层重计算次数。无需重生成 Plan/profile 即可应用代码修复，但失败输出应保留
并使用新的空输出目录重新正式运行；框架不会据此自动恢复中断训练。

## 初始化与特殊优化器

共享初始化作业不绑定 DDP rank。MasFL/AdaMasFL 和 ACED 在各自
`configure_local_training` 中绑定真实 rank/world size，避免八卡重复同一数据分片。

- MasFL/AdaMasFL：初始固定模型梯度仍覆盖原 local-epoch 工作，不执行 optimizer.step。
  普通训练保留 SGD/归一化 SGD，不落入历史 AdamW 分支。真实梯度在同步后收集，再执行
  原控制变量钩子，不重复 all-reduce。
- ACED：仍只用一个参考有效批量做每客户端初始化，随后执行原首步服务器下降。
  普通 AdamW DDP 已支持同步梯度收集；缓存筛选、时间戳、首纳入计数均不变。
- FedCompass：原分组、速度、K 上下限和真实 K 不变；不把 K 因八卡再乘八或除八。

## UniFed-LoRA / Pilot 边界

UniFed 的客户端只训练普通同构 LoRA，DDP 同步这些梯度。服务器超网络按原固定 seed
初始化，各 rank 根据相同客户端结果独立执行相同 VJP 元更新，不放进客户端 DDP。
**这是复制，不是显存分片**：每卡同时存放 LLaVA、本地激活和完整超网络。
30/60 客户端的完整输出头可能仍需约30/60 GiB训练状态，32GB卡不保证能跑。
保留原显存预估/拒绝，不自动降级算法、转CPU或改超参数。显式 `hyper_device: cpu`
也会有多进程副本成本；默认 auto 使用各 rank 自己的设备。

Pilot 的阶段/任务会改变 `requires_grad`。本次不强改公共 `find_unused_parameters=False`
来迁就它，也不把它与重入 checkpoint 强行组合，故继续用客户端并行。

## 公平性、旧快照与启动

参考8卡的 `reference_ddp` 快照下，B4/GA4 的有效批量仍为128，普通 epoch 方法真实
optimizer steps 与此前 worker B4/GA32 对齐。保留原采样/补齐/尾窗口；不承诺单卡与
DDP 逐位一致，dropout、归约顺序和 FedASMU 本地反馈观测均有差异。
`ours`/OPCM 既有八卡实现没有被改动。

能力来自方法代码而不是 YAML 版本锁，因此这六个方法的旧能力策略 profile 在升级后
也会选 DDP。缺少参考元数据的 `per_device` 旧快照会因此改变有效批量，不能直接用于
公平对照。应新建参考8卡快照；已有正确参考元数据的快照可使用，但建议新输出目录。
数据、训练和系统参数都不变时可复用旧普通 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile baselines_ddp_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from e1_bs4_ga4_r10_s42

python tools/select_runtime_backend.py \
  --config experiment_profiles/baselines_ddp_e1_bs4_ga4_r10_s42/configs/2clients/fedasmu.yaml

bash scripts/run_one.sh fedasmu 2 baselines_ddp_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh fedcompass 2 baselines_ddp_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh masfl 2 baselines_ddp_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh adamasfl 2 baselines_ddp_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh aced 2 baselines_ddp_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh unifed_lora 2 baselines_ddp_e1_bs4_ga4_r10_s42
```

旧 profile 名替换为真实名称；条件不一致则去掉 reuse。在八卡服务器生成自动保存参考8。
新快照从当前模板生成全部方法配置，不修改旧快照，也不复制旧快照的各方法参数。
FedCompass仍基于同一 system profile 编译它自己的方法调度。

## 验证边界

仅做能力选择、YAML、真实 rank 分片/128批量、算法代数、反馈归约、策略所有权、实例隔离
与历史 DDP AST 回归。未加载7B、未执行训练forward/backward、未启动NCCL。
非训练检查不能证明真实八卡兼容、显存一定充足或最终指标正确，需正式服务器运行验证。
