# ACED-LoRA：固定任务异步指令微调适配

实现：`code/afl_vlm/methods/aced.py`；配置：`configs/methods/aced.yaml`。
只注册 `aced`，不再添加高度重叠的 `ace`。

## 原文与保留的机制

依据 [Mitigating Participation Imbalance Bias in Asynchronous Federated Learning,
arXiv:2511.19066v1](https://arxiv.org/html/2511.19066v1)，2025 年预印本，
附录 D 的 Algorithm a.1；参数依据附录 F.4，压缩扩展依据 F.3.3。
这里固定可核验的论文版本，不声称与未核验的作者代码逐行一致。

ACED 为每个客户端缓存一个梯度。首先在共同初始模型上收集所有客户端梯度，
以均匀均值做一次服务器下降，之后进入普通异步到达：

1. 收到客户端梯度后替换其缓存。
2. 在独立算法时钟 `t` 上，选择 `t - activity_timestamp[i] <= delay_threshold` 的客户端。
3. 使用活跃缓存的**均匀平均**做一次下降，不按样本数、任务或本地步数加权。
4. 活跃集合为空时跳过模型更新，但算法时钟仍推进。
5. 聚合/跳过之后，才把到达客户端的活动时间戳设为 `t+1`，允许下一次迭代重新参与。

核心服务器操作为 `theta -= server_lr * mean(active_cached_gradients)`。
不是 FedAsync 的端点插值，不是 FedBuff，也不是 CA²FL 的缓存差分校准。
阈值足够大、所有客户端始终活跃时，退化为全缓存 ACE 更新。

**活动年龄与真实梯度陈旧度不能混淆。** 按上述伪代码顺序，一个已被排除的客户端本次
到达后，当前聚合仍不使用它，下次才重新纳入；重置活动时间戳不等于重新计算了缓存梯度。
因此实现分别保留 `activity_ages_before`、`cached_gradient_base_versions` 和
`cached_gradient_true_staleness`。不将活动年龄上界宣传为真实梯度陈旧度的严格上界。

## 与 AFVLM-CM 的适配及边界

- 六个固定任务、现有客户端分区、完整多轮训练、虚拟 Plan、EDF 物理队列与全局评估均沿用。
  不构造持续任务流，不使用 validation/test 校准，不改客户端快慢配置。
- 只传递、缓存和更新联邦可训练状态，默认是语言 LoRA A/B；冻结 7B 和 CLIP 不参与融合。
  ACED 自己的 LoRA 参数转换成 FP32，避免较小的梯度下降被 FP16 参数量化吞掉；冻结主干
  仍使用原配置精度。这个转换不作用于其他方法的模型。
- 论文实验主要使用单 minibatch 梯度。本项目保留统一 `local_epochs`、AdamW、学习率、
  micro-batch、梯度累积和本地采样。**上传内容是本地 optimizer windows 上真实梯度的算术均值**，
  在已有反向传播后、优化器更新前读取，不额外反向，不修改 `.grad`，不使用 `delta / lr`。
  本地最终参数仍会产生，但 ACED 聚合用的是梯度缓存，不是该参数端点。
- 多步训练中的梯度来自不同本地参数位置，不能视为同一个下载模型上的无偏梯度。
  因此应在论文中称为 **ACED-LoRA 多步轨迹梯度适配**，不声称继承原文 K=1 的全部理论保证。
- 使用现有 `client_parallel` 路径，全部可见 GPU 各训练一个客户端；要求真实梯度收集，
  不强行接入历史 DDP 路径，也不修改它。`executor_policy: capability` 自动选择此路径。
  显式强制 `client_ddp` 会被配置校验拒绝。已有参考有效批量机制照常生效：参考八卡时
  DDP B4/GA4 与单客户端 worker B4/GA32 对齐到 128，并非位级数值等价。
- ACED 均衡的是客户端缓存，而不是强制六任务等权；任务内客户端样本数不同也不额外加权。
  已完成全部 Plan 工作的客户端仍可在活动阈值内被缓存重用，随后自然老化退出。

## 初始化成本、计数和评估

先在同一 `theta_0` 上为每个客户端计算**一个参考有效批量窗口**的梯度，不执行 optimizer.step
或 weight decay；不缩减普通 local-epoch 训练。全部梯度齐备后执行初始化服务器下降，
模型版本 0→1，再开始普通 Plan，首批普通客户端下载版本 1。

初始化额外产生 N 次梯度计算与客户端梯度消息，并非免费；记录在
`algorithm_initialization.json` 中的 `clients`、`wall_time_seconds`、
`server_initialization.extra_client_gradient_messages`、初始梯度范数和服务器步长。
它不伪装成普通 Plan 上传，不消耗普通客户端轮数，不触发普通上传预算评估。
比较总计算、通信和实际用时必须把这部分一起披露。当前虚拟 Plan 不额外模拟初始化耗时，
不能据此声称包含初始化的完整虚拟时延收益。

评估仍使用项目统一的 `incorporated_client_updates` 里程碑和相同六任务全局指标。
每条普通上传仅在其缓存第一次参与聚合时计入 accepted 数；以后被缓存重用不重复计数。
若到达客户端当前被排除，该上传可在下一次服务器迭代才计入；若使用前被同客户端新上传
覆盖，会明确记录 `overwritten_unincorporated_update_id`。因此收到的上传数、首次纳入数和
服务器版本号可能不同，不能相互代替。

有限 Plan 最后一条被排除的上传可能尚未使用；`pending_cached_update_ids` 和方法状态
保留它，不静默假称已经使用，也不添加原算法没有的额外服务器迭代。最终评估依然运行。
初始化引起版本偏移 1 是真实更新，不应为了日志美观减回去。

## 参数

```yaml
method:
  name: aced
  params:
    server_lr: null
    lr_scale: 0.2
    delay_threshold: 10
    cache_bits: 32
    cache_seed: 42
```

`server_lr: null` 时，`eta = lr_scale * sqrt(N / T)`，N 为客户端数，
T 为普通 Plan 到达数加 1 次初始化服务器迭代。`lr_scale=0.2` 沿用论文给出的系数；
也可显式设置正数 `server_lr`，此时 `lr_scale` 不参与计算。它不是本地 AdamW 的学习率。
`delay_threshold=10` 是论文出现过的设置，不是已在 VLM 上验证的最优阈值；
应预先声明搜索预算并仅用 validation 选择，不得按 final/test 调参。

默认 `cache_bits=32` 是未压缩版本；8 启用无偏随机舍入的全向量 min/max 仿射缓存编码，
保存 uint8 codes 与 FP32 边界/步长。论文讨论了 8-bit 无偏量化，但未规定本项目这一具体编码器；
这是可关闭的工程选择，不是声称作者使用了相同量化器。随机数使用独立 generator 和
`cache_seed + update_id` 的稳定哈希，不消耗本地训练 RNG。

完整缓存约 N×P×4 bytes，8-bit 约 N×(P+8) bytes，P 是联邦参数元素数，不是 7B 总参数数。
服务器还需要全局状态与临时梯度向量，各 GPU 的本地模型/优化器另计。
聚合在配置的 aggregation device 上执行；多张 worker GPU 不自动分摊服务器缓存。
当前直接求活跃缓存均值，每次服务器计算 O(NP)，没有假称 O(P) 全缓存增量实现。

## 运行

三档入口为 `configs/experiments/llava/afvlm_cm/{2,5,10}clients/aced.yaml`。
已下载模型、图片并准备本地数据后，推荐新建不可变 profile：

```bash
python tools/generate_system_profiles.py \
  --profile aced_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from e1_bs4_ga4_r10_s42

bash scripts/run_one.sh aced 2 aced_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh aced 5 aced_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh aced 10 aced_e1_bs4_ga4_r10_s42
```

将 `e1_bs4_ga4_r10_s42` 替换成服务器实际旧 profile 名称。只有数据、训练和系统设置匹配才
复用旧 Plan；不匹配时去掉 `--reuse_plans_from` 生成新的公共 Plan。新 profile 从当前模板生成
所有常规方法配置，不是复制旧方法配置，旧 profile 和其 Plan 不会改写。
不需要为了 ACED 改 `base.yaml`、`ours.yaml` 或 OPCM 配置。
默认 `run_baselines.sh` 列表保持原样，ACED 使用上面的独立入口启动。

## 日志与非训练验证

- `algorithm_initialization.json`：额外初始梯度工作、第一次服务器更新、实际 eta。
- `updates.jsonl`：普通上传、optimizer steps、批量信息。
- `events.jsonl` 的 `result_metadata`：活跃/排除客户端、两种年龄、缓存容量、梯度范数、
  量化误差、首次纳入 IDs、尚未使用 IDs、被覆盖 IDs 和逻辑时钟。
- 原有 checkpoint 的 `method_state`：完整缓存、活动时间戳、真实来源版本、未使用 IDs。
  这里只保存原框架已有状态，不增加断点恢复工作流。

`tests/test_aced.py` 使用张量代数验证论文伪代码顺序、重新参与、空活跃集合、真实梯度来源、
初始化/上传/评估计数、量化、状态与配置。它不调用 VLM、optimizer、forward 或训练。
静态/代数检查不能证明 GPU 显存一定充足、CUDA/NCCL 一定兼容或 VLM 最终效果达到论文水平。
方法自身隔离；公共生命周期只增加一个默认返回 None 的初始化服务器钩子，旧方法不执行更新。
