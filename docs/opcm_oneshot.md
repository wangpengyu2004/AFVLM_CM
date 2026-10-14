# One-shot federated OPCM

这是独立对比方法 `opcm_oneshot`，不是替换原异步 `opcm_lora`。
注册表用 `requires_one_shot` 标明其协议，普通 profile 生成器和默认 baseline
批处理不包含它。现有模型、DDP 本地训练、客户端 worker、runner 和评估实现不改动。

## 实验协议

1. 每个客户端使用完全相同的 LLaVA checkpoint、共同初始 LoRA A/B 和冻结主干。
   所有训练都从 version 0 出发，客户端任务与本地数据划分保持不变。
2. 在本地连续训练多个 epoch，训练期间不接收其他客户端更新、不刷新全局模型。
   默认 `local_epochs: 10`，使用共同 AdamW、学习率、batch 和梯度累积设置。
3. 每个客户端只提交一次完整的联邦 LoRA 训练状态，不能重复上传或进入第二轮。
   2/5/10 clients/task 对应12/30/60次上传，而不是120/300/600次。
4. 服务器只收集，不边收边更新；收齐 ALL clients 后，才开始离线顺序 OPCM 合并。
   不允许缺失客户端时偷偷使用部分模型完成实验。
5. 合并内部有 N 次专家处理，但只产生 ONE 次对外全局聚合/version 增长。
   内部 OPCM 步骤不冒充客户端上传，也不把合并结果回传给训练客户端。
6. 合并完成后，使用同一个最终全局模型评估六个任务的 final/test 集。
   每任务指标、预测分片汇总、DDP/parallel评估继续使用现有实现，不平均GPU指标。

运行时复用现有一轮同步 Plan 编译接口表达收集屏障。上传本身可以有不同虚拟到达
时间，但不构成持续更新的异步联邦优化；全部上传的 `base_version=0`、staleness=0。
DDP 默认全部可见 GPU 一起训练一个客户端，完成后换下一个；物理执行顺序不改变共同
初始化。也可用现有 fixed client_parallel policy，每张卡训练独立客户端。两者沿用
共同 reference batch 语义，不能宣称随机数和浮点轨迹完全一致。

## OPCM 如何用于独立客户端专家

参考 [OPCM 原文与官方代码](https://github.com/tanganke/opcm)。原文独立微调专家共享
预训练起点；one-shot 数据流更接近该条件，但模型仍是 LLaVA LoRA，不是论文 CLIP。

本方法组合调用项目现有、未修改的 `OPCMLoRA`：

- 每个专家的功能增量是 `B_local A_local - B_initial A_initial`，而不是因中间服务器
  更新而重新定基的参数。默认初始 B=0，增量就是该专家的 LoRA 功能矩阵。
- 保留当前 OPCM 的奇异方向主块/对角过滤、跨模块范数控制和逐步合并机制。
- 不逐元素平均 A/B；通过共同紧凑 QR 坐标处理 BA 功能矩阵，再 SVD 压缩回固定 rank。
- 不下载额外模型，不在服务器访问训练数据，不使用验证/测试数据做参数合并。
- 与原文密集权重合并仍存在低秩适配：只过滤非零奇异方向，逐个合并后压回固定 LoRA
  rank，结果会有可记录的压缩误差。不能宣称完全复现原文所有数值或精度。

## 独立参数和合并顺序

修改 `configs/methods/opcm_oneshot.yaml`，不会改变 `configs/methods/opcm_lora.yaml`：

| 设置 | 默认 | 含义 |
|---|---:|---|
| `training.local_epochs` | 10 | 一次连续本地训练的 epoch 数 |
| `federation.rounds` | 1 | 必须为1；配置校验拒绝其他值 |
| `merge_weight` | 1.0 | 离线 OPCM 后不再附加异步松弛；是项目选择，不冒充官方参数 |
| `retain` | 0.5 | 沿用现有 OPCM 奇异值累计质量保护比例 |
| `eps` | 1e-8 | 沿用现有 OPCM 数值容差 |
| `merge_order` | seeded_random | 与设备快慢无关的确定性打乱顺序 |
| `merge_seed` | 42 | 专门控制合并顺序，与训练 seed 分别记录 |

可选 `merge_order: client_id` 表示按客户端ID排序；`arrival` 表示按虚拟到达时间排序，
相同时间按客户端ID打破平局。OPCM不是顺序无关的聚合，因此正式实验建议固定
seeded_random，并通过不同 merge_seed 做顺序敏感性对照，不能按照测试结果挑选顺序。

现有异步 `opcm_lora` 的 `merge_weight: 0.5` 原样保留。本独立方法默认1.0不意味着
原方法被修改；若要对照0.5，应只改 one-shot YAML并生成另一个不可变快照。

## 创建快照与运行

在服务器代码仓库根目录执行。默认自动识别全部可见训练 GPU：

```bash
python tools/generate_oneshot_profiles.py \
  --profile opcm_oneshot_e10_bs4_ga4_s42 \
  --local_epochs 10

bash scripts/run_oneshot.sh 2 opcm_oneshot_e10_bs4_ga4_s42
# 同一个profile还包含另外两档配置；按需运行，不是同时运行。
bash scripts/run_oneshot.sh 5 opcm_oneshot_e10_bs4_ga4_s42
bash scripts/run_oneshot.sh 10 opcm_oneshot_e10_bs4_ga4_s42
```

名称是人工标签，只有实际 YAML和命令参数决定训练设置。`bs4/ga4` 对应当前 base 中
的默认值，若改了 batch 请改标签避免混淆。只改本方法 epoch 可用 `--local_epochs 5`，
不必改变 base.yaml，从而不影响其他方法未来的实验模板。

如果想使用某次普通实验完全相同的客户端速度、网络延迟和基础计算成本，仅复用其
system profile，不复用原来多轮的 TrainPlan；将占位名称替换为真实旧快照名称：

```bash
python tools/generate_oneshot_profiles.py \
  --profile opcm_oneshot_e10_same_system \
  --local_epochs 10 \
  --system_profile_from YOUR_OLD_PROFILE
```

工具核对实际客户端ID、任务、数据源、样本数、任务成本因子和系统种子；不兼容时
拒绝复用。数据已经换版时不要拿旧数据画像来复用。未指定来源时，通过现有稳定种子
生成方法无关的系统属性。one-shot的一次长作业必然有新的训练时间，不能复用120次
短上传的原 Plan 或声称 wall-clock/虚拟训练时间与异步主实验相同。

纯离线生成配置时可显式加 `--reference_world_size 8`，用于冻结目标服务器的八卡
有效 batch；正常在服务器生成时不需要该参数，也不需要修改 base 的 GPU 字段。

目录独立于现有 `experiment_profiles`：

```text
one_shot_profiles/<name>/
  manifest.yaml
  configs/{2,5,10}clients/opcm_oneshot.yaml
  plans/{2,5,10}clients/system_profile.json
  plans/{2,5,10}clients/one_shot_train_plan.json

runs/llava/afvlm_cm/oneshot/profiles/<name>/<setting>clients/opcm_oneshot/seed42/
```

同名 profile 不覆盖；改参数、epoch或合并顺序后必须使用新名称。`run_oneshot.sh`
启动前只核对小型配置/Plan/manifest哈希，不扫描所有图片，也不执行默认全量数据检查。
单独工程验证快照：

```bash
python tools/generate_oneshot_profiles.py --validate opcm_oneshot_e10_bs4_ga4_s42
```

该profile不适用于 `run_baselines.sh`，也不是 `generate_system_profiles.py` 的共享
异步快照。三个源实验 YAML仍登记在普通配置目录以供静态检查/直接入口使用，正式
运行推荐独立快照及上述脚本，避免兼容目录中的旧system profile与现有数据不匹配。

## 日志与检查点

- `updates.jsonl`：每客户端恰好一行，记录样本数、实际 optimizer_steps、有效 batch
  信息和sample hash。上传参数严格检查共同起点；没有接收新版本再训练的情况。
- `events.jsonl`：前 N-1 行为等待收集，最后一行的 result_metadata 包含
  `one_shot`、`offline_opcm`、`merge_client_order`、`merge_steps`、`expert_training`。
  每个离线步骤保留原 OPCM 模块投影/过滤范数、尺度和 rank 压缩诊断。
- `metrics.json`：只评估一次最终全局模型。原周期计数无需改变，将 interval 设为
  N+1即可；配置验证拒绝 one-shot 下小于等于N的周期阈值。
- 原最终联邦检查点与保存开关沿用。中途方法 state_dict可表示待收集专家，但没有新增
  任意事件恢复 CLI。合并完成后释放专家模型缓存，最终checkpoint不包含所有独立专家，
  因此不能直接用它免训练重复测试另一种合并顺序。

## 公平性与资源限制

当异步基线是10轮、每轮1epoch时，one-shot10epoch可作为相近样本遍历预算的对照。
但客户端 optimizer在one-shot内连续维护10epoch，而原多轮训练每个作业重建optimizer，
下载模型、数据顺序、客户端间信息交换也不同；不保证轨迹/总尾部step完全等价。
通信预算必须单列：one-shot每客户端1次上传，主实验每客户端10次。学习率日程、
epoch数及模型/LoRA设置应在论文中明确披露，不能仅用同名seed声称完全公平。

待合并缓存只存LoRA，不存冻结7B模型，也不重复保存每个客户端的base/delta。
若有8.4M LoRA参数，60个FP16专家约0.94GiB，FP32约1.88GiB；DDP每个进程都会持有
自己的服务器副本，而非自动跨卡分摊缓存。OPCM矩阵运算沿用服务器设备上的FP32，
仍会增加合并算力与显存开销；没有进行真实GPU训练或CUDA显存验证。
