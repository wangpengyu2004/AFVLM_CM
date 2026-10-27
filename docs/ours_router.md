# ours_router：一次容量探测＋客户端私有评分器＋模块级异步融合

依据《详细讲解这篇论文的方法》对话在2026-10-27最后确认的设计实现。
这是新方法 `ours_router`，不替换旧 `ours` 或 Hard Concrete 版本 `ours_hc`。
最新设计不再每次学习连续门控容量：容量预先估计，私有router只负责“选哪些”。

## 统一配置生成与启动

参数源：

- `configs/base.yaml`：共有训练、系统与评估设置。
- `configs/models/llava15_7b_router_lora.yaml`：本方法候选容量，默认R32、alpha64。
- `configs/methods/ours_router.yaml`：探测、router和服务器融合系数。
- `configs/experiments/llava/afvlm_cm/{2,5,10}clients/ours_router.yaml`：三个继承模板。

在服务器项目环境、仓库根目录使用原生成器：

```bash
python tools/generate_system_profiles.py \
  --profile router_r32_e1_bs4_ga4_r10_s42
```

一次生成**所有普通方法、三档设置**的完整配置（当前22种方法×3=66份），以及三档
system profile、TrainPlan和manifest。独立one-shot协议继续由原来的one-shot生成器处理。
profile名称只是标识，实际训练参数来自当前配置源。默认自动识别可见GPU；CPU电脑上
仅离线生成配置时可以显式传 `--reference_world_size 8`，无需改base里的GPU参数。

若要与已有实验复用完全相同的系统条件和虚拟Plan，在数据和本地训练预算兼容时：

```bash
python tools/generate_system_profiles.py \
  --profile router_r32_reuse_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from hc_r32_e1_bs4_ga4_r10_s42
```

不会改旧profile。参数改变后应创建新名称，不编辑已经运行的冻结配置。

选择一个规模启动；启动器自动让全部可见GPU共同训练一个客户端：

```bash
bash scripts/run_one.sh ours_router 2 router_r32_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ours_router 5 router_r32_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ours_router 10 router_r32_e1_bs4_ga4_r10_s42
```

如使用复用Plan的profile，将上述最后一个参数换成对应名称。YAML中的
`backend: client_parallel`配合`executor_policy: capability`是统一入口，能力选择结果为
`client_ddp`。不要强制改为每卡一个客户端：此版私有router跨参与留在协调执行器中，
手工强制worker-pool会提前报错。八卡时B4/G4有效批量128；参考预算和原DDP训练函数不变。
也支持现有serial协调执行器，但正式八卡实验用上面的run_one入口。
serial模式要求整份模型在一张GPU上，不使用跨卡的device_map自动切分。

## 1．一次性估计每个客户端、每个模块的容量

正式Plan开始前，每个客户端从**相同初始全局LoRA**独立训练一次探测LoRA。
默认1个本地epoch，训练集、学习率、batch、梯度累积和主训练相同；完成后恢复初始
参数，不把探测权重聚合进服务器，不生成正式上传、不增加server_version。
通常PEFT初始化为A随机、B零，所以初始有效更新BA=0；不会把A也置零。

对每个模块估计有效更新 `kappa*B_probe*A_probe` 的谱能量，选满足
`sum(sigma[:r]^2)/sum(sigma^2) >= capacity_energy_threshold` 的最小整数r，再施加上下界。
默认阈值0.95、最小1、最大不超过服务器候选R，允许任意整数，不固定为2/4/8等档位。
若达到上限仍不满足阈值，或探测更新能量为零，会明确记录threshold_met/energy_valid，
零能量回退最小容量；不能把回退解释为已估计最优容量。

用 `QR(B)` 与 `QR(A.T)` 的小核心 `R_B*R_A.T` 求奇异值，不物化4096×4096的BA，
得到的非零奇异谱与完整BA一致。这里只用谱确定数量，**不旋转全局因子、不把奇异方向
当成全局分量编号**。不运行多个rank对照，不做压缩后额外前向检查。
各客户端所有模块的容量在此次正式运行中保持固定。

探测是实质性的额外算法开销，单独记录在`client_calibration.json`和
`capacity_calibration.jsonl`。虚拟Plan时间不包含它；论文需要额外报告探测optimizer
steps和wall time。steps按协同DDP任务计，不再乘GPU数。不会为其他baseline悄悄增加预算。

## 2．每次虚拟下载时选择分量，本次集合不变

客户端保留私有MLP，对每个候选分量输入：

```text
x_j = concat(a_j/max(||a_j||,eps), b_j/max(||b_j||,eps), ||a_j||, ||b_j||)
s_j = MLP_client(x_j)                 # 无sigmoid/softmax，允许负数
S = topk(s, calibrated_capacity)
```

同一客户端对输入/输出维度兼容的模块共享MLP；不同维度使用独立输入头。默认一层
hidden32的SiLU MLP和线性实数输出。它不输入任务ID、样本或梯度；梯度只用于制作标签。
未获得有效监督的冷router随机选择指定数量的分量，客户端独立、可复现。
晚加入的冷router也用随机选择，但全局LoRA已非零时不声称各种选择输出相同。

选择在**Plan的start事件**根据该时刻下载快照完成；即使物理DDP稍后才计算，也不拿
更新后的全局模型重新评分。索引保留原始编号、允许不连续和跨客户端重叠。
本次本地训练期间不换索引，分数不乘进LoRA。rank0广播控制索引，保证DDP各副本形状一致。

## 3．物理提取小因子，主训练与router旁路分开

本方法的模型实例把选中A行/B列变成真正的小Parameter，前向和主AdamW只处理这些因子，
不是仅把完整R的梯度乘掩码。沿用服务器原 `kappa=alpha_global/R_global`，不会因局部r变小
而改成alpha/r。DDP包装器在更换参数形状后重建，整个本地job内形状和optimizer保持固定。
本地job结束后恢复完整候选结构；公共LLaVA实现和其他方法没有安装这些实例扩展。
旧DDP包装器在重建前释放并回收，避免保留旧reducer钩子；清理行为可参照
[PyTorch 2.0.1 Reducer析构实现](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/csrc/distributed/c10d/reducer.cpp#L239)。

每个完整optimizer step：

1. 主模型正常任务前向、梯度累积、DDP同步。
2. 在主optimizer更新前，读取参数和梯度，保存detached特征与有符号标签
   `u_j = -sum(b_j * grad_b_j)`；两者来自同一份更新前参数。
3. 主optimizer正常更新A/B，不加入router损失。
4. 私有MLP用保存的特征和标签做Huber回归；其输入和标签均断开主模型图。

梯度中已包含LoRA缩放，不再次乘kappa。模块内标签尺度使用mean(abs(u))的EMA，首次
有效值直接初始化；默认beta0.9。全零标签没有排序信息，跳过该模块的监督，不把未选
分量标成零。私有MLP默认FP32 AdamW、lr0.001、weight_decay0、eps1e-8、Huber delta1、
梯度范数上限1。只有这个小网络发生额外反向，不增加VLM前向或VLM反向。
当前主训练没有GradScaler；标签读取的是累积并同步后的原尺度梯度。

router未注册为VLM子模块，不进入主模型DDP reducer或联邦可训练参数。各DDP副本用已
同步的参数/梯度更新自己的同配方私有副本；最终控制索引仍以rank0为准。
router初始化使用隔离随机流，辅助分支不消耗主模型dropout随机数。

模块敏感度仍是已有Module-Gate绝对响应均值，跳过首个optimizer step：
`h=0.5*(<grad_A,A>+<grad_B,B>)`，`S=mean(abs(h))`，默认
`C=0.25+0.75*S/(mean_modules(S)+eps)`。这里按整模块统计，和分量级有符号router标签分开。
没有Hard Concrete正则、连续门控学习、尾部切换或逐rank历史。

## 4．稀疏增量上传和模块级服务器融合

上传只有版本、原始索引、选中因子的base-relative增量、模块敏感度和数值诊断。
不上传router权重/optimizer、冻结骨干、完整下载参数或完整本地端点。
某些内部字段仍沿用既有indexed_delta_v1协议的`hc_*`名字，只是兼容的张量编码键；
本方法不构造或吸收Hard Concrete幅度。

```text
D = sum_m C_m * ||kappa*(B_now[:,S] A_now[S,:] - B_base[:,S] A_base[S,:])||_F^2
U = sum_m C_m * ||kappa*(B_local A_local - B_base[:,S] A_base[S,:])||_F^2
rho = exp(-gamma*sqrt(D/(U+eps)))             # gamma=1，整条上传共用
w = clip(((window_length/Q+1)/(count_q+1))**0.5, 0.5, 1.5)
N_m = rho*w*C_m
alpha_m = N_m/(N_m + history_lambda*Omega_m + eps)
A_global[S,:] += alpha_m*delta_A
B_global[:,S] += alpha_m*delta_B
```

距离在三个模型的**同一选中支持**上比较，保留分量间交叉项，使用小Gram矩阵计算。
未选位置不写回；全局因子不裁剪、不置零。每模块一个alpha，无0.5额外硬上限。
频率窗口默认最近18次成功接受上传，Q仍是配置的六任务总数，当前上传成功后才加入。

历史沿用当前模块级规则：未观测H=1，首次成功C替换，后续普通EMA beta0.9；不乘rho/w，
无零初始化偏差修正，无外部固定1。`Omega_base=sum_q H_q/Q`。
保留当前任务结束选层累加保护：所有计划上传成功后冻结该任务最后EMA，扣敏感度底座
只用于选择高于平均份额的Q/K/V/O层，额外需求按固定任务先验求和，boost3/time_scale18。
每次按年龄重算，不跨步重复累积。结束后的第一条上传聚合前age=0。
`history_age_enabled: false`可关闭这一额外项，基础历史仍保留。
拒绝上传不推进版本、任务历史、频率窗口或成功结束计数。

最后仍用完整服务器LoRA评估所有六任务，不使用router或客户端子模型。
原来的统一指标、评估阈值、固定数据划分和虚拟Plan顺序保持原样。

## 保存与导出

新profile下输出路径：
`runs/llava/afvlm_cm/profiles/PROFILE/2clients/ours_router/seed42/`。
已有events/updates/task_metrics、最终指标和全局检查点继续使用。

- `client_calibration.json`、`capacity_calibration.jsonl`：谱能量、容量、阈值是否满足、额外成本。
- `events.jsonl`的`router_diagnostics`：每模块所选数量/索引、C、D/U/rho/w、历史、Omega和alpha。
- 本地trajectory：逐optimizer-step任务loss、router loss、有效监督模块数、标签幅度及累计学习步数。
  task_loss沿用现有microbatch/进程平均口径，不是全语料token加权困惑度。
- `client_private/TASK/client_N/router_latest.pt`：模拟客户端私有router和optimizer的最新状态，
  只由rank0写入本地磁盘，不参与上传、聚合或全局模型检查点。

```bash
python tools/export_ours_router_diagnostics.py \
  runs/llava/afvlm_cm/profiles/router_r32_e1_bs4_ga4_r10_s42/2clients/ours_router/seed42
```

得到`router_diagnostics/{updates,modules,local_steps,capacities}.csv`。
communication中的张量字节和JSON体积是payload口径，不等同实际网络协议总字节。
导出目录存在时拒绝覆盖，可用`--output-directory 新目录`。逐步轨迹在本地训练结束并
处理上传时写入事件；不是每步实时检查点，失败前未上传部分不保证保留。

## 实现边界

这是一项研究设计：谱能量不保证最优容量，一阶删除标签不保证真实删除损失或未来训练收益。
只监督选中分量存在选择偏差；当前没有额外探索策略或额外删除前向。
完整候选LoRA仍需下载，容量探测有额外成本，候选R32与基线r8不是相同参数预算。
DDP协调执行器保留客户端私有状态；当前不支持任意worker迁移，也不提供整套训练中途恢复。
全局检查点本身不包含私有router，不能拿它单独恢复客户端经验。

只新增默认关闭的calibration能力和生命周期钩子，旧方法不会执行探测；统一注册表发现
新方法，普通生成器和run_one不用加方法分支。原配置/profile、数据、旧ours/ours_hc与
baseline算法未改写。新超参数是公开的工程起点，不称为已调优结果。
当前只进行非训练工程、张量方程、协议和配置检查，未加载7B或启动训练；真实八卡
DDP参数切换、显存、性能和最终精度还需要服务器正式运行验证。
