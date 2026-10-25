# 独立方法 ours_hc：Hard Concrete 选择＋模块级异步融合

实现依据是你在《详细讲解这篇论文的方法》对话中最终确认的模块级设计，以及本对话的
最新选择：**每次本地训练都重新学习门控**，不保留上次的掩码或门控 optimizer。
原 `ours`、其结束保护机制、所有 baseline 和 rank 探索实验保持原样。

## 本地：一次训练内学习容量，最后固定门控

全局 LoRA 维持固定候选容量 R。每个语言侧 Q/K/V/O 模块的本地分支为：

```text
DeltaW_m = kappa_m B_m diag(z_m) A_m
theta_m = log_alpha_m                 # 本地 FP32 可学习分布参数
u ~ Uniform(0,1)
s = sigmoid((log(u)-log(1-u)+theta)/temperature)
z = clamp(s*(upper-lower)+lower, 0, 1)
p_active = sigmoid(theta-temperature*log(-lower/upper))
J = task_loss + l0_lambda * sum(p_active)    # 对所有模块/分量求和，不是均值
```

采样、解析开启概率与确定性门控分别对应
[Hard Concrete 原论文式10–13](https://arxiv.org/html/1712.01312#S2.SS2)。这只是本地门控工具的
来源，整个异步联邦方法是你的设计，不声称原论文已经包含它。
训练期间不物理删除候选行列，不做 SVD、换基、位置轮转或预设最终保留数量。

每个本地 job 都重置 theta 和私有 AdamW；LoRA 因子则从本次下载的全局版本开始。
沿用原 `local_epochs` 对应的总 optimizer steps，划分为：

1. 预热：默认第 1 个 optimizer step，全部门控为 1，学习 LoRA，不更新 theta。
2. 联合阶段：采样 Hard Concrete，联合学习 A/B 和 theta，同时加入 L0 容量代价。
3. 固定尾部：默认最后 `ceil(total_steps*0.2)` 步，固定最终门控，仅继续更新 A/B，
   统计模块敏感度。它是原预算的一部分，不增加额外训练步。

至少需要 3 个 optimizer steps，以便保留联合阶段和固定尾部。小预算下会限制预热长度；
不会偷偷增加 epoch 或改 Plan。例：本地15步，默认1步预热、11步联合、3步固定。

最终门控与索引：

```text
zhat = clamp(sigmoid(theta)*(upper-lower)+lower, 0, 1)  # 没有温度
S_m = {j: zhat_m[j] > 0}                              # 没有0.5阈值或top-k
Abar_m = A_local[S_m, :]
Bbar_m = B_local[:, S_m] * zhat_m[S_m]                 # 幅度吸收到B列
deltaA_m = Abar_m - A_download[S_m, :]
deltaB_m = Bbar_m - B_download[:, S_m]
```

每模块可以得到不同的保留分量数，索引可以不连续，并保留原始全局编号。
数量表示保留分量数，不声称等于有效权重矩阵的严格数学秩或最优容量。
短本地训练、初始容量、L0强度和学习率都会影响结果，不保证一定产生稀疏掩码。

## 固定阶段的模块敏感度

在完整梯度累积、DDP同步之后，原 A/B optimizer.step 之前，读取已有任务梯度：

```text
h_m = 0.5 * (sum(grad(A_m)*A_m) + sum(grad(B_m)*B_m))
S_raw_m = mean_fixed_steps(abs(h_m))
C_m = (1-eta) + eta*S_raw_m/(mean_modules(S_raw)+eps)
```

默认 eta=0.75，即0.25均匀底座。这里整个模块一个敏感度，不计算逐 rank 敏感度；
Hard Concrete 开启概率也不充当敏感度。绝对值在求平均之前取，避免方向抵消。
它使用固定尾部的已有 backward，不额外反向传播。尾部不含 L0 更新；即使联合阶段
存在 L0 项，该项也仅依赖 theta，不给 A/B 增加正则梯度。
同时记录 signed_gate_mean、gate_direction 等观察量，不参与融合系数。
如果全部模块响应为零或观测无效，沿用原统计器的均匀回退，并标记
`sensitivity_valid=false`；不将这种回退解释成成功识别了任务重要性。

## 上传与下载快照

worker 返回前将本地结果编码成原始索引、选中位置的 deltaA/deltaB、模块 C 与控制信息。
不传冻结7B/CLIP、不传可训练 theta、不传完整本地端点或下载因子。
服务器在下载发生时保留对应版本的不可变 LoRA 快照，到达后计算可靠性，再释放这个
在途引用。不会用实际 GPU 完成顺序改变虚拟 Plan 或客户端下载版本。

框架增加一个通用 `pack_local_update` 钩子，默认原样返回更新，已有方法走完全相同的
更新内容；只有 `ours_hc` 在此处压缩。它位于 worker 传输之前，覆盖 DDP、serial 和
client_parallel。原来的服务器 `prepare_upload` 仍只在父进程执行一次。

空模块传空分量；它不要求全局对应模块归零。整个上传没有有效分量/功能移动时拒绝，
不增加服务器版本、成功接受计数、历史 EMA 或频率窗口。

## 服务器：同一选中支持上的功能可靠性

对每个模块、同一组选中索引，分别构造下载、当前、本地三个功能：

```text
X_base_m  = kappa * B_base[:,S] A_base[S,:]
X_now_m   = kappa * B_now[:,S]  A_now[S,:]
X_local_m = kappa * Bbar Abar
D = sum_m C_m * ||X_now_m-X_base_m||_F^2
U = sum_m C_m * ||X_local_m-X_base_m||_F^2
relative = sqrt(D/(U+eps))
rho = exp(-gamma*relative)              # 默认gamma=1
```

按你引用对话中的公式求和，没有另外除以模块参数量。实现用秩维 Gram 矩阵计算内积，
不生成4096×4096完整 BA。不拿完整全局 LoRA 与裁剪本地端点直接比较。
一个上传使用一个 rho，不是每个 rank 或模块独立计算可靠性。

## 频率、历史与增量融合

服务器保留最近18个成功接受上传的任务标签，在加入当前任务之前计算：

```text
w = clip(((window_length/Q+1)/(task_count_in_window+1))**0.5, 0.5, 1.5)
Omega_m = sum_q q_q H[q,m]            # 固定任务权重，默认每任务1/Q
P_m = rho * w * C_m
alpha_m = P_m/(P_m + history_lambda*Omega_m + eps)

A_server[S,:] += alpha_m*deltaA
B_server[:,S] += alpha_m*deltaB
```

这里 Q 为配置客户端所包含的任务总数，不按上传次数给历史分配任务权重，也不根据
近期沉默推测任务结束。所有选中分量共用所属模块的 alpha。未选中位置逐元素不变。
不增加固定外部保护1，不裁剪 alpha 到0.5，不加入原 ours 的任务结束年龄保护。

计算 alpha 使用聚合之前的历史。成功聚合后，当前任务模块历史按下列规则更新：

```text
未观测任务 H[q,m] = 1
首次成功上传 H[q,m] = C_m
后续成功上传 H[q,m] = beta*H[q,m] + (1-beta)*C_m
```

默认 beta=0.9；这是首次观测初始化的普通 EMA，不做零初始化偏差修正。
不乘 rho 或频率权重再写历史。历史统计为模块级，即使某模块选中了多个分量，也不
把同一个 C 复制成“逐 rank 重要性”。当前有效上传更新模块历史时包含模块敏感度向量
全部模块；其中无选中分量模块的响应为零，C仍依归一化/底座规则定义。

训练进度的 loss 在联合阶段为含 L0 项的 J，预热和固定阶段为任务损失；不能将不同
阶段的 loss 变化全部解释成任务性能变化。诊断中的期望活跃数量是本步 gate optimizer
更新后的统计。拒绝上传的实际 module_alpha 记为0，另保留提议值用于核对。

## 配置生成和启动：与现有方法使用同一流程

配置文件：

- `configs/methods/ours_hc.yaml`：新方法全部系数。
- `configs/models/llava15_7b_hc_lora.yaml`：仅新方法候选 R，默认32、alpha64。
- `configs/experiments/llava/afvlm_cm/{2,5,10}clients/ours_hc.yaml`：三个静态模板。

候选 R 可以是任意正整数，不限8，也不是只有2/4/8三档。默认32只是可修改的实验
起点；实际每次本地训练的保留数量由门控学习。门控新增 FP32 参数和 optimizer 只在
客户端私有模型中，DDP 的八张卡协同训练一个客户端时会正常同步这些门控梯度。

默认 l0_lambda=1e-4、gate_lr=0.2、initial_active_probability=0.9、temperature=2/3、
lower=-0.1、upper=1.1、fixed_tail_fraction=0.2。L0项采用分量数之和，lambda 的尺度
会随候选R和模块数有关。这些是公开的**工程初始值**，不是你已验证的最优参数或
原论文联邦默认参数。调整时创建新的方法配置快照，不能改旧实验。

修改参数时，训练预算仍在 `configs/base.yaml`，新方法系数在
`configs/methods/ours_hc.yaml`，候选容量在 `configs/models/llava15_7b_hc_lora.yaml`。
然后使用原来的统一生成器，一次生成全部普通方法、2/5/10三档的配置和 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile hc_r32_e1_bs4_ga4_r10_s42

bash scripts/run_one.sh ours_hc 2 hc_r32_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ours_hc 5 hc_r32_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ours_hc 10 hc_r32_e1_bs4_ga4_r10_s42
```

以上三个启动命令分别运行三档实验，按需要选一条执行。profile 名称只是目录名，
实际参数来自配置内容。生成后修改参数源不会改变已冻结的配置；再次修改应使用新名称。

本次已在本地生成 `experiment_profiles/hc_r32_e1_bs4_ga4_r10_s42/`：

- `configs/{2,5,10}clients/ours_hc.yaml`：新方法三档完整配置。
- 同目录包含其他20个普通方法的完整配置，总计63份。
- `plans/{2,5,10}clients/system_profile.json` 与 `async_train_plan.json`：基于当前68k分区。
- `manifest.yaml`：参数、方法名单、八卡参考有效批量和配置/Plan哈希。

本地生成时使用 `--reference_world_size 8` 固定服务器八卡参考；在八卡服务器自行生成时，
工具默认自动识别可见GPU，不必传此参数。已生成的上述 profile 无需再生成；同名目标
存在时生成器拒绝覆盖，所有旧 profile 保留原状。参考有效批量为 `8*4*4=128`。

若希望与服务器上已有实验使用完全相同的系统画像和虚拟Plan，且数据、训练参数兼容，
可以生成另一个新名称并复用旧Plan：

```bash
python tools/generate_system_profiles.py \
  --profile hc_r32_reuse_v1 \
  --reuse_plans_from OLD_PROFILE
bash scripts/run_one.sh ours_hc 2 hc_r32_reuse_v1
```

将 `OLD_PROFILE` 替换成实际旧profile名称。数据量或本地预算改变时应生成新Plan，
不能复用不兼容的旧Plan；不要编辑旧配置来加入新方法。

启动脚本根据能力使用全部可见 GPU 的 DDP；八卡服务器无需手写 GPU 编号。
默认一轮本地epoch由原训练配置决定。新方法预热/联合/尾部都包含在这份预算里。
YAML 中的 `runtime.backend: client_parallel` 与 `executor_policy: capability` 是统一
入口，启动器根据 `ours_hc` 的能力自动选择 `client_ddp`，全部可见GPU共同训练一个客户端。
不需要另外使用新方法专属启动器，也不需要手工改成每卡一个客户端。

新方法加入普通 `generate_system_profiles.py` 的方法集合；不加入默认
`run_baselines.sh` 批次。其他方法继续生成各自原有的配置，包括原候选rank和算法系数；
只有 `ours_hc` 使用新的候选R配置。本次不修改旧方法算法或旧profile。

## 输出、观察与评估

输出目录类似：

```text
runs/llava/afvlm_cm/profiles/hc_r32_e1_bs4_ga4_r10_s42/2clients/ours_hc/seed42/
```

沿用原 events.jsonl、updates.jsonl、task_metrics.jsonl、metrics.json、system_stats.json
和最终联邦可训练状态检查点。`events.jsonl` 的 result_metadata 内新增 hc_diagnostics，
记录原始索引、每模块保留数、保留门控幅度均值/每模块期望活跃数量、阶段步数、
L0期望活跃数量轨迹、
模块原始/归一化敏感度与方向、D/U/rho、频率w、历史前后值、Omega和alpha。
不会保存每步完整7B状态；最终全局检查点不含私有门控。
逐分量门控幅度在 worker 内吸收进 B 后从上传元数据删除；不上传逐分量 theta、
门控幅度或开启概率向量。上面的模块统计只用于观察，不参与服务器历史或系数计算。

导出这些数值方便画图：

```bash
python tools/export_ours_hc_diagnostics.py \
  runs/llava/afvlm_cm/profiles/hc_r32_e1_bs4_ga4_r10_s42/2clients/ours_hc/seed42
```

生成 hc_diagnostics/updates.csv、modules.csv、local_steps.csv。目录已存在时拒绝覆盖；
重新导出可用 `--output-directory 新目录`。因子元素数和张量字节记录的是稀疏因子
payload，control_json_bytes另记控制/诊断信息的JSON体积，不等同网络协议的实际总字节。

评估沿用同一套六任务全局模型评估与成功纳入客户端更新计数间隔；始终测试完整全局
LoRA，不带任何客户端门控，不换数据集、指标或参考答案。每十个成功接受上传的间隔
由原配置控制，不按 gate optimizer 的步数触发验证。

## 隔离边界与验证限制

模型扩展通过本方法专属实例 hook 安装，不修改 PEFT 类或公共 LLaVA forward。
LoRA 的原 AdamW、采样、梯度累积、DDP包装器和总本地步数不变；为让私有门控的
L0损失与 DDP 兼容，仅新方法实例使用非重入 decoder checkpoint，保留重算时随机数。
原 ours、OPCM 和其他方法继续使用原来的检查点设置和训练函数。

公共运行边界只新增默认 identity 的 pack_local_update，方法注册表多一个独立方法；
现有方法的更新对象、调度、优化器和评估没有改变。旧方法的配置、数据和历史 profile
没有改写。新方法增加的门控优化和候选容量是算法开销，不宣称稠密门控训练会自动
获得物理推理/训练加速，也不把较大候选R与基线r8视为相同参数预算。

目前只进行了非训练的公式、稀疏协议、配置、序列化和历史训练代码检查，没有加载
7B或启动真实训练。CUDA/DDP、显存、实际稀疏程度和精度需要服务器正式实验验证。
