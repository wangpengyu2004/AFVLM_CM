# ours_router：容量预估、私有 Router 与近期任务冲突选择

当前默认实现依据《详细讲解这篇论文的方法》对话在 **2026-11-02** 最后确认的设计：
**完整预估并预热 Router → 初始化共同全局 LoRA → 贡献评分减冲突惩罚选择
→ 紧凑 LoRA 训练 → 按陈旧度立即增量聚合 → 用实际变化更新近期方向。**

最新窗口为 **每任务最近 L=3 次成功聚合上传，并且记录年龄不超过 T=24 次全局成功聚合**。
不是每客户端三次，不是每分量三次，也不是全局最近18次。

此版本仍注册为 `ours_router`。旧 `ours`、`ours_hc` 和其他基线保留。
源码不自动迁移旧冻结配置：缺少模式参数时仍为 legacy；显式
`overlap_switch` 仍运行前一版集合开关/历史保护。新模式为 `conflict_router`。

## 配置与直接启动

共同训练参数在 `configs/base.yaml`；候选容量在
`configs/models/llava15_7b_router_lora.yaml`；本方法参数在
`configs/methods/ours_router.yaml`。默认候选 rank32、LoRA alpha64；
默认本地训练由 base 控制，当前 epoch1、每卡 batch2、梯度累积4、学习率2e-5、
文本上限2048。八卡有效批量64，与其他方法共同采用新预算；训练循环、采样、DDP和
多模态collator代码不变。单卡客户端使用batch2/累积32对齐64，不要求继续保持128。
旧bs4/累积4的profile仍保留原预算，必须新建profile/Plan；不要复用旧128预算的Plan。

在服务器项目根目录、已配置的 Python 环境中生成新的不可变 profile：

```bash
python tools/generate_system_profiles.py --profile all_e1_bs2_ga4_r10_s42
```

生成所有普通方法在2/5/10三档的完整配置，以及 system profile、TrainPlan、manifest。
profile 名称只是标识，不会代替 YAML 设置参数。生成前检查 base，勿仅修改名字。
离线无 GPU 的电脑生成八卡参考配置时，可额外传 `--reference_world_size 8`；
服务器默认自动识别可见 GPU，无需在 base 手填卡数。

数据、epoch、batch等预算未变时，可以复用既有 Plan：
在同一生成命令末尾加 `--reuse_plans_from 你的旧profile名称`。
新算法不要求更改虚拟系统条件；不得覆盖旧 profile，不能把旧模式配置直接当成新方法。
准备阶段的额外训练和初始化不属于正式 Plan。

选择一个规模运行即可，无需另一套生成器或 torchrun 命令：

```bash
bash scripts/run_one.sh ours_router 2 all_e1_bs2_ga4_r10_s42
bash scripts/run_one.sh ours_router 5 all_e1_bs2_ga4_r10_s42
bash scripts/run_one.sh ours_router 10 all_e1_bs2_ga4_r10_s42
```

启动器按 capability 自动选择 client_ddp，全部可见 GPU 共同训练一个客户端。
YAML 的 client_parallel + executor_policy: capability 是统一入口，不表示本方法实际
使用每卡一个客户端。私有 Router 需要协调执行器，不能强制改成 worker pool。
普通生成器、run_one、数据、评估协议及正式 Plan 的起始/到达顺序均保持原设计。

## 1．一次完整预估与共同初始化

每个客户端从同一初始 LoRA 独立训练完整候选 rank，默认一个本地 epoch，仅用训练集。
在每个完整 optimizer step，读取已累积且同步的、更新前的 A/B 和梯度：

```text
u_j = -sum(b_j * grad_b_j)
Feature_j = concat(a_j/max(norm(a_j),eps), b_j/max(norm(b_j),eps),
                   norm(a_j), norm(b_j))
```

特征和标签来自同一更新前状态，并完全 detach；主 LoRA 正常 AdamW 更新后，
私有 FP32 MLP 用独立优化器做有符号 Huber 回归。没有 v 标签，没有额外 VLM
前向/反向，Router 的损失和预测不进入主模型损失。当前主训练无 GradScaler；
已有梯度 hook 位于完整累积/DDP同步之后、主 optimizer 更新之前。
若以后引入 AMP scaler，必须保留“unscale 后读梯度”的约束。

Router 当前模板设置 router_sharing: layer：每个客户端、每个 Transformer 层拥有
独立评分网络，同一层 Q/K/V/O 共用该层网络；不同层即使维度一致也不共享权重。
保留原结构：输入 -> Linear(hidden32) -> SiLU -> Linear(1)，仅一个隐藏层，
对当前LLaVA投影即8194->32->1，没有增加隐藏层或改成64维。
评分和预估/正式监督均按真实 layers.N 模块名定位同一网络，不依赖客户端编号。
同层模块必须有相同输入/输出特征维度；不满足时明确报错，不静默拆成多个网络。
32层时每客户端32个网络、约839万参数；12客户端权重和Adam状态常驻估算约1.125GiB/卡，
不含临时内存。Router每次更新后立即清除已消费梯度，不再为所有闲置客户端保留梯度；
当前正在回归的一个客户端仍临时需要约32MiB梯度。Router权重和Adam状态仍在GPU，
不在每个optimizer step搬运Adam状态；真实峰值仍须服务器测量。
各层参数独立，沿用原客户端AdamW参数组、全局损失平均和梯度裁剪规则。

默认 hidden32/SiLU、学习率0.001、Huber delta1、weight_decay0、eps1e-8、
梯度范数上限1。模块内标签使用绝对幅度 EMA（beta0.9）作尺度，第一次有效值初始化；
它不是方向历史。全零标签跳过，不把未观测分量填成零标签。
如果整个预估没有任何有效 Router 监督，明确报错，不冒充已经预热的评分器。

旧冻结配置缺少 router_sharing 时仍使用原 dimension 共享，旧私有checkpoint仍可加载。
新的按层私有checkpoint包含router_layout，拒绝静默加载维度共享权重或不匹配的层映射。
服务器应生成新profile启用按层配置；本地不预生成。仅改变Router结构不要求新建系统
画像/Plan，预算兼容时仍可用 --reuse_plans_from 旧profile名称。旧实验结果不能直接
当作新结构结果，两种结构的评分/选择和正式性能可能不同。

预估结束，用有效更新 kappa*B*A 的奇异值平方能量估计容量：
达到95%累计能量的最小整数 rank，默认下限1、上限不超过候选 rank。
使用 QR 后 rank 规模的小核心做 SVD，不物化完整 BA，不重排或旋转 A/B；
容量不足/零能量会记录 energy_valid、threshold_met、bounded_by_maximum。
当前预估假设初始有效 LoRA 输出为零；代码显式拒绝非零 B 的初始状态，
防止把端点 BA 错当成相对更新。后续各模块容量固定，选择位置可变。

所有客户端预估后，形成正式训练的共同版本零起点：

```text
omega_i = (1 / number_of_tasks) * n_i / sum_{clients in task(i)} n
A_global_0 = sum_i omega_i * A_probe_i
B_global_0 = sum_i omega_i * B_probe_i
```

这是明确的工程选择：任务等权、同任务内部按样本数加权。
分别平均 A/B 不等价于平均 BA，不声称无损融合。
私有 Router、其 optimizer 与标签尺度保留；正式 LoRA optimizer 重新初始化。
初始化不计入正式上传、评估间隔或 server_version；正式方向窗口从空开始。
额外预估成本与初始化权重完整记录，不能忽略它们与无准备阶段基线的成本差异。

## 2．虚拟下载时评分，固定支持紧凑训练

客户端下载完整全局 A/B，仅为所有候选分量打分：

```text
s_j = private_router(Feature(A_global[j], B_global[:,j]))
s_normalized_j = s_j / (max(abs(s)) + eps)
z_j = s_normalized_j - conflict_penalty * c_task,j
S = TopK(z, calibrated_rank)
```

默认 conflict_penalty=0.3 是对话中的工程起点，不是调优结论。
只有本任务和至少另一个任务在该位置均有有效累计方向时才扣分；
未知位置暂不惩罚，也不放进平均分母，不设中性先验或强制探索。
即使客户端首次正式参与，若选择时已有有效冲突信息，也照常使用。
精确并列用独立固定种子打破，不消耗主模型/dropout随机流。

评分和冲突快照取自 **Plan 的虚拟下载事件**。物理训练稍后执行不会重新选择；
rank0 广播整个控制选择，DDP各副本索引/形状相同。
客户端提取选中的 A 行/B 列组成真正的小 Parameter，未选分量不参与本地前向。
始终保留全局 kappa=alpha_global/R_global，不按局部 rank 改缩放。
单次参与的索引固定，Router继续旁路学习，只影响下一次参与。
不申请空位、不重置少用分量，不强制独占、换位或永久封禁。
结构切换沿用既有方法私有的 DDP 解包/重建路径，公共训练器不改变。

## 3．按原功能性陈旧度统一立即聚合

上传：任务/客户端/版本、原始分量索引、相对下载快照的选中 A/B 增量、模块敏感度。
不上传骨干、Router权重/optimizer或完整端点。协议中的 hc_* 是兼容字段名，不是门控。

模块敏感度沿用现有 Module-Gate：跳过首个 optimizer step，累计
abs(0.5*(<grad_A,A>+<grad_B,B>)) 的均值，默认 C=0.25+0.75*S/mean(S)。
它仅参与已有的功能性陈旧度计算，不再生成不同模块的融合系数。

在 base/current/local 三个模型完全相同的本次选中支持上：

```text
D = sum_m C_m * ||kappa*(B_current[S] A_current[S] - B_base[S] A_base[S])||_F^2
U = sum_m C_m * ||kappa*(B_local A_local - B_base[S] A_base[S])||_F^2
rho = exp(-gamma * sqrt(D/(U+eps)))
alpha = server_step_size * rho
A_global[S,:] += alpha * delta_A
B_global[:,S] += alpha * delta_B
```

默认 gamma=1、server_step_size=1。同次上传所有模块/分量共用 alpha，无额外0.5上限。
未上传位置不写回。没有重叠开关、频率补偿、历史敏感度分母或任务结束年龄增强。
冲突不再次乘进当前聚合系数。极小能量、非有限值和重复身份仍校验。

## 4．精确近期功能方向：每任务 L=3，最大年龄 T=24

记录真实服务器本次聚合前后变化，而非原始客户端提议：

```text
E_u,m,j = kappa * (b_after a_after - b_before a_before)
```

代码使用等价、数值更稳定的 rank2 因子：

```text
L_u = [kappa*(b_after-b_before), kappa*b_before]
R_u = [a_after, a_after-a_before]^T
E_u = L_u @ R_u.T
<E_u,E_v> = sum((L_u.T @ L_v) * (R_u.T @ R_v))
```

仅缓存每次实际上传分量的因子及内积表 G，历史内积不重复计算；
新增记录只算与已有记录的内积，移除记录同步切除相应行列。
没有随机投影、完整权重矩阵、拼接 delta 的余弦或方向 EMA。
FP16输入提升为FP32因子，小Gram运算/累加用FP64，结果仍存在浮点误差。

全局每成功聚合一个正式上传 t 加1，记录本次 accepted_at：
同任务只保留最近3次成功上传，且 t-accepted_at<=24 才有效；
第24次仍有效、第25次失效。同任务不同客户端共用队列，
新记录只能刷新自己的时间，不能刷新旧记录或未上传位置。
失败、拒绝、重复上传、准备阶段均不进入此窗口。
任务结束后也不永久冻结旧方向，按同一有效期淘汰。

对每个模块、每个分量，把同任务有效记录相加成 M_q。
利用 Gram 分组求和计算 <M_q,M_k> 和范数：

```text
cosine_qk = <M_q,M_k> / (norm(M_q)*norm(M_k)+eps)
d_qk = max(0, -clip(cosine_qk,-1,1))
c_q = mean_{valid OTHER tasks k} d_qk
```

每任务一份累计方向，不按上传次数投票。
没有本任务方向、没有有效其他任务、累计变化抵消或近零时，都标记未知。
范数有效性同时考虑 conflict_min_norm=1e-12 与Gram抵消舍入容差。
这表示“近期功能方向兼容性代理”，不证明另一任务损失一定上升，也不保证频率公平。

候选全局参数和候选冲突窗口先计算并校验，既有 opt-in 事务回调在服务器模型/版本
成功应用后才发布记录。失败不推进方向时钟。保护其他方法的默认回调仍为空。
精确历史因子无损存于CPU内存，dtype和数值不变，小Gram表仍在服务器GPU。
新增聚合时按模块搬回历史因子，再在GPU做索引、FP64内积和累加；不把矩阵计算
改到CPU，不降低缓存精度，也不修改L/T窗口。局部反向传播不再同时占用整份方向因子
显存。缓存checkpoint字段不变，旧缓存可读取，恢复时也按上述设备布局放置。
协调DDP每个进程仍维护相同缓存，CPU内存消耗须乘进程数，并增加每次聚合的传输开销。
因子存储量仍随任务数*L*选择rank*(输入维度+输出维度)增长；factor_bytes是CPU存储字节，
gram_bytes是小Gram表字节，另记录factor_storage_device和gram_storage_device。
这项显存修正不要求重新生成配置或Plan，不会修改批量、梯度累积、优化步数或聚合公式。

本次共同默认已改为batch2/累积4。有效批量由128变64，optimizer steps也会改变，
因此新旧实验不是同一训练预算；所有对比方法应使用同一新profile重跑。
旧结果及配置保留。不要降低文本上限来丢弃多轮监督。

## 日志、导出与实现边界

正常 events/updates/task_metrics、最终指标和可选全局检查点继续使用。
新增/扩展：

- client_calibration.json / capacity_calibration.jsonl：容量、真实准备成本、初始化权重、
  每步任务/Router loss、有效监督数。
- client_private/TASK/client_N/router_latest.pt：客户端私有Router/optimizer，rank0本地保存；
  不属于联邦上传。预估结束时也保存，后续每次成功本地训练更新。
- router_diagnostics：C、D/U/rho、统一alpha、上传索引、下载时所有候选的原评分/归一化评分/
  冲突/有效比较数/最终分数、选择改变数和有效比较覆盖率、各任务窗口时间/年龄/内存。
- 方法checkpoint schema4：精确方向因子、小Gram、记录时钟、Plan、在途选择等；
  跨 legacy/overlap/conflict 模式不能静默迁移。全局检查点不包含全部客户端私有经验，
  本修改没有增加中途训练resume入口。

导出无需模型或图片：

```bash
python tools/export_ours_router_diagnostics.py runs/llava/afvlm_cm/profiles/all_e1_bs2_ga4_r10_s42/2clients/ours_router/seed42
```

输出 router_diagnostics 下的 updates、modules、local_steps、capacities、components、
resolutions、conflict_selection、conflict_windows、probe_steps CSV。
新增 router_ranking.csv：逐客户端、任务、optimizer step、模块记录 Spearman 排序相关性、
Top-k重合率和Top-1重合率。local_steps.csv 与 probe_steps.csv 同时包含有效模块均值和
有效模块数（router_spearman_mean、router_topk_overlap_mean、router_top1_overlap_mean）。
其中router_head标识实际使用的层网络；updates.csv另记录router_sharing与router_head_count。

诊断复用现有 Router 更新前预测与同一 LoRA 更新前状态的有符号 u 标签，不增加前向、
反向、损失项或DDP通信。只由主进程计算；小统计在每个完整step一次性传回CPU。
Spearman 使用并列平均秩；Top-k = 两个前k集合交集大小/k，默认k=ceil(观察数*25%)，
观察数>=2时保证k小于观察数，另记录Top-1。每行记录n、k、全局索引、候选rank和
独立随机排序的参考重合率k/n（Top-1为1/n），避免把不同池大小的数值直接等同。
常量预测/标签、仅一个分量、无有效监督时结果为null；Top-k截断边界存在并列时也
标记该项无效，不把任意并列打破当成真实排序能力。均值只统计对应指标有效的模块。

预估阶段scope=full_candidate_components，覆盖完整候选池；正式阶段
scope=selected_components_only，仅比较当前已选分量内部排序。正式阶段未选分量没有
标签，不能声称该指标验证了全部候选中的最终Top-k选择。诊断不包含冲突惩罚。
这是同一训练梯度代理上的在线拟合诊断，不是独立验证集的排序泛化或真实性能证明。
旧日志没有这些观测时导出留空，不补造数据；旧配置无需为本次日志改动重新生成Plan。
新模式旧历史列留空，不能把它们当作零历史保护；conflict_selection覆盖所有候选而非
仅上传位置。拒绝时alpha为0。保留导出拒绝覆盖行为，可用 --output-directory 新目录。
日志是上传结束后写入，不是逐步恢复检查点。

最终仍评估完整全局 LoRA 在六个任务的表现，不启用私有Router或任务子模型。
公共训练/DDP、生成指标、数据、原Plan与其他方法配置不变。
额外容量预估、Router回归、方向缓存的内存/显存与传输成本、准备聚合须单独披露；
候选R32也不能宣称与r8基线有相同参数预算。
已提供非训练方程、事务、有效期、配置和兼容性检查；
真实8卡NCCL、峰值显存及最终性能必须在服务器正式实验验证。

## 历史配置兼容

- 缺少 rank_aggregation_mode 或显式 legacy：7c5638b 的预估后恢复初始模型、冷Router、
  原模块级敏感度/频率/历史/结束年龄聚合。
- 显式 overlap_switch：保留此前每客户端最近选择并集、重叠位置使用历史保护分式、
  非重叠位置乘rho、任务结束冻结支持和历史的实现。
- 当前模板 conflict_router：本文新设计。使用新profile，不编辑旧结果对应的冻结配置。
