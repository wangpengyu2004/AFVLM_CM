# Baseline 功能审查与修正（2026-10-11）

标准是保留核心机制并正确适配固定任务 AFVLM-CM + LLaVA LoRA，不要求逐行复制原框架。
此文区分论文机制、作者实现、当前实验参数及必要适配。静态/代数检查不等于完成了 GPU
训练或复现了论文精度；没有加载 7B、下载模型、启动训练或改写已有 profile/数据。

## 当前结论

| 方法 | 审查后实现 | 必须披露的区别 |
|---|---|---|
| local | 客户端独立持久 LoRA，无服务器聚合 | 本地模型按任务公共测试集评估 |
| fedavg | 同步、样本数加权共享可训练状态 | full-model SGD → LoRA AdamW |
| fedprox | 下载锚点上的可微 proximal loss + FedAvg | 同上，只约束联邦可训练参数 |
| fedadam | 同步 delta + 服务器 Adam，支持 paper/keras/legacy 变体 | 新配置采用 Keras 位置的 epsilon；客户端仍用共同 AdamW |
| fedasync | 陈旧模型插值、staleness 函数、客户端 proximal、可选 alpha 衰减 | 默认论文损失版 proximal + AdamW；另有作者 contraction 模式 |
| fedbuff | 到达缓冲、下载相对 delta、均值与服务器 LR、可选陈旧衰减、尾缓冲 | 无安全聚合/DP；sample_size 是可选适配，不是原算法默认 |
| fedcompass | 动态速度估计、工作分配、活组与 deadline、迟到缓冲、半异步聚合 | 虚拟时钟移植；修正作者递增陈旧权重及迟到更新覆盖问题 |
| fedasmu | 动态 server/local 系数元梯度、LSTM/Q 请求时机、新全局模型本地混合 | 私有客户端训练目标反馈替代全局 F；epoch 时隙→optimizer 时隙；额外成本单列 |
| orthofl | 独立客户端轨迹、全局原模型插值、逐层正交漂移校准、下一轮私有下载 | 在同层全部 LoRA A/B 坐标上投影；不涉及冻结主干 |
| masfl | 真实初始梯度控制、客户端 SGD 校正、服务端双层 momentum/control | LoRA；持久 Plan 到达缓冲替代论文随机抽取客户端缓存 |
| adamasfl | 上述机制 + 全向量归一化 SGD + 实际 eta*K 归一的本地增量 | 同上；当前提供显式可调参数，不宣称已实现 tuning-free 理论设置 |
| pilot | 分阶段私有/任务连接器、token CT-MoA、路由损失、task/text 聚合 | 原生个性化为附加结果；阶段预算与数值归一化见下文 |
| unifed_lora | 真正的 A/B 超网络、任务/模态/层/投影描述符、客户端输出头、delta VJP | 同构主干；全参与 meta-batch；共同初始化；全局部署视图 |
| opcm_lora | 奇异方向过滤、全局范数控制 | 下载相对功能增量、低秩部署、非零子空间适配 |
| dop_lora | 双投影损失、MGDA、EMA、服务器矩阵拟合 | 紧凑坐标 Adam、低秩部署；不是全矩阵 Adam 的等价实现 |
| nufilt_lora | null-space filter、低秩残差与双投影服务器拟合 | 非零紧凑子空间、秩受 LoRA 子空间限制 |

`ours` 的算法文件、参数模板及原有 DDP backward 不在本次 baseline 修正范围。
共同扩展点默认是 no-op；个性化附加评估只由 Pilot/UniFed-LoRA 返回非空状态。

## 来源与参数（作者版本固定，避免“官方默认”歧义）

没有一组在所有数据/模型上唯一正确的超参数。不能把 CIFAR 演示脚本的学习率直接当作
LLaVA 的最佳值。共同配置保留语言 LoRA r8/alpha16 和 base.yaml 的训练预算；特殊算法
必须保留原生优化器/架构例外。新方法模板可继承到三档共51份实验配置。

- FedAvg：[McMahan 等，AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html)。
- FedProx：[作者仓库](https://github.com/litian96/FedProx/tree/d2a4501f319f1594b732d88315c5ca1a72855f50)，
  `flearn/optimizer/pgd.py`。项目 `mu=0.01` 可调，proximal只作用于下载时联邦参数。
- FedAdam：[FedOpt 论文](https://arxiv.org/abs/2003.00295)及
  [作者实验](https://github.com/google-research/federated/tree/a5af8c4433c9ee2d2b1565f1bcc68c89e2000a6b/optimization)。
  当前 `variant=keras, server_lr=.01, beta1=.9, beta2=.999, epsilon=.001`。
  Keras 变体先乘 `sqrt(1-beta2^t)/(1-beta1^t)`，分母为 `sqrt(v)+epsilon`；
  paper 变体使用未修正 moments，并按 Algorithm 2 初始化 `v_-1=epsilon²`；
  Keras/legacy 从零初始化；legacy 保留旧 profile 的 corrected-moment 分母。
- FedAsync：[论文](https://arxiv.org/abs/1903.03934)和
  [作者 async_fl](https://github.com/congxie1108/async_fl/tree/912ecf33e0fb811ac8551c4f3cdab344cdedfb53)。
  CLI 默认 alpha=.9/rho=1e-5；CIFAR 演示 alpha=.8/rho=.01/client SGD lr=.1。
  当前模板 alpha=.8、constant、`proximal_mode=paper_loss, proximal_mu=.1`，共同本地 AdamW。
  paper_loss 数值保留 `mu/2 * ||theta-anchor||²`，但正则项不进入 autograd；
  在完整累积/DDP同步后由已有 gradient hook 解析加入 `mu*(theta-anchor)` 一次，
  再执行 AdamW。不是移除正则，也不是 weight decay；规避旧版 reentrant checkpoint
  对同一 LoRA 参数触发额外 ready hook。FedProx 和 contraction 路径不作此改写。
  alpha_decay=.5/milestone800 来源于作者演示，当前120次上传通常不会触发。
  `official_contraction` 在 backward 后、optimizer.step 前把参数拉向下载锚点，保留
  已算的梯度；作者 rho 是 `client_lr * paper_proximal_mu`，不是 loss 的 mu。
  模板 rho=2e-6 在默认 paper_loss 下不使用；若手动切换 contraction，必须检查它与
  当前学习率的对应关系，不能把 CIFAR 强度无解释照搬到 VLM。
  constant/polynomial/hinge、作者 floor-sigmoid 及可选 alpha 衰减均可配置。
  服务器仍是 `(1-alpha_t)*theta_server + alpha_t*theta_client`，不是 `ours` 的增量式。
- FedBuff：[AISTATS 2022 原文 Algorithm 1](https://proceedings.mlr.press/v151/nguyen22b.html)；
  [FLSim 作者组织实现参考](https://github.com/facebookresearch/FLSim/tree/7311f2db471f89f14a0c8c9ea1c9584171677f5a)。
  项目 `buffer_size=5, server_lr=1, weighting=uniform, staleness_weighting=false,
  flush_last_buffer=true`。论文推荐 K10，不要求固定为10；FLSim示例还使用其他服务器
  LR/momentum，不是所有任务通用默认。可选衰减作用在 delta 上再求平均，不能只缩放
  后又归一化权重，否则 buffer_size=1 时会取消衰减。尾缓冲显式聚合。
- FedCompass：[ICLR 2024 论文](https://arxiv.org/abs/2309.14675)；
  [APPFL 作者实现](https://github.com/APPFL/FedCompass/tree/af96db2204795b0b0addfdec219b2a541f4b1ab3)。
  `min_local_steps=20, max_local_steps=100, speed_momentum=.9, latest_time_factor=1.5,
  alpha=.9, gradient_based=true` 参考作者 config；项目 polynomial 指数 **-.5**，
  修正作者代码 +.5 导致越旧权重越大的行为。速度来自既定系统画像的虚拟完成观测，
  不使用实际 GPU 抖动；快客户端可接更多工作，组内完成/超时触发释放，不设全轮 barrier。
  聚合系数为 alpha*s(age)/N，N 是全部客户端数，不是到达组大小或样本数。
- FedASMU：[AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29297)。
  按最新要求实现不预留服务器校准集的适配：客户端在当前混合模型上返回真实目标方向
  导数；用于学习每个客户端的 lambda/sigma/iota，而不是用 loss 差伪装元梯度。
  gamma/upsilon 同样经真实导数学得；LSTM 冷启动与客户端 Q-learning 决定请求时隙。
  原文全局 F 被到达客户端目标代理替换，有任务/到达偏差；不能声称原理论直接适用。
  系数公式含 1/sqrt(t)，不是 sqrt(t)。全部系数可配置，默认元梯度 lr1e-4、RL lr1e-3
  参考论文附录常用量级；控制初值/控制器尺寸等为公开项目选择，非官方 VLM 最优参数。
  详见 [OrthoFL/FedASMU 适配与操作](orthofl_fedasmu.md)。
- OrthoFL：[Orthogonal Calibration for Asynchronous Federated Learning (2025)](https://arxiv.org/html/2502.15940v1)。
  Eq.4 的全局插值与 Eqs.5--8 的下一轮客户端校准分别实现；beta=.6、a=.5 参考原文。
  漂移锚点是上次该客户端聚合后的 global，不会被下一次 dispatch 的最新 global 覆盖。
- MasFL/AdaMasFL：[ICML 2025 Algorithms 1–4](https://proceedings.mlr.press/v267/yan25f.html)。
  未验证到作者官方仓库，因此只主张基于原文实现。模板 beta=.9，client_lr=2e-5，
  buffer_size=1；Mas server_lr=.01，Ada server_lr=1，normalize_epsilon=1e-12。
  这是可调 VLM 设置，不是论文 Theorem 4.1 自动选参的复现；该理论假设固定K，而
  epoch-controlled 异构样本数会得到不同 K，不能直接宣称原收敛/公平抽样保证。
  这两个方法使用 **无 momentum/weight-decay 的客户端 SGD**；可训练 LoRA 保持FP32，
  防止归一化小步在FP16中被舍入为零，冻结7B仍使用配置精度。
- Pilot：[AAAI 2025 论文 Eqs.3–12](https://arxiv.org/abs/2501.13985)。未验证到官方代码。
  lambda0=.1/lambda1=.1/lambda2=.01、text_neighbors=6 参考论文；语言 LoRA 使用共同r8
  而非论文r64。任务与私有视觉连接器均复制预训练 MLP，不是旧残差 bottleneck。
- UniFed-LoRA：[CVPRW 2026 原文](https://openaccess.thecvf.com/content/CVPR2026W/FedVision/html/Milasheuski_UniFed-LoRA_Exploiting_Semantic_Task_Correlation_for_Heterogeneous_Multimodal_Federated_Fine-Tuning_CVPRW_2026_paper.html)。
  未验证到官方代码。4层hidden256 trunk，task/modality/layer/module embeddings，客户端
  A/B 输出头及负本地 delta 的 VJP；hyper_lr=2e-5/r8参考论文，embedding_dim16为项目配置。
  共同 LoRA alpha16 而非论文32。当前task descriptor为配置的六任务one-hot；并不
  等于已经推断出六任务真实语义相关性。客户端编号仅选私有输出头，不充当任务描述符。
- OPCM：[作者仓库 f083162](https://github.com/tanganke/opcm/tree/f083162178e53d8e2d09b467978d58ee44286b83)。
  retain=.5；原作者 cutoff 把索引当计数，项目修正为覆盖所需奇异值质量的方向数量，
  retain=1正确保留全部非零方向，不过滤任意零奇异空间补全。
- DOP：[作者仓库 6ddf5c8](https://github.com/EnnengYang/DOP/tree/6ddf5c87d7b31c2e9fcaa301e4166c023b6fbbf1)。
  inner_steps200/lr1e-4/retain.99999/EMA beta.999/initial alpha.8、MGDA+EMA+uv；
  修正 cutoff 的0方向边界，零损失加epsilon防止NaN。提供author_slice仅作显式对照。
- NUFILT：[作者仓库 4cd298f](https://github.com/zihuanqiu/NUFILT/tree/4cd298fb7ea22bd4bcd343748fb4b8e3f9815324)。
  inner_steps50/lr.001/init_std.02；作者 filter/projection/adaptation ranks=128/8/64，
  当前紧凑LoRA版8/8/8（有效秩再由非零子空间限制），不能声称已保留原密集模型容量。
  DOP/NUFILT模板 merge_weight=1；按用户要求 OPCM 恢复之前的 merge_weight=.5。
  OPCM 的 .5 是本项目异步 relaxation，不是原文的自适应 lambda_t；陈旧衰减默认constant。
  它们仍是固定任务异步持续**融合**，不改变为客户端持续学习基准。

## 特殊协议的完整衔接

### MasFL/AdaMasFL

先在每个客户端真实训练数据上、固定共同初始参数计算平均原始梯度 ci0，
再置 c0=mean(ci0)、g0=c0。该计算只做梯度观测，不执行 optimizer.step；不是
smoke/dry-run，不使用测试数据、不制造上传。`algorithm_initialization.json` 独立记录
观测次数、batching、wall time、`extra_work_not_in_train_plan=true`。

训练 hook 在完整梯度累积后读取原始梯度，再执行控制变量递推。Ada使用全LoRA向量
范数归一，不是逐层归一；服务端用实际客户端学习率和 optimizer_steps 除本地位移，
不会使用 `max(1, eta*K)` 这种错误分母。客户端最新计算控制与服务器最后接受控制分开。
服务端uniform缓冲均值及1/N全局控制更新保持原文机制。

额外初始化成本没有伪装进普通 Plan 的客户端上传预算；比较总系统成本时必须另外
计入初始化，不能仅引用训练回放的virtual_training_time。当前也不宣称复制了原文
随机选择最新缓存的采样分布；本项目采用同一系统画像下的到达缓冲适配。

### FedCompass

`scheduling/compass.py` 移植计算能力感知分组逻辑。Plan编译聚合释放时刻，另存
communication_arrival_time/scheduling_group_id/aggregation_release_time/reason，
因此缓冲延迟与真正上传到达可区分。每次作业保留完整数据与下载版本，迟到更新使用
列表保存，避免作者字典覆盖；有限训练预算结束显式flush。模型更新由公共server调用，
物理EDF不会抢先聚合。min/max是实际optimizer-step边界，不是固定epoch；参考8卡
批量的每步成本相应换算，基础speed/network/availability不随方法随机重建。

### Pilot

Stage1只训练当前任务连接器、私有连接器与语言LoRA；Stage2冻结私有/其他任务连接器，
训练当前任务、cross-task adapters、token router与语言LoRA。Stage2不再加difference loss。
默认stage1_rounds1，不额外增加第一轮stage2 local epoch，这是统一训练预算的适配。
差异损失使用逐图片的特征空间 Frobenius 约束，默认对token特征作L2归一以避免原
裸特征尺度在FP16下溢出/溢出；`normalize_difference=false`可切换裸论文表达式。
初始化cross-task adapters于客户端自身任务连接器；不复用上一作业私有残留。

只有task adapter bank+语言LoRA作为共享状态。客户端私有connector、cross adapters、router
按client保存，不跨client平均。task聚合是任务内样本加权；邻近text聚合同时保留样本数与
inverse distance，再对组合权重作凸归一，避免原Eq12系数和不为1导致整体factor缩小。

### UniFed-LoRA

没有scalar gate、update-norm回归或服务器校准数据。服务器生成真实全部A/B，客户端
训练后以其**实际生成下载模型**为base计算delta，并对超网络计算负delta VJP。
当前采用同轮full-participation meta-batch，而非原文每iteration选一个client；VJP在
同一超网络参数版本累积后更新一次。超网络初始化偏移中心为共同PEFT state，确保初始
B=0；初始零输出头会使第一步主要更新输出头，此后共享trunk也获得梯度。

完整client-specific heads成本随客户端数增长，估算parameters+grad+Adam moments后
检查GPU显存，不能保证30/60 clients能在32GB上容纳。没有自动CPU退回、参数低秩化或
标量替代；需要用户明确设置hyper_device或提供足够显存。默认client_parallel避免把完整
服务器超网络复制到每个DDP rank；冻结LLaVA worker仍每卡一份，训练批量使用共同参考预算。
UniFed的生成A/B与本地LoRA都用FP32，防止下载时FP16舍入被当作客户端delta进入VJP；
冻结主干精度不变。显存估算只作提前拒绝检查，不保证与worker激活共用GPU时绝对不OOM。

## 评估、重新运行与兼容性

主评估仍只用统一六任务evaluator，在原有共同accepted-client-update里程碑进行，
指标不跨任务生硬平均。Pilot主模型使用shared task connector+global text LoRA；
UniFed主模型使用所有client生成A/B的平均全局部署视图。这两个global视图是新增部署
协议，不等于论文原生personalized score。

训练结束另写 `personalized_metrics.json`：每个client用它自己的模型评估所属任务的
完整同一held-out corpus，再按task宏平均client指标。并行路径仍先合并预测再算corpus
指标，不平均GPU分片分数。附加评估在主metrics写完后执行，不改主评估频率或上传计数。
附加生成成本须与训练时间区分。其他方法没有额外个性化评估任务。

需要新建实验profile以获取修正后的方法参数；已存在的YAML快照不自动改写。
若数据分区、local_epochs、B/G/seed及系统条件未变，可以复用原普通异步Plan：

```bash
python tools/generate_system_profiles.py \
  --profile baseline_audit_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from YOUR_EXISTING_PROFILE
bash scripts/run_one.sh fedasync 2 baseline_audit_e1_bs4_ga4_r10_s42
bash scripts/run_baselines.sh 2 baseline_audit_e1_bs4_ga4_r10_s42
```

`YOUR_EXISTING_PROFILE`必须替换为真实名称。CPU电脑生成可显式加`--reference_world_size 8`；
服务器可自动检测。分区或训练工作量变化时不能reuse，必须生成新Plan。FedCompass从同一
system_profile重新编译它的专用动态计划，不使用旧的静态分组近似。默认batch包含
无服务器数据的FedASMU适配和OrthoFL；三种持续融合仍单独用run_merging_baselines.sh。

旧Pilot `adapter_bottleneck` / UniFed `conditioning_scale` / Compass `group_window`
快照描述错误旧机制，启动会
明确要求重建profile。其旧结果、旧FedASMU启发式结果不能以完整原方法名报论文。
修正核心机制的FedAsync/FedBuff/Mas/Compass等也应重跑；单独调参不能修复旧算法结果。

## 验证边界

新增非训练检查覆盖样本加权、proximal梯度、Adam多变体递推、FedAsync模型插值与contraction、
FedBuff K1衰减/尾flush、Mas初始控制/SGD/归一化/eta*K、动态Compass元数据往返/作业完整性、
Pilot冻结与token辅助损失、UniFed真实factor/VJP/随机状态保护/状态恢复、SVD质量边界。
它们不运行VLM训练、不伪造checkpoint、不代替GPU/DDP/NCCL兼容性验证。

本地实际检查：146项非训练公式/状态回归通过；51份实验配置、注册表、继承关系、
已有不可变profile与本地数据完整性检查通过；Ruff、Python语法编译及Git空白检查通过。
本地Torch报告未安装NumPy的警告，未因此安装依赖或加载模型。

FedASMU现有无服务器数据功能实现，但并非原始全局目标oracle的完全复现。
各方法当前实现均为上述列明的**可审计适配**。
仍需真实环境验证显存、混合精度数值、caption Java依赖和原论文性能，不提供“完全正确、
保证论文指标”的未经训练证实承诺。
