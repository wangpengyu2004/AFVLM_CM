# Baseline 独立复审：完整性、适配边界与运行风险

审查日期：2026-10-11。对象是当前本地工作目录，包括尚未提交的上一轮实现；不是仅检查 GitHub 上的旧版本。

## 结论

**目前不能保证全部 baseline 已实现原论文的全部功能，也不能保证所有默认配置在 8×V100 32GB 上正常训练。**

本次重新追踪了注册表、下载锚点、客户端优化、虚拟到达、聚合、附加个性化评估以及方法模板，并重新核对论文算法与可取得的作者代码。基础方法的主要优化机制没有发现新的缺失；但是 Pilot 存在实际的半精度数值风险，UniFed-LoRA 存在默认规模不可运行的显存问题。其他部分属于实质性的算法/实验协议适配，不能用“论文全部功能完全一致”概括。

以下区分三个问题：核心机制有没有实现；与原文的差异是什么；当前证据是否足以证明真实训练有效。公式和配置检查只能回答前两者的一部分，不能替代第三者。

本次是只读算法审查，没有修复算法、调整配置、改变 `ours`、重建数据或 profile、加载模型、启动训练、提交或上传。新增的是本报告与本地规划记录。

## 需要优先处理的运行问题

### P1：Pilot 新增视觉参数直接使用 FP16 AdamW，可能出现非有限参数

位置：`code/afl_vlm/methods/pilot/adapters.py:30`、`:36`；`code/afl_vlm/models/llava15.py:304`、`:582`。

默认 LLaVA checkpoint 使用 FP16。Pilot 的任务、私有、跨任务连接器复制该连接器，router 也显式沿用其 dtype。这些参数直接进入普通 AdamW，没有单独的 FP32 主参数/状态方案。将上传快照转成 FP32，并不能使实际优化器内的 FP16 参数和状态变成 FP32。

[PyTorch 2.0.1 AdamW 实现](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/optim/adamw.py)用 `zeros_like(parameter)` 初始化 moments；当前调用采用默认 epsilon=1e-8。只进行 FP16 张量算术，不执行训练，已确认：

```text
gradient ≈ 1e-4
second_moment = (1 - 0.999) * gradient² -> 0
float16(1e-8) -> 0
sqrt(second_moment) + epsilon -> 0
```

因此小梯度或零梯度坐标可能在 AdamW 除法中产生 Inf/NaN。现有 Pilot 单元检查使用 FP32 连接器，没有覆盖默认 FP16 路径。这里证明的是数值风险条件，不是宣称已经观察到某次真实 Pilot 训练失败。

建议在正式运行前，限定在 Pilot wrapper 内建立 FP32 可训练参数/优化状态和正确的输入、输出精度转换，并增加相应 dtype/finite 回归检查。不要为修复 Pilot 而改变其他方法的共同训练路径。也不要把所有语言 LoRA 都误判为这个问题：所检查的 PEFT Linear LoRA 初始化与这些复制出的视觉参数不是同一条 dtype 路径。

### P1：UniFed-LoRA 默认输出头不支持全部三档的 32GB GPU 运行承诺

位置：`code/afl_vlm/methods/unifed_lora/hypernetwork.py:69`、`:80`、`:125`。

默认 hidden_size=256，r=8，LLaVA-7B 的每种投影 A/B 输出合计 65,536 个元素；每个客户端有四个独立输出头。按当前实现预检采用的 FP32 参数、梯度、两个 Adam moments 计算：

| 设置 | 客户端数 | 完整输出头训练状态估算 |
|---|---:|---:|
| 2 clients/task | 12 | 约 12.05 GiB |
| 5 clients/task | 30 | 约 30.12 GiB |
| 10 clients/task | 60 | 约 60.23 GiB |

上述数字尚不包含超网络 trunk、生成的 LoRA 状态、计算图、LLaVA worker 和激活。`hyper_device: auto` 将超网络放在一个服务器设备上，不会因为机器有八张卡就自动分摊这些 heads。代码中的显存检查会拒绝部分配置；它不是解决内存问题的实现。

因此当前不能说 5/10 档在现有 32GB 卡上可直接跑；2 档也需检查与 worker 共用设备后的峰值。建议显式选择并实现论文轻量 specialization、分片/其他可解释的容量方案，或明确要求足够大的独立设备。减少 hidden_size 是改实验架构，CPU 是部署选择，都不能偷偷当成与默认原模型等价。

## 逐方法审查结果

下表的“核心已检查”表示代码及非训练检查未发现主要机制缺失，**不是**真实 VLM/GPU 运行通过。

| 方法 | 已检查的机制 | 与原文的差异 / 当前判定 |
|---|---|---|
| local | 客户端独立持久 LoRA；无服务器混合 | 参考基线，非聚合论文复现；没有发现新的核心缺失 |
| fedavg | 同步、样本数加权、仅共享可训练状态 | LoRA + 共同 AdamW 替代原 full-model SGD；核心已检查 |
| fedprox | 下载锚点上的可微 proximal loss；FedAvg 服务端 | 只约束联邦参数；核心已检查 |
| fedadam | 下载相对 delta、m/v、服务器自适应更新 | 模板选择 Keras 变体；paper/legacy 分开，不能混称同一递推；核心已检查 |
| fedasync | 陈旧客户端模型与当前全局模型插值；proximal；衰减函数 | 不是 `ours` 的下载相对增量。AdamW/proximal 模式和参数须披露；核心已检查 |
| fedbuff | 缓冲下载相对 delta；统一平均；服务器 LR；尾缓冲 | 原非 DP 聚合机制已检查。没有实现 SecAgg/DP 协议，不能宣称提供这些隐私功能 |
| fedcompass | 速度估计、动态本地工作量、活组、deadline、迟到缓冲 | 核心 scheduler 存在，不只是 staleness 权重；虚拟速度、有限预算 flush、衰减与作者配置存在适配 |
| orthofl | 私有客户端轨迹与原始全局聚合分离；逐层正交校准 | 同层 LoRA A/B 向量投影，不等于 dense 权重/BA 功能空间投影；核心已检查 |
| fedasmu | server/local 动态系数、真实目标导数、LSTM/Q 请求、新模型本地混合 | 到达客户端目标替代全局目标；冷启动控制器替代预训练策略；应称 no-server-data adaptation |
| masfl | 初始真实梯度、客户端控制变量、校正 SGD、服务器控制/momentum | 到达缓冲替代论文随机选择最新客户端缓存；固定可调超参数，不是完整原抽样协议 |
| adamasfl | 上述机制 + 全向量归一化 SGD 与实际 eta*K 增量归一 | 同上；没有实现论文 tuning-free 理论选参；不是未经修改的原方法 |
| pilot | 任务/私有连接器、CT-MoA、router 辅助损失、任务与 text 聚合 | 数值风险尚未处理；阶段预算和邻居权重变化；global 主视图不是原生个性化模型 |
| unifed_lora | 真正 A/B 超网络、可学习描述符、私有输出头、delta VJP | 内存问题；全参与 meta-batch 替代单客户端 iteration；global 视图与原生个性化不同 |
| opcm_lora | 奇异方向主块/对角过滤、全局范数缩放 | 功能增量、固定秩重分解、merge_weight=.5 是项目适配；保留用户之前设置，不应称密集原算法等价 |
| dop_lora | 左右双投影、奇异值加权损失、MGDA、EMA | compact-coordinate Adam 与 dense element-wise Adam 不等价；固定秩压缩改变容量 |
| nufilt_lora | 零空间过滤、低秩 residual、双投影拟合 | 原密集 128/8/64 秩变为受 LoRA 子空间限制的 8/8/8；明显容量适配 |

`ours` 注册存在，但不是本次 baseline 论文复现审查对象。

## 为什么部分实现不能叫“论文全部功能”

### FedASMU：两个自适应机制保留了，但目标与请求策略并不完全相同

[原文](https://arxiv.org/html/2312.05770v1)用全局目标学习服务器混合系数，并包含请求时机的学习策略。当前用本次到达客户端训练 mini-batch 的真实方向导数，没有读取服务器校准数据，也不是用 loss 差假冒梯度。它满足用户“不建服务器数据集”的约束，却可能带有当前任务/到达顺序偏差，不能认为与全局目标等价。请求策略也从预训练序列方案变成冷启动单请求时隙 + 后续 Q-learning。

这是有明确计算路径的适配，不是换名字的 FedAsync；但原文目标、策略初始化与原理论不能整套迁移。建议论文标注 `FedASMU (client-feedback adaptation)`。

### MasFL / AdaMasFL：动量与归一化公式存在，抽样及 tuning-free 未完整保留

[ICML 2025 原算法](https://raw.githubusercontent.com/mlresearch/v267/main/assets/yan25f/yan25f.pdf)的客户端缓存抽样与 Theorem 4.1 选参不是目前到达缓冲 + 显式 YAML 参数的同一协议。当前控制变量、SGD、归一化和 eta*K 处理已经实现；但速度相关的到达次序并不是均匀客户端抽样，local_epochs 也导致不同客户端 K 不相同。

因此不应套用原文无偏采样和 tuning-free 的完整保证。需要保留原 Plan 时，应以“Plan-arrival adaptation”披露该差异；若要比较原生采样协议，则需要单独设计实验，而不是无说明替换共同 Plan。

### Pilot / UniFed-LoRA：共同全局评估是有效的额外协议，但不是原生主结果

[Pilot 原文](https://arxiv.org/html/2501.13985v1)保留个性化跨任务适配；当前 global 主评估只使用任务连接器及全局语言 LoRA，不启用该客户端的 private/cross/router。附加的个性化评估才覆盖完整客户端结构。

[UniFed-LoRA 原文](https://openaccess.thecvf.com/content/CVPR2026W/FedVision/papers/Milasheuski_UniFed-LoRA_Exploiting_Semantic_Task_Correlation_for_Heterogeneous_Multimodal_Federated_Fine-Tuning_CVPRW_2026_paper.pdf)是个性化超网络。当前平均生成 A/B 的全局视图是新增部署协议；平均 factors 不等于平均有效 BA 矩阵。单 iteration 选一个客户端改成同轮全参与，也改变了超网络 Adam 更新频率。

配置的 one-hot 经过可学习线性 task embedding，不能仅因输入 one-hot 就说没有实现可学习任务描述符；但描述符实现也不证明六任务相关性已学好。两种方法的 `personalized_metrics.json` 应单独报告，不要把它们的 global score 说成原论文原生个性化精度。

### 持续融合：实现了主要投影机制，不是密集专家融合的精确等价变换

重新核对了 [OPCM 作者代码](https://github.com/tanganke/opcm/tree/f083162178e53d8e2d09b467978d58ee44286b83)、[DOP 作者代码](https://github.com/EnnengYang/DOP/tree/6ddf5c87d7b31c2e9fcaa301e4166c023b6fbbf1) 与 [NUFILT 作者代码](https://github.com/zihuanqiu/NUFILT/tree/4cd298fb7ea22bd4bcd343748fb4b8e3f9815324)。原共同起点独立专家被替换为下载相对 BA 增量，这是异步固定任务适配。

compact core 可精确表示参与的低秩矩阵，但截断回 r=8 有损；重分解 A/B 也改变后续因子优化轨迹。DOP 的 Adam 不对坐标旋转保持等价，NUFILT 容量下降。OPCM 当前 `.5` 保持之前实验设置；其额外放松与压缩也使原 lambda 范数递推不再严格等价。应统一标注为 `*-LoRA (AFVLM adaptation)`，有效性要由真实实验验证。

## 共同流程及公平性边界

- 基础 speed/network/availability 来自共同 system profile，没有按 aggregation method 随机重新生成；物理 EDF 不替换虚拟聚合顺序。
- 方法实际下载状态与普通 server snapshot 有区别时，已检查专用锚点处理：OrthoFL 的私有状态、UniFed 的生成状态、FedASMU 的新模型混合不能简单替换成旧 Plan 中的通用参数。
- 当前周期评估以成功纳入模型的客户端更新数为共同基准，不把 FedBuff 的一次 buffer aggregation 错当成一个客户端上传。批量聚合跨越评估阈值时，只能评估当时实际存在的模型，不虚构中间模型。
- 并行/分布式评估先汇总完整预测，再算任务指标，不把 GPU shard 的 CIDEr/VQA 等指标直接平均；本次未发现这条流程新的缺失。
- 公共有效批量对齐不代表所有算法的计算预算相同。Mas 使用原生 SGD、Compass 分配特殊 steps，Pilot 多 adapter，UniFed 超网络，还有服务器拟合/反馈工作，都必须单独披露。
- Mas 初始化、FedASMU 额外反馈、DOP/NUFILT 内层拟合并非普通 Plan 上传预算里的相同工作；不能仅凭 `virtual_training_time` 宣称公平的端到端加速，应另外测量并报告这些工作与实际总耗时。
- 原模板已修正而 immutable profile 未重建时，旧快照不会自动获得这些修正。不能依据当前源代码的审查结论，直接认定旧结果来自当前算法。

## 本次实际执行的非训练检查

| 检查 | 实际结果 | 不能证明的内容 |
|---|---|---|
| 现有公式、状态与配置回归 | 146 项通过 | 不是 LLaVA 训练、显存或论文精度验证 |
| Ruff | 通过 | 不证明算法理论等价 |
| 仓库配置检查 | 17 个注册名、51 份实验配置、1 个旧 profile | 不证明旧 profile 已更新 |
| 数据完整性 | 158 文件、94,471,289 bytes；hash 保持现有 68k 分区 | 没有重新划分或修改标注 |
| FP16 算术诊断 | epsilon/二阶矩下溢条件已确认 | 没有执行 Pilot 优化器或 VLM 训练 |
| UniFed 容量估算 | 默认 heads 状态约 12/30/60 GiB | 不是实际 GPU 峰值测量 |

本地 Torch 有“未安装 NumPy”警告；检查完成，没有借此安装依赖、加载 checkpoint 或启动训练。

在修复 Pilot 数值路径、解决 UniFed 的目标规模部署，并补足真实环境验证前，不建议使用“所有 baseline 已完整实现、保证适配有效”的表述。其余改动是否要做，应按论文目标区分必要 bug 修复与会改变历史结果的算法协议变更。
