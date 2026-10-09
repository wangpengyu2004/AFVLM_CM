# 持续模型融合基线：AFVLM-CM 固定任务异步 LoRA 适配

本次增加 `opcm_lora`、`dop_lora`、`nufilt_lora`。三者均在原有虚拟 Plan 的到达事件上执行服务器融合，不把客户端改成任务增量流；不读取服务器训练/验证/测试样本来拟合融合参数，不额外运行 LLaVA forward/backward。客户端仍采用共同 LLaVA、LoRA、优化器、local_epochs、batch、学习率和数据分区。

**论文中必须使用带 AFVLM/LoRA adaptation 的名称，不能宣称原论文设置的精确复现。** 代码放在独立的 `code/afl_vlm/methods/continual_merge/`，现有方法实现、训练编码、评估评分、Plan 调度器均未改动。

## 1 方法选择与来源

| 注册名称 | 工作 | 保留的核心 |
|---|---|---|
| opcm_lora | Merging Models on the Fly Without Retraining: A Sequential Approach to Scalable Continual Model Merging，NeurIPS 2025 | 旧任务奇异方向的主块及对角过滤；持续融合；全模型范数自适应缩放 |
| dop_lora | Continual Model Merging without Data: Dual Projections for Balancing Stability and Plasticity，NeurIPS 2025 | 左右双投影、奇异值加权稳定/可塑损失、MGDA 自适应折中、系数 EMA |
| nufilt_lora | Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity，ICLR 2026 | 右零空间过滤；投影感知低秩残差适配；过滤与适配共同融合 |

原论文与作者代码：

- [OPCM 论文](https://arxiv.org/abs/2501.09522)、[作者实现](https://github.com/tanganke/opcm)。算法依据 `fusion_bench/method/opcm/opcm.py`：奇异坐标主块/对角清零和全模型平均任务向量范数控制。
- [DOP 会议条目](https://proceedings.neurips.cc/paper_files/paper/2025/hash/37d9f19150fce07bced2a81fc87d47a6-Abstract-Conference.html)、[作者实现](https://github.com/EnnengYang/DOP)。依据论文 §4.2 式 (2)、(3) 与 `cal_loss_i`、MGDA 分支。
- [NUFILT 论文](https://openreview.net/pdf?id=HDIf3fYqPP)、[作者实现](https://github.com/zihuanqiu/NUFILT)。依据式 (8)、(15)–(17)、Algorithm 1，以及 `fusion_bench/models/filter_lora.py` 的 `solve_lora` / `merge_to_base`。

2026-10-10 核对的上游提交：OPCM `f083162178e53d8e2d09b467978d58ee44286b83`；DOP `6ddf5c87d7b31c2e9fcaa301e4166c023b6fbbf1`；NUFILT `4cd298fb7ea22bd4bcd343748fb4b8e3f9815324`。实现为依据方程重新编写，不引入 FusionBench 或下载其数据/模型。

用户提供资料的其他工作暂未注册：

- **NSC，CVPR 2026**：[论文与代码入口](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Label-Free_Cross-Task_LoRA_Merging_with_Null-Space_Compression_CVPR_2026_paper.html)。虽直接验证 LLaVA-1.5-7B，但优化需要无标签校准数据、特征前向及多专家。不能用参数间余弦或 SVD 代替其激活 null-space-ratio objective 再叫 NSC。当前不引入校准集或测试时拟合。
- **Merge before Forget / SLAO，ICLR 2026**：[论文](https://openreview.net/pdf?id=i1Rj7yU6eF)。同时改变新任务 LoRA 初始化、A/B 非对称融合和序列时间缩放；原来依赖前一任务专家，并非只替换一个聚合函数。本次不加入不完整的“只平均 B”版本。
- **FedFisher** 是 one-shot Fisher 融合，不是当前连续到达融合设置；**delay-compensated ASGD** 是异步优化而不是持续 LoRA 模型融合，因此不在这组三个新增基线内。

## 2 共同功能空间与异步适配

对某个模块，普通 LoRA 的有效更新是 `F = s B A`，`s = alpha/r`。当前模型所有模块同一 rank/scale，融合代数以未缩放 `BA` 表示；共同 scale 不改变子空间、相对范数及最终投影形式，服务器 surrogate 的绝对损失/步长不能声称与任意异构 scaling 的原论文等价。不支持 DoRA、rsLoRA、不同客户端 rank/scale 或 projector/bias 联邦训练。

当前服务器更新记为 `O = B_t A_t`，本次客户端的功能增量是：

```text
Delta = B_local A_local - B_download A_download
```

这里下载状态取 `Update.base_state` 的不可变快照；**不是**到达时服务器状态，也**不是** `(B_local-B_download)(A_local-A_download)`。后者会丢失交叉项。

普通 PEFT 初始化 B=0，所以共同功能起点是冻结主干 W0。启动时检查初始 B 全零、LoRA-only 完整 A/B 配对；预调好的非零初始适配器明确拒绝，不偷偷假设共同起点。

- OPCM 对 `Delta` 做连续投影并融合；原论文独立专家任务向量被替换为客户端本次功能增量。
- DOP / NUFILT 的新专家目标重基为 `N = O + Delta`。这是到达时的**代数目标**，不重新训练客户端、不替换下载快照或 base_version，也不将 global drift 当作客户端新知识。
- 每次到达完成一种候选融合 `F*` 后，用 `F_next = O + eta (F* - O)` 放松；默认 `eta=0.5`。配置 `merge_weight=1` 可以关闭放松，但原专家起点/rank 适配仍存在。
- 默认 `staleness.type=constant`，不会叠加一个人为 FedAsync 衰减。若做扩展消融，可用共同 staleness 函数令 `eta=merge_weight*s(staleness)`，必须明确额外衰减不是上述论文原组件。
- 默认每个有效到达返回一次 mutation，version/accepted update count 按共同服务器机制递增。即使零功能增量仍计该到达，状态保持不变；不会改变周期评估触发规则。

## 3 不构造密集 7B 权重

对 global/local/download 三组 A/B，分别对 `[B_t,B_local,B_download]` 和 `[A_t.T,A_local.T,A_download.T]` 做 thin QR，得到左右正交基 `Qo, Qi`：

```text
BA = Qo C Qi.T
C = (Qo.T B)(A Qi)
```

core 维度最大约 `3r × 3r`；rank=8 时通常不超过 `24 × 24`。投影、内层拟合和 SVD 在该共同坐标内进行，矩阵代数留在服务器状态所在设备（训练通常为 GPU）。不会把 frozen backbone 移到 CPU 平均、不会构造所有模块的 `4096×4096` 更新或 projector。日志标量转到 Python 时仍有少量 GPU 同步。

core 表示对这些低秩矩阵是精确的；**最后固定 rank 压缩不是精确的**。计算候选的最优 Frobenius rank-r 截断 SVD，再回写同名、同尺寸 A/B。A 保留单位奇异方向、B 承担奇异值，不把零奇异方向的 A/B 同时置零导致后续无法学习。全零候选保留原 A、B=0。

保护子空间使用数值 rank：相对奇异值阈值取 `max(eps, machine_epsilon × max(core.shape))`，避免 FP32 舍入误差被误认为旧模型的真实方向。DOP/NUFILT 的零增量模块保留原 A/B，不做无意义重分解；OPCM 的零增量模块仍按全局历史缩放参与递推，整次上传均为零增量时则保持全部参数不变。

重新分解改变了 A/B 参数坐标，虽截断前/无丢弃时函数可保持一致，但后续 AdamW 的参数轨迹不与原坐标完全相同。固定-rank 截断也可能破坏投影前的保护性质。这是必要且已披露的 PEFT 部署适配。

## 4 OPCM-LoRA 算法

对旧 O 的非零 thin SVD：`O=U Sigma V.T`。令 `Z=U.T Delta V`，按累计奇异值质量的 `retain=0.5` 确定受保护 rank。清除 Z 的该主块及全部非零奇异方向的对角，并从 Delta 中减去被清除的重叠部分。其余左右空间分量保留。

不能把它简化成 `Delta(I-VV.T)`：原 OPCM 是奇异坐标主块+对角过滤，不是把全部旧右子空间投影都删除。

原密集实现完整 SVD 还会清除零奇异值补空间的对角；低秩 LoRA 的补空间基不唯一。本实现仅清除**非零旧奇异方向**，避免任意补基造成 gauge-dependent 过滤；这项差异必须披露。

```text
raw = lambda_previous O + filtered_Delta
mean_norm = 历次客户端 Delta 全模块 Frobenius 范数的算术平均
lambda_new = ||raw||_F / max(mean_norm, eps)
F* = raw / max(lambda_new, eps)
F_next = rank_r(O + eta(F* - O))
```

范数跨全体模块联合计算，不单独归一化每层。`lambda_previous` 使用此前未放松、未固定-rank压缩候选的缩放值；放松/截断后它不再具有原论文密集精确递推的等价性。该基线明确保留投影和 norm control，但不是原论文定理的直接实例。

## 5 DOP-LoRA 算法

从 O 和 `N=O+Delta` 的 SVD 得到左右基与奇异值。对待融合 C 定义：

```text
Ls = ||diag(so) Uo.T(C-O)||_F² + ||(C-O) Vo diag(so)||_F²
Lp = ||diag(sn) Un.T(C-N)||_F² + ||(C-N) Vn diag(sn)||_F²
```

对应论文式 (2)，保留**左右两项和奇异值权重**。不能只实现一个右投影距离。

从 `C=(O+N)/2` 开始，求 Ls/Lp 的梯度，按作者实现可选用 loss normalization 求 MGDA 系数：

```text
a = clip(<gp, gp-gs> / (||gs-gp||²+数值保护), 0, 1)
alpha <- beta alpha + (1-beta) a
```

按 `alpha Ls+(1-alpha)Lp` 的原始损失梯度做 core Adam 更新。默认 `mgda_ema_beta=0.99`、初始 alpha=0.5；设 beta=0 可关闭平滑做消融。与作者代码一致，归一化用于系数求解，组合反向仍用原始损失。

默认服务器内层 50 步、lr=1e-4；作者实现默认内层预算200步，可在 YAML 改 `inner_steps: 200`。`inner_steps` **不是客户端 optimizer_steps/local_epochs**，不会改变客户端 TrainPlan。

这里 core 坐标内的投影损失与其对应密集矩阵一致，但 Adam 是逐坐标预条件，不在正交旋转下等变；core Adam 与原密集矩阵 Adam 的优化轨迹不相同，不能声称逐更新精确复现。

## 6 NUFILT-LoRA 算法

从旧 O 的非零右奇异方向构造 `P=I-VoVo.T`，从 N 取 Vn，先形成 `M=O+NP`。在 core 输入空间引入临时 residual gate `R=Bg Ag`，优化：

```text
F* = M + N Bg Ag
Ls = ||(F*-O) Vo||_F²
Lp = ||(F*-N) Vn||_F²
L = stability_weight Ls + plasticity_weight Lp
```

对应论文式 (15)–(17) 和作者 `FilterLoRA.solve_lora`。作者实现确认正确形式是 `O+N(P+BgAg)`，不是把尺寸不匹配的输入 residual 直接加到输出权重。

临时 Bg 初始为0、Ag 用确定性 seeded Gaussian 初始化（默认 std=.02），只对这两个服务器 core 张量做 surrogate Adam；不挂到 LLaVA、不加入客户端 optimizer、不参与通信或保存为部署模型的附加参数。随机种子依赖该 update 的 seed 与模块顺序，不消耗客户端全局 RNG。每个到达重新拟合，结果融合后丢弃临时张量。

当前默认 filter/projection/adaptation rank 均8，最大只保留非零奇异方向；作者密集设置 `rp=128, rl=64, rv=8`，rank参数不能直接照搬到仅rank8的部署适配器。默认内层50步、lr=1e-3，两个损失权重均1。当前 core 参数化及 rank-r 部署会限制原始 residual 的表达/优化轨迹。

## 7 配置与运行

方法配置：

```text
configs/methods/opcm_lora.yaml
configs/methods/dop_lora.yaml
configs/methods/nufilt_lora.yaml
```

各方法均有 `configs/experiments/llava/afvlm_cm/{2,5,10}clients/<method>.yaml`。继承共同模型/数据/base配置，不复制训练参数。注册器新名称自动进入新生成 profile，总计16方法、48实验配置。原 `scripts/run_baselines.sh` 的12方法序列保持不变，新持续融合组用独立批处理，`ours`仍独立。

### 首次运行：用当前真实分区建立一个新快照

先确认 `configs/base.yaml` 的训练参数和服务器实际分区是你要比较的版本，再执行：

```bash
python tools/generate_system_profiles.py --profile merging_e1_bs4_ga4_r10_s42

bash scripts/run_one.sh opcm_lora 2 merging_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh dop_lora 2 merging_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh nufilt_lora 2 merging_e1_bs4_ga4_r10_s42
```

profile名称仅是人工标签，**不会自动设置epochs/batch/rounds**；当前基础默认参数是e1/bs4/ga4/r10/s42，若更改，名称也应相应调整。执行器按共同 capability 策略选 DDP，默认使用全部可见GPU；不需要直接指定GPU编号。要显式保留客户端并行，可在base新快照前设置 `executor_policy: fixed`、`backend: client_parallel`。

批量运行指定档：

```bash
bash scripts/run_merging_baselines.sh 2 merging_e1_bs4_ga4_r10_s42
bash scripts/run_merging_baselines.sh 5 merging_e1_bs4_ga4_r10_s42
bash scripts/run_merging_baselines.sh 10 merging_e1_bs4_ga4_r10_s42
```

为公平比较，旧基线/ours也用这个**同一个新快照**运行；不要比较不同数据、不同文本上限、不同有效batch或不同Plan的结果：

```bash
bash scripts/run_one.sh fedasync 2 merging_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ours 2 merging_e1_bs4_ga4_r10_s42
```

### 是否必须重生成 Plan

新增纯服务器融合方法本身**不需要改变Plan内容**。但旧 immutable profile不包含新方法配置，不会被静默添加/改写。因此可以创建新profile复用旧Plan：

```bash
python tools/generate_system_profiles.py \
  --profile merging68k_compare_s42 \
  --reuse_plans_from YOUR_EXISTING_COMPATIBLE_PROFILE
```

把最后参数换成实际旧profile名称。该命令使用**当前配置模板**，不是复制全部旧训练配置；必须先让base/model/dataset设置与比较的旧实验一致。生成器检查复用profile的任务成本、Plan事件数、客户端/round标识及基于复用profile样本数算出的local optimizer steps；它并不重新核对当前标注内容、所有训练/模型字段或system seed是否一致。数据变更时不要依赖复用检查来自动发现所有差异，应重新生成；学习率等变更也须自行确认比较目标。旧13方法快照仍通过静态检查，可继续跑旧方法，旧目录保持原状。

不要直接用无profile的默认实验配置搭配不兼容的历史默认Plan。输出不覆盖原实验：`runs/llava/afvlm_cm/profiles/<profile>/<setting>clients/<method>/seed42/`，非空目录默认拒绝覆盖。

## 8 日志、评估和状态

复用原 `updates.jsonl` 的client/task/base_version/optimizer_steps、`events.jsonl` 的虚拟到达、server version/staleness 和 accepted count。每次 `events.jsonl.result_metadata` 额外含：

- `continual_merge`, `adaptation`, `merge_count`, `merge_weight`, `incoming_functional_norm`；OPCM另有 `opcm_scale`。
- 按模块 `compression_relative_error`, `retained_energy_ratio`, `functional_norm`。
- OPCM：protected_rank、过滤后norm、移除norm。
- DOP：protected/new rank、MGDA最终alpha、内层步数、stability/plasticity loss。
- NUFILT：protected/new/adaptation rank、内层步数、filtered/residual norm、两项loss。

压缩能量比并非任务性能；投影损失小也不保证下游指标提升。不要用这些内部数值替代完整六任务评估。

评估沿用共同 server_global 管道与 accepted/incorporated client update 预算，默认每10个纳入更新做验证，最后共同final；多GPU完整汇总预测后评分，不新增校准阶段。真实服务器代数耗时计入总体wall-clock，不改变虚拟训练时间，后者尚未模拟通信以外的服务器计算成本。

方法 `state_dict` 保存 params、到达计数、累计增量norm和OPCM scale，最终检查点由原流程写入。此接口不是新增中间事件恢复CLI。原有模型/客户端状态不会被这个方法的内层优化器引用或修改。

## 9 验证范围和限制

仅执行注册器、配置继承/公平字段、旧profile完整性、Python语法与非训练矩阵代数检查；无7B权重加载、无图像下载、无客户端训练、无fake LLaVA forward或微型训练流程。矩阵检查的CPU PyTorch仅在本地工程环境使用，服务器训练沿用已有requirements。

当前不保证正式多GPU训练已经跑通，更不保证新基线的论文性能。正式实验需要检查目标服务器的CUDA/PEFT/LLaVA版本、压缩误差、融合参数和内层预算。矩阵融合基线额外需要服务器QR/SVD/小矩阵拟合开销；这部分是算法成本，不应从wall-clock比较隐藏。本次方法发布不提交工作区内单独的68k数据构建/完整性摘要修改；本地数据版本取决于各机器部署，不能把拉取代码当作同步数据。三种方法不依赖固定64.5k/68k数量，生成profile读取实际分区。

固定LoRA rank、重新分解后的参数轨迹、异步重基专家、OPCM补空间及缩放差异、NUFILT core residual维度和DOP core Adam差异均需在论文中披露。不能沿用原论文的无遗忘/范数/收敛定理作为本适配的证明。

## 10 本次文件清单

新增：

```text
code/afl_vlm/methods/continual_merge/__init__.py
code/afl_vlm/methods/continual_merge/algebra.py
code/afl_vlm/methods/continual_merge/method.py
configs/methods/opcm_lora.yaml
configs/methods/dop_lora.yaml
configs/methods/nufilt_lora.yaml
configs/experiments/llava/afvlm_cm/2clients/opcm_lora.yaml
configs/experiments/llava/afvlm_cm/2clients/dop_lora.yaml
configs/experiments/llava/afvlm_cm/2clients/nufilt_lora.yaml
configs/experiments/llava/afvlm_cm/5clients/opcm_lora.yaml
configs/experiments/llava/afvlm_cm/5clients/dop_lora.yaml
configs/experiments/llava/afvlm_cm/5clients/nufilt_lora.yaml
configs/experiments/llava/afvlm_cm/10clients/opcm_lora.yaml
configs/experiments/llava/afvlm_cm/10clients/dop_lora.yaml
configs/experiments/llava/afvlm_cm/10clients/nufilt_lora.yaml
scripts/run_merging_baselines.sh
tests/test_continual_merging.py
docs/continual_merging.md
```

修改：`code/afl_vlm/methods/registry.py`（注册导入）、`tools/validate_repository.py`（新增配置/方法及旧快照兼容）、`README.md`、`docs/baselines.md`、`IMPLEMENTATION_REPORT.md`（运行/适配说明）。被Git忽略的task_plan/notes仅用于工作记录。

开始任务时已有68k数据修改涉及README、IMPLEMENTATION_REPORT、configs/datasets/afvlm_cm_integrity.json、docs/AFVLM_CM.md、tools/partition_afvlm_cm.py。本次仅在前两份说明中增加基线章节，其余既有数据改动保留不动，不把它们误报成本次重划分。
