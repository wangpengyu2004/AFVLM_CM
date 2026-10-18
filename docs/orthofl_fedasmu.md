# OrthoFL、无服务器数据 FedASMU 与 OPCM 设置

更新（2026-10-18）：FedASMU、UniFed-LoRA 现支持方法内部 DDP 适配，Pilot 仍使用
客户端并行。下文私有目标和算法公式保留；新增反馈归约、rank0 请求决策、显存边界与
启动流程以 [DDP 适配说明](baseline_ddp.md) 为准。

2026-10-11。本次不改 `ours` 公式、DDP backward、分区或历史 profile。
代码开发与注册规范用于将新行为封装为 Method 生命周期 hooks；普通方法默认 no-op。
这里只报告非训练检查，不保证未经真实环境验证的 GPU 性能或论文准确率。

## OPCM：恢复已运行设置

`configs/methods/opcm_lora.yaml` 恢复 `merge_weight: 0.5`，
`retain: 0.5`、`eps: 1e-8`、constant staleness 不变。
原文 [OPCM](https://arxiv.org/abs/2501.09522) / [作者实现](https://github.com/tanganke/opcm/blob/f083162178e53d8e2d09b467978d58ee44286b83/fusion_bench/method/opcm/opcm.py)
使用投影与自适应范数缩放 `lambda_t`，没有本项目新增的固定 relaxation
`merge_weight`。其完整候选直接部署相当于本项目该 relaxation 取1；不是说论文写了
一个叫merge_weight的1。`retain=.5` 是奇异值子空间的保留阈值，与之不同。
保留 `.5` 作为公开适配参数可以，但论文要写 OPCM-LoRA adaptation，不应声称完全原版。
恢复此模板不改旧快照、数据、seed、batch或输出；也不是对所有旧代码的逐字节回滚，
因此不能承诺数值/指标逐位复现。旧结果若使用不同数据/预算，仍须披露。
OPCM奇异值截断恢复原先 `searchsorted(cumsum, retain*sum)+1` 的计算顺序，
避免比例化CDF在边界引入舍入差异；不撤销DOP/NUFILT独立的功能修正。

## OrthoFL

原文：[Orthogonal Calibration for Asynchronous Federated Learning](https://arxiv.org/html/2502.15940v1)，2025预印本。

- Eq.4：服务器 `(1-beta_t)*global + beta_t*uploaded_client`。
- `beta_t=.6*(paper_staleness)^(-.5)`，论文age1对应无延迟。
- Eqs.5--6：维护每个客户端独立模型、其本次本地训练起点，以及上次该客户端
  **聚合后**的 global 参考；global期间漂移不从下一轮最新下载重置。
- Eq.7：逐transformer层，将同层所有联邦可训练 A/B 坐标拼成向量，
  `d_perp = d_global - <d_global,d_client>/||d_client||² * d_client`。
- Eq.8：下次客户端训练从 `uploaded_client+d_perp` 开始。
  全局聚合用的是原上传模型，不用校准后的客户端模型。
- 零方向不除零，其正交补是整个空间；参考/私有状态和在途下载可保存。

适配仅投影LoRA坐标，而非冻结7B权重；保留共同LLaVA与优化器。默认能力策略采用
client_ddp，也可用固定策略的client_parallel；物理执行器不改变上述虚拟事件。
相同时间先聚合，再下一轮dispatch，防止客户端拿到上轮未校准的状态。
主评估始终是单一全局模型，公共任务指标不变。

## FedASMU：不建立服务器校准集

原文：[AAAI2024 FedASMU](https://ojs.aaai.org/index.php/AAAI/article/view/29297)，
[含完整公式/附录的arXiv](https://arxiv.org/html/2312.05770v1)。

### 服务端 Eqs.1--3

`age=server_version-base_version+1`，超过tau99则拒绝；不发反馈、不递增version。

```text
xi = lambda / (sqrt(max(1,server_version))*age**sigma) + iota
alpha = mu_alpha*xi/(1+mu_alpha*xi)
global_new = (1-alpha)*global + alpha*client
```

每个客户端保存自己的lambda/sigma/iota。不是静态衰减系数：先求混合系数的真实
目标超梯度，更新这些控制量，再重新计算alpha用于服务器聚合。
初始version0的平方根用max(1,t)防止除零，不改变版本计数定义。

无服务器数据方案：冻结当前server版本，发送**LoRA状态**到一个GPU worker，
worker读取当前到达客户端自己的训练分区，按作业seed抽取feedback_samples条记录。
在候选 `mixed=(1-alpha)*global+alpha*client` 上做真实CE前向与梯度，返回
`h=<grad L_client_train(mixed),client-global>` 和损失/计数，不返回文本或图像。
以 `h*dalpha/dcontrol` 学lambda/sigma/iota。仅接受对应当前server版本的反馈。

**重要区别**：这是到达客户端的私有训练目标代理，不是论文样本加权全局F的精确导数，
非IID/到达偏差仍存在。没有用validation/test，也没有偷偷集中客户端训练集。
该反馈是额外算法计算/通信，不是假训练、不计入本地optimizer_steps或上传数。

### 客户端 Eqs.7--9

请求时得到虚拟请求时刻固定的fresh global，只有其version大于base才混合：

```text
phi = gamma/sqrt(g) * (1-upsilon/sqrt(g-o+1))
beta = mu_beta*phi/(1+mu_beta*phi)
local = (1-beta)*local + beta*fresh_global
```

在本地最近真实mini-batch上测量混合前后CE与方向导数，学习gamma/upsilon，
reward=loss_before-loss_mixed。不污染.grad、不另做本地optimizer.step；
目标观测时一致关闭dropout，结束后恢复全部LoRA及训练模式，再继续原训练。
FedASMU单独使用FP32 LoRA master避免元梯度的FP16数值损失，冻结主干仍用共同dtype。
本地AdamW保留共同VLM设置；原文SGD是已披露适配，不声称SGD实验的逐位复现。

### 请求时机 Eqs.5--6

服务器LSTM+FC生成客户端第一次请求时隙；真实本地混合reward经REINFORCE更新策略，
之后每个客户端用Q表和epsilon-greedy在minus/stay/add动作之间选择，按Bellman目标学习。
持久保存LSTM、奖励baseline、Q表、客户端控制量和在途决策。

原文按epoch请求；本项目常用local_epochs1，因此改为最多request_bins8个标准化
optimizer时隙，严格落在步骤1到K-1之间，不改变K。只有一步时不请求。
LSTM改为单请求分类而非原二元序列；结构、特征与随机初始化公开，不加载外部策略权重。
服务端在dispatch时根据当时已知状态决定时隙，新增虚拟refresh事件；不偷看未来global。
request_time=start_time+(request_step/K)*estimated_train_time。
为避免全部worker等待未来fresh状态的死锁，此类作业暂存到虚拟refresh可用才提交GPU，
其原始下载状态/base_version/start/arrival不变。普通baseline的提交路径不变。

### 参数、日志与成本

全部可调参数在`configs/methods/fedasmu.yaml`，元学习率1e-4、RL lr1e-3参考论文常用量级。
mu、控制初值、LSTM16、bins8、feedback_samples4等为本项目公开默认，不是官方VLM最优值。
`refresh_fraction`只兼容旧字段，当前学习式控制器不使用固定比例；正式实验重建配置。

- `events.jsonl/result_metadata`：server controls前后、alpha、local beta与reward、请求决策、反馈结果。
- `algorithm_objective_feedback.jsonl`：当前到达的server版本、训练样本ID哈希、方向导数、
  两次objective forward/一次backward、零optimizer step、wall time、额外工作标记。
- 最终`train_plan.json`：实际选择的虚拟request时间/step/fresh version。
- 基础Plan的virtual_training_time不包含额外反馈成本；比总训练效率时必须另外计入反馈
  wall time/通信/计算，不能宣称额外计算免费。任务指标和accepted-update评估基准不变。

## Pilot、UniFed-LoRA是否可运行

已实现各自核心机制并提供正式配置：Pilot两阶段视觉/文本训练、task/private adapters、
token CT-MoA与路由损失、任务/文本聚合；UniFed真实A/B超网络、描述符、客户端输出头、
本地delta的VJP元更新。两者保留全局主评估并输出native personalized_metrics.json。

不是毫无差异的原版：共同LLaVA/固定任务、阶段预算、初始化、全局部署视图等见
[完整审查](baseline_audit.md)。老Pilot bottleneck和UniFed scalar conditioning快照不能使用。
特别是UniFed完整输出头可能超出32GB GPU，当前显存预估会拒绝不足的配置而不退化成
scalar gate。需要更大显存、显式hyper_device或调整公开结构参数；不能承诺默认参数在
所有客户端规模都能直接跑。两者均未在本电脑加载7B验证。

## 正式运行

服务器上新建配置快照；如果旧profile的数据、seed、本地预算及系统参数完全匹配：

```bash
python tools/generate_system_profiles.py \
  --profile baseline_v2_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from YOUR_EXISTING_PROFILE
```

将YOUR_EXISTING_PROFILE替换为真实profile名。若旧数据/batch/epochs不同，去掉reuse选项
重新生成Plan。新profile收录51份配置与新OrthoFL；不改任何旧profile或旧结果。

```bash
bash scripts/run_one.sh opcm_lora 2 baseline_v2_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh orthofl 2 baseline_v2_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh fedasmu 2 baseline_v2_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh pilot 2 baseline_v2_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh unifed_lora 2 baseline_v2_e1_bs4_ga4_r10_s42
```

2可换为5或10。默认使用全部可见GPU；OrthoFL、FedASMU、UniFed按能力策略DDP，Pilot
保留client_parallel。默认base B4/G4、参考8卡时有效128；单卡worker按同一参考窗口
累积32个micro-batches，既不强行B128造成OOM，也不改变既有八卡DDP过程。
启动仍只使用本地模型与图像，无自动权重下载。此次没有上传GitHub。
