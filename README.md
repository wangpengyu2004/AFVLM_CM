# AFVLM-CM

AFVLM-CM 是用于论文实验的异步联邦视觉语言模型指令微调框架，研究固定任务客户端下的任务异构与系统异构：每个客户端永久属于一个任务，客户端速度、可用时间和网络延迟不同，但不存在持续学习或任务增量流。

正式实验只使用 LLaVA-v1.5-7B、CLIP ViT-L/14-336 和 LoRA。默认 LoRA 为 r=8、alpha=16、dropout=0.05、bias=none；默认随机种子为 42。一个进程中只保留一份冻结的 7B 主干；逻辑客户端通过装载各自训练起点的 LoRA/显式允许模块状态顺序训练。冻结主干不会上传、聚合或写入联邦检查点。

## 已检测的数据

框架直接读取现有的 `data/AFVLM_CM`，不生成、不重分区、不改写任何指令文件。

```text
data/AFVLM_CM/
├── dataset/                         # 所有图片的唯一根目录
└── partitioned/
    ├── 2_clients/                   # 6 × 2 = 12 clients
    ├── 5_clients/                   # 6 × 5 = 30 clients
    ├── 10_clients/                  # 6 × 10 = 60 clients
    └── metadata/
```

每个规模下均检测到六个任务目录：

| task ID | 来源数据 | 指标 | 图片相对目录 |
|---|---|---|---|
| `cls` | ImageNet-R | accuracy | `ImageNet-R/train/` |
| `caption` | Flickr30k | CIDEr、ROUGE-L | `Flickr30k/train/` |
| `vqa` | AOKVQA | VQA accuracy | `COCO2014/train2014/`、`val2014/` |
| `chart_vqa` | DVQA | answer accuracy | `DVQA/images/` |
| `visual_reasoning` | FigureQA | answer accuracy | `FigureQA/images/` |
| `grounding` | COCO grounding | mean IoU、IoU@0.5 accuracy | 与 AOKVQA 共用 `COCO2014/` |

每个任务目录包含原有的 `client_N.json`、`statistics.json`、`val.json` 和 `test.json`。启动时会再次扫描实际文件数；配置中的 12/30/60 不是客户端清单的替代品。`configs/datasets/afvlm_cm_integrity.json` 记录只读分区的静态完整性摘要。

三个数据配置为：

- `configs/datasets/afvlm_cm_2clients.yaml`
- `configs/datasets/afvlm_cm_5clients.yaml`
- `configs/datasets/afvlm_cm_10clients.yaml`

## 安装

正式实验环境面向 Linux x86_64 与 NVIDIA GPU。先创建一个干净的 Python 3.10 环境，然后只使用仓库根目录的 `requirements.txt` 安装：

```bash
conda create -n afvlm-cm python=3.10 pip -y
conda activate afvlm-cm
python -m pip install -r requirements.txt
```

不使用 Conda 时，也可以用 Python 3.10 的 `venv`，安装命令仍然相同：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 已包含 CUDA 11.8 对应的 PyTorch 2.0.1/TorchVision 0.15.2、LLaVA、Transformers、PEFT、数据处理、绘图和工程检查依赖，并以 editable 模式安装当前 AFVLM-CM 包。LLaVA 源码固定到官方 `v1.1.3`，通过体积固定的 release archive 安装而不是执行 `git clone`，可避免训练服务器连接 GitHub git 服务超时。环境安装不会下载 7B/CLIP 预训练参数。请在仓库根目录运行命令，并确保宿主机 NVIDIA 驱动兼容 CUDA 11.8。NF4 量化需要 bitsandbytes；默认配置仍使用非量化 bf16。正式训练只读取本地模型权重。

安装后可检查关键版本和 CUDA 可用性：

```bash
python -c "import torch, transformers, peft, llava; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__, peft.__version__)"
```

## 下载并放置模型

期望的本地路径固定为：

```text
pretrained/
├── llava-v1.5-7b/
└── clip-vit-large-patch14-336/
```

使用 Hugging Face 镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python tools/download_models.py
```

或显式参数（显式参数优先于环境变量）：

```bash
python tools/download_models.py --hf_endpoint https://hf-mirror.com
```

可使用 `--llava_only` 或 `--clip_only` 单独下载，`--resume` 续传，`--skip_existing` 跳过完整目录。下载先进入 `.partial` 目录，只有存在 `config.json` 后才原子移动为正式目录；失败不会伪造完整权重目录。

## 下载并放置图片

```bash
HF_ENDPOINT=https://hf-mirror.com \
python tools/download_afvlm_cm_images.py --all --resume --skip_existing
```

该工具支持 `--imagenet_r`、`--flickr30k`、`--dvqa`、`--figureqa`、`--coco2014` 与 `--output_root`，并在结束时逐条核对指令中的相对路径。ImageNet-R/Flickr30k 先读取 `HaiyangGuo/UCIT` 的真实文件树，再用 `snapshot_download + allow_patterns` 仅取所需归档；DVQA/FigureQA 通过 `list_repo_files` 解析 `MLLM-CL/FCIT` 的实际 `dataset/` 文件，不硬编码压缩包名。仓库结构不唯一或变化时明确报错，不会静默下载整个仓库。AOKVQA 与 Grounding 只保留一份 COCO2014。HF 镜像仅用于 Hugging Face 资源，COCO 始终访问官方 `images.cocodataset.org`，支持 HTTP retry、timeout 与 Range 续传。

只检查当前覆盖率而不下载：

```bash
python tools/download_afvlm_cm_images.py --validate_only
```

当前本地审计检测到 64,934 个唯一图片引用，其中 61,304 个尚未就位；因此正式训练会在加载 7B 模型前停止并提示缺失路径。

工具不会修改 `data/AFVLM_CM/partitioned` 来迎合猜测的目录。

## 配置结构

配置采用递归 `inherits` 深合并，并拒绝重复 YAML 键与继承环：

```text
configs/
├── base.yaml
├── models/llava15_7b_lora.yaml
├── datasets/afvlm_cm_{2,5,10}clients.yaml
├── methods/{local,...,unifed_lora,ours}.yaml
└── experiments/llava/afvlm_cm/
    ├── 2clients/    # 13 个可直接运行配置
    ├── 5clients/    # 13 个可直接运行配置
    └── 10clients/   # 13 个可直接运行配置
```

同一规模的所有方法继承相同模型、LoRA、客户端优化器、学习率、本地 epoch、batch size、随机种子、数据与评估配置。`plans/afvlm_cm/{2,5,10}clients/system_profile.json` 是固定、方法无关的速度/网络/可用性画像；`async_train_plan.json` 是普通异步方法共同重放的基础作业计划。除 FedCompass 的本地工作量分配外，异步算法不得改变这些基础条件。

普通方法的本地训练量只由 `training.local_epochs` 控制，不再同时设置通用的 `base_local_steps`。TrainPlan 中记录的 `local_steps` 是根据客户端样本数、batch size、梯度累积和 `local_epochs` 推导出的预计优化器步数，只用于虚拟耗时和算法钩子，不会截断普通客户端训练。FedCompass 是唯一例外：其计算能力感知调度器必须动态分配本地迭代，因此使用 `min_local_steps` / `max_local_steps` 作为方法专属边界。

修改 `local_epochs`、batch size 或梯度累积后，需要重新生成与训练量匹配的共享计划：

```bash
python tools/generate_system_profiles.py
```

本地优化器策略明确为 **reset per client job**：每个作业从其下载的联邦状态创建新的 AdamW，一、二阶矩不跨作业保留。该策略对所有比较方法一致；FedAdam 等服务器优化器状态独立保存在方法状态中。

## 异步语义

运行器先用虚拟时钟构造 TrainPlan，再按开始/到达事件重放。客户端在 `start_time` 下载当时的 `base_version` 和 LoRA 快照；即使服务器随后更新，该作业仍从旧快照训练。上传包含 `client_id`、`task`、`dataset`、`num_samples`、`base_version`、`arrival_time`、`local_state` 和 `delta`。服务器在到达时计算 `staleness = server_version - base_version`。

这是连续事件式异步协议，不等同于 FLGo 在同一时钟 tick 汇总同时返回模型的具体实现。两者都具有旧版本训练和异步到达，但本项目用持久化虚拟时间隔离系统异构，保证方法间使用完全相同的到达条件。普通异步方法训练中不会获得新全局模型；只有声明专属能力的 FedASMU 可执行一次中途新鲜模型调整。

## 方法

统一注册表支持：

- `local`：每个客户端独立 LoRA，无服务器聚合。
- `fedavg`：同步、样本量加权的 LoRA FedAvg。
- `fedprox`：在联邦可训练参数上加入 `mu/2 ||theta-theta_global||²`。
- `fedadam`：同步客户端 delta 与服务器 Adam 一、二阶矩。
- `fedasync`：逐到达插值，支持 constant、polynomial、hinge 陈旧函数。
- `fedbuff`：异步缓冲聚合，可选样本量/陈旧加权，结尾强制处理残余缓冲。
- `fedcompass`：计算能力感知的本地步数分配与分组半异步聚合。
- `fedasmu`：动态陈旧服务器更新以及一次训练中途新鲜全局调整。
- `masfl`、`adamasfl`：客户端/全局控制变量、历史下降动量；Ada 版本使用归一化局部方向与实际局部位移聚合。
- `pilot`：任务/客户端视觉适配器、CT-MoA 和任务/文本自适应聚合。
- `unifed_lora`：任务、模态、层和模块描述符驱动的服务器 LoRA 超网络。
- `ours`：仅注册未来接口；当前调用会抛出 `NotImplementedError: The proposed AFVLM method has not been implemented yet.`

FCIT、C2-AFCL 和 FedSpace 属于联邦持续/任务增量学习，不是当前固定任务客户端的强制 baseline；FedAST 的原问题是并行训练多个联邦模型，也不作为当前主 baseline。注册表保留后续扩展能力。

## 运行单个方法

接口为 `bash scripts/run_one.sh <method> <setting>`，其中 setting 为 2、5 或 10。以下是 2 clients/task 的每个正式 baseline 命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh local 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedavg 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedprox 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedadam 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedasync 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedbuff 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedcompass 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedasmu 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh masfl 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh adamasfl 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh pilot 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh unifed_lora 2
```

将最后一个参数改为 5 或 10 即选择对应数据集；例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh fedcompass 5
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh adamasfl 10
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh pilot 5
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one.sh unifed_lora 10
```

直接使用完整配置的等价命令为：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/llava/afvlm_cm/2clients/fedasync.yaml
```

批量运行全部 12 个 baseline（默认不包括尚未实现的 `ours`）：

```bash
bash scripts/run_baselines.sh 2
bash scripts/run_baselines.sh 5
bash scripts/run_baselines.sh 10
```

## 评估与输出

项目采用常见异步联邦学习评估协议：以**成功的服务器模型更新数**作为评估间隔。每达到 `evaluation.eval_every_server_updates`，复制当前服务器联邦状态并在公共 validation 集上分别评估六个任务；评估过程不计入虚拟训练时间，也不改变训练状态。全部 TrainPlan 到达事件完成后，先执行方法的结尾处理（例如刷新 FedBuff 残余缓冲），再用最终服务器状态在公共 test 集上评估。

所有存在服务器聚合的算法都以 `server_global` 为主评估结果。Pilot 仍使用任务路由，但使用当前服务器状态及同任务服务器端客户端视觉适配器的均值，不使用个性化 LoRA 替代全局主结果。Local-only 没有全局模型，因此是唯一例外：分别用每个客户端本地模型测试其所属任务的同一公共测试集，再在任务内等权平均，输出协议标记为 `client_local_mean`。

训练完成后可独立复评：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --config configs/experiments/llava/afvlm_cm/2clients/fedasync.yaml \
  --checkpoint runs/llava/afvlm_cm/2clients/fedasync/seed42/checkpoints/final_trainable.pt \
  --output runs/llava/afvlm_cm/2clients/fedasync/seed42/metrics.reproduced.json
```

每个运行目录包含 `resolved_config.yaml`、`system_profile.reference.json`、`train_plan.json`、`events.jsonl`、`updates.jsonl`、`task_metrics.jsonl`、`metrics.json`、`system_stats.json`、`train.log` 与 `checkpoints/final_trainable.pt`。评估文件显式记录 `protocol`、`split`、`server_version`、`virtual_time` 和 `per_task`。检查点只保存联邦可训练状态、服务器/方法/调度器/随机状态及必要 metadata，不保存冻结的 7B 主干。

系统统计包括 mean/median/max staleness、总/接受更新数、聚合次数、客户端/任务更新分布与虚拟训练时间；FedBuff 额外报告缓冲聚合次数、平均占用和平均等待时间；FedCompass 额外报告本地步数分配、分组完成时间与组内完成跨度。不同任务的量纲不兼容，因此不会把 accuracy、CIDEr 和 IoU 粗暴平均为一个原始分数。

数据接口和每种 baseline 的组件/方程/适配边界另见 `docs/AFVLM_CM.md` 与 `docs/baselines.md`。

## 静态工程检查

以下命令只检查 Python 语法/导入、YAML 解析与继承、注册表、39 个配置、路径以及原始分区完整性；不会加载 LLaVA、下载权重或启动训练：

```bash
python -m compileall -q code scripts tools
python tools/validate_repository.py
ruff check code scripts tools
```

## 与论文原始设置的差异

- FedCompass：保留计算能力感知步数分配和分组半异步核心，复用本项目的持久化虚拟时钟，没有复制原框架进程架构；优化状态限定为 LoRA。
- FedASMU：保留论文动态陈旧混合和中途新鲜模型调整。当前用确定性的可配置刷新位置及在线系数更新代替原工作完整的设备强化学习请求控制器，因此属于明确适配而非逐代码精确复现。
- MasFL/AdaMasFL：实现控制变量与两级动量方程；模型接口同时返回实际平均 LoRA 梯度。所有向量仅覆盖联邦可训练状态。
- Pilot：把专属视觉适配器和 CT-MoA 隔离在 Pilot 包装器内，其他方法的 LLaVA 不受影响；数据仍使用 AFVLM-CM 固定分区。当前一次运行联合优化两个阶段的损失，而不是另行执行论文中的独立预训练阶段。
- UniFed-LoRA：原论文支持异构主干；当前 `backbone heterogeneity = disabled`、`task heterogeneity = enabled`。超网络仍由任务/模态/层/模块描述符条件化，未使用硬编码客户端编号。

## 当前限制

- 代码未替用户启动 7B 训练；显存、CUDA、原始 LLaVA 版本与权重兼容性需要在目标训练机确认。
- Flickr30k、DVQA、FigureQA 和 COCO 受各自许可与分发条件约束；下载工具不会绕过许可页面。
- 当前训练入口保存可独立复评的最终联邦检查点，但尚未提供从任意中间异步事件恢复并继续训练的 CLI；这需要同时恢复仍在途的客户端作业快照。
- Flickr30k 指令当前每条只有一个参考答案；内置 CIDEr 使用标准 1--4 gram TF-IDF 余弦构造，但与拥有五参考标注的官方 COCO caption scorer 不能宣称数值完全等价。
- Pilot 的联合阶段适配和 FedASMU 的确定性刷新策略必须在论文中按上述差异披露。
- `ours` 故意没有算法实现，也不在 baseline 批处理里。

## 主要论文与数据来源

- FedCompass, ICLR 2024: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/a9f3457fa97f106f1756885237787789-Abstract-Conference.html>
- FedASMU, AAAI 2024: <https://ojs.aaai.org/index.php/AAAI/article/view/29297>
- MasFL/AdaMasFL, ICML 2025: <https://proceedings.mlr.press/v267/yan25f.html>
- Pilot, AAAI 2025: <https://ojs.aaai.org/index.php/AAAI/article/view/35476>
- UniFed-LoRA, CVPR Workshops 2026: <https://openaccess.thecvf.com/content/CVPR2026W/FedVision/html/Milasheuski_UniFed-LoRA_Exploiting_Semantic_Task_Correlation_for_Heterogeneous_Multimodal_Federated_Fine-Tuning_CVPRW_2026_paper.html>
- DVQA 官方数据仓库: <https://github.com/kushalkafle/DVQA_dataset>
- FigureQA 官方项目: <https://www.microsoft.com/en-us/research/project/figureqa-dataset/>
