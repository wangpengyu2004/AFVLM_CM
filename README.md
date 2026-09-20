# AFVLM-CM

AFVLM-CM 是用于论文实验的异步联邦视觉语言模型指令微调框架，研究固定任务客户端下的任务异构与系统异构：每个客户端永久属于一个任务，客户端速度、可用时间和网络延迟不同，但不存在持续学习或任务增量流。

正式实验只使用 LLaVA-v1.5-7B、CLIP ViT-L/14-336 和 LoRA。默认 LoRA 为 r=8、alpha=16、dropout=0.05、bias=none；默认随机种子为 42。默认运行时自动发现全部可见 GPU，并为每张卡创建一个独立 worker；每个 worker 常驻一份完整的冻结 7B 主干并训练不同客户端。权威服务器的 LoRA 聚合、FedAdam 矩状态以及方法状态默认放在第一张可见 GPU，CPU 只负责调度、进程通信、日志和数据加载。冻结主干不会上传、聚合或写入联邦检查点。

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

数据读取器兼容该 benchmark 实际存在的三种标注形式：LLaVA `conversations`、扁平的 `text + answer`，以及 Grounding 测试集的 `text + answer_bbox`。正常训练和独立评估不会在启动时全量扫描图片；样本真正进入 batch 时才读取对应图片。需要检查数据时，显式运行独立工具，它会验证当前设置下的全部 `client_N.json`、`val.json`、`test.json`，并对去重后的图片路径逐一检查：

```bash
python tools/validate_afvlm_cm_data.py --setting 2
python tools/validate_afvlm_cm_data.py --setting 5
python tools/validate_afvlm_cm_data.py --setting 10
# 或一次检查三档
python tools/validate_afvlm_cm_data.py --all
```

检查工具显示注解记录和唯一图片两个进度条；任一错误会报告任务、split、文件和记录位置。该工具只读取数据，不修改指令文件或图片。

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

`requirements.txt` 已包含 CUDA 11.8 对应的 PyTorch 2.0.1/TorchVision 0.15.2、LLaVA、Transformers、PEFT、数据处理、绘图和工程检查依赖，并以 editable 模式安装当前 AFVLM-CM 包。LLaVA 源码固定到官方 `v1.1.3`，通过体积固定的 release archive 安装而不是执行 `git clone`，可避免训练服务器连接 GitHub git 服务超时。环境安装不会下载 7B/CLIP 预训练参数。请在仓库根目录运行命令，并确保宿主机 NVIDIA 驱动兼容 CUDA 11.8。NF4 量化需要 bitsandbytes；Tesla V100 正式配置使用非量化 FP16。正式训练只读取本地模型权重。

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

这三类目录职责不同，不能互相替代：

| 目录 | 作用 | 是否应修改/删除 |
|---|---|---|
| `configs/methods/` | 各算法的可编辑参数源；39 个实验入口通过 `inherits` 读取它们，创建新 profile 时也从这里解析当前方法参数 | 可以按实验需要修改；不能删除 |
| `plans/afvlm_cm/` | 默认兼容运行使用的 system profile 与 TrainPlan；保证 `bash scripts/run_one.sh <method> <setting>` 和原始实验 YAML 仍可直接运行 | 不要手工修改；当前不能删除 |
| `experiment_profiles/` | 已冻结的完整配置、system profile 和 TrainPlan 快照，用于正式实验复现和切换旧实验 | 不要修改或覆盖；新参数应创建新 profile |

因此，平时应修改 `configs/base.yaml`、`configs/models/` 或 `configs/methods/`，然后生成新的 `experiment_profiles/<name>/`。正式运行优先指定 profile；`plans/` 仅作为默认兼容入口保留。

同一规模的所有方法继承相同模型、LoRA、客户端优化器、学习率、本地 epoch、batch size、随机种子、数据与评估配置。`plans/afvlm_cm/{2,5,10}clients/` 保留最初的默认系统画像和 TrainPlan；新的正式实验应创建不可变的命名 profile。每个 profile 同时保存完全解析后的 39 份实验配置、三档 system profile/TrainPlan、文件哈希和唯一输出路径，因此以后修改 `base.yaml` 或方法配置不会改变旧实验。除 FedCompass 的本地工作量分配外，同一 profile 内的异步算法使用相同到达条件。

普通方法的本地训练量只由 `training.local_epochs` 控制，不再同时设置通用的 `base_local_steps`。TrainPlan 中记录的 `local_steps` 是根据客户端样本数、batch size、梯度累积和 `local_epochs` 推导出的预计优化器步数，只用于虚拟耗时和算法钩子，不会截断普通客户端训练。FedCompass 是唯一例外：其计算能力感知调度器必须动态分配本地迭代，因此使用 `min_local_steps` / `max_local_steps` 作为方法专属边界。

仓库已保存初始快照 `default_e1_bs1_ga4_r10_s42`。查看已有 profile：

```bash
python tools/generate_system_profiles.py --list
```

修改参数后创建一个新 profile。若修改了 `local_epochs`、batch size、梯度累积或 rounds，必须从服务器本地数据生成新 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile e2_bs1_ga4_r10_s42
```

如果只修改学习率、模型参数或方法专属参数，且本地训练量与 rounds 没变，可以显式复用旧 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile lr1e5_alpha03 \
  --reuse_plans_from default_e1_bs1_ga4_r10_s42
```

profile 名称只允许字母、数字、点、下划线和连字符。同名目录默认拒绝覆盖；要保留旧实验时必须使用新名称。生成器会先检查复用 Plan 与当前 `local_epochs`、batch size、梯度累积和 rounds 是否兼容。

## 从修改参数到服务器运行的完整流程

以下命令均在服务器仓库根目录执行。服务器本地的 `data/`、`pretrained/`、`runs/`、`wandb/`、`outputs/` 和 `checkpoints/` 已由根目录 `.gitignore` 排除，正常的 `fetch + fast-forward merge` 不会删除或提交这些目录。不要使用 `git clean -fdx`、`git reset --hard` 或手动删除上述目录。

### 1. 拉取代码并检查本地资源

```bash
cd /userhome/bcx/AFVLM_CM
git switch main
git fetch origin main
git merge --ff-only origin/main
conda activate afvlm-cm

test -d data/AFVLM_CM/partitioned
test -f pretrained/llava-v1.5-7b/config.json
test -f pretrained/clip-vit-large-patch14-336/config.json
```

三个 `test` 命令没有输出且退出码为 0，表示路径存在。数据与权重只保存在服务器，不需要上传 GitHub。

### 2. 修改配置

公共训练参数在 `configs/base.yaml`，LoRA 和本地权重路径在 `configs/models/llava15_7b_lora.yaml`，算法专属参数在 `configs/methods/<method>.yaml`。例如：

```bash
nano configs/base.yaml
nano configs/methods/fedasync.yaml
```

修改后先判断是否需要新 Plan：

| 修改内容 | 是否可复用旧 Plan | 操作 |
|---|---:|---|
| 学习率、最大文本长度、LoRA 参数、评估间隔 | 是 | 创建新 profile 并指定 `--reuse_plans_from` |
| FedProx `mu`、FedAsync `alpha`、FedBuff `buffer_size` 等方法参数 | 是 | 创建新 profile 并指定 `--reuse_plans_from` |
| GPU 数量、FP16/BF16、运行时 backend | 是 | 创建新 profile 并指定 `--reuse_plans_from` |
| `local_epochs`、`batch_size`、`gradient_accumulation` | 否 | 创建新 profile，不传 `--reuse_plans_from` |
| 六个 `task_compute_factors` | 否 | 创建新 profile，不传 `--reuse_plans_from` |
| `federation.rounds`、随机种子或系统异构性生成规则 | 否 | 创建新 profile，不传 `--reuse_plans_from` |
| 只把客户端档位从 2 换成 5 或 10 | 不需要修改配置 | 运行时修改第二个参数 |

### 3A. 参数不影响 Plan：复用旧 Plan

假设只把学习率改为 `1e-5`、FedAsync `alpha` 改为 `0.3`：

```bash
python tools/generate_system_profiles.py \
  --profile lr1e5_alpha03 \
  --reuse_plans_from default_e1_bs1_ga4_r10_s42
```

该命令复用完全相同的客户端速度、网络延迟、可用时间和到达顺序，但会把当前全部配置解析后保存到新 profile，适合公平的算法参数对比。

### 3B. 参数影响 Plan：生成新 Plan

假设把 `local_epochs` 改为 2、`rounds` 改为 20：

```bash
python tools/generate_system_profiles.py \
  --profile e2_bs1_ga4_r20_s42
```

此时不要传 `--reuse_plans_from`。生成器会读取服务器现有的 `data/AFVLM_CM/partitioned/`，为 2、5、10 clients/task 三档分别生成 system profile 与 TrainPlan，并保存全部 39 份解析后配置；不会改写 `plans/afvlm_cm/` 或任何旧 profile。

如果需要新的随机种子，可显式指定：

```bash
python tools/generate_system_profiles.py \
  --profile e2_bs1_ga4_r20_s123 \
  --seed 123
```

### 4. 检查新 profile

```bash
python tools/generate_system_profiles.py --list
python tools/validate_repository.py
```

profile 的固定内容位于：

```text
experiment_profiles/<profile>/
├── manifest.yaml
├── configs/{2,5,10}clients/*.yaml
└── plans/{2,5,10}clients/
    ├── system_profile.json
    └── async_train_plan.json
```

`manifest.yaml` 记录创建时间、源提交、训练参数、rounds、方法清单和文件 SHA-256。不要直接编辑 profile 内部文件；参数有变化时创建新名称。若提示 `Experiment profile already exists`，说明旧实验受保护，应换一个 profile 名称。

### 5. 运行单个方法或全部 baseline

```bash
# 单个方法：<method> <2|5|10> <profile>
bash scripts/run_one.sh fedasync 2 lr1e5_alpha03

# 同一 profile 下依次运行全部 12 个 baseline
bash scripts/run_baselines.sh 2 lr1e5_alpha03
```

创建 profile 会为每种方法写入唯一输出路径。上述单方法结果位于：

```text
runs/llava/afvlm_cm/profiles/lr1e5_alpha03/2clients/fedasync/seed42/
```

`configs/base.yaml` 中的 `output.directory: runs/afvlm_cm` 只是公共回退值；普通实验配置会将其覆盖为方法目录，profile 配置还会进一步覆盖为上述独立目录。因此看到 `runs/llava/afvlm_cm/...` 是预期行为。

### 6. 切回旧参数和旧 Plan

不需要恢复或再次编辑 `configs/base.yaml`，直接把运行命令的第三个参数换回旧 profile：

```bash
bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_e1_bs1_ga4_r10_s42
```

历史 profile 中保存的 `worker_queue: fifo` 不会被静默改写，仍可复现原物理调度；要使用
EDF 必须创建新 profile。上面的新 profile 可以复用原 TrainPlan，但会从当前基础配置记录
`plan_arrival_edf`，无需重新随机生成系统异构条件。

注意：`default_e1_bs1_ga4_r10_s42` 是历史不可变的单执行器/BF16 快照。V100 八卡不要继续使用该旧 profile；按下文“八卡运行”先生成新的 FP16 profile。切回任何旧参数仍然只需替换第三个参数。

正式论文实验建议始终传入第三个 profile 参数。省略第三个参数时使用当前可变的 `configs/` 与兼容目录，只适合临时调试，不适合作为最终可复现实验记录。

### 7. 保存服务器上新建的实验定义到 GitHub

服务器生成的 profile 可以提交，数据、权重和运行输出仍会被忽略。先检查提交范围，不要使用 `git add -A`：

```bash
git status --short
git add configs/base.yaml \
  configs/methods/fedasync.yaml \
  experiment_profiles/lr1e5_alpha03
git commit -m "chore(experiment): add lr1e5 alpha03 profile"
git push origin main
```

将方法配置路径和 profile 名替换为本次实际内容。若服务器只负责运行、实验定义统一由电脑端维护，则无需在服务器执行这一提交步骤。

本地优化器策略明确为 **reset per client job**：每个作业从其下载的联邦状态创建新的 AdamW，一、二阶矩不跨作业保留。该策略对所有比较方法一致；FedAdam 等服务器优化器状态独立保存在方法状态中。

## 异步语义

运行器按常见异步联邦框架拆分为权威 Server/Method、虚拟时钟 Scheduler、Dispatcher、每张可见 GPU 一个常驻 Client Trainer 和 Aggregator。TrainPlan 只保存开始时间、到达时间、客户端速度、网络延迟、本地工作量和可选调度组，**不保存全局模型或指定客户端必须下载哪个参数版本**。运行器在 `start_time` 读取当时的 `base_version`，再调用该方法自己的 `prepare_download`：普通方法下载全局 LoRA，Local-only 下载客户端本地状态，Pilot 下载个性化状态，UniFed-LoRA 下载任务条件化状态，MasFL/AdaMasFL 同时冻结本地控制变量。即使服务器随后更新，该作业也仍从已冻结的 dispatch package 训练。

worker 完成后先返回未经服务器处理的原始更新；父进程在计划的 `arrival_time` 调用权威方法实例的 `prepare_upload` 和 `on_arrival`。上传包含 `client_id`、`task`、`dataset`、`num_samples`、`base_version`、`arrival_time`、`local_state` 和 `delta`。服务器在到达时计算 `staleness = server_version - base_version`。FedASMU 的中途新鲜模型访问被建模为显式虚拟事件，不依赖某一次运行中偶然的 GPU 完成先后。

默认 `runtime.arrival_policy: planned` 会按持久化虚拟到达顺序应用更新，`runtime.worker_queue: plan_arrival_edf` 则只优化物理 GPU 执行顺序。逻辑 start 时先冻结 `base_version` 和方法产生的下载状态，在下一次计划 arrival 前把所有已经合法启动的作业放入候选堆，再优先执行计划 `arrival_time` 最早者；提前算完但尚未轮到的结果进入缓存。EDF 不重算下载版本、不提前聚合，也不允许未来 start 事件偷跑，因此某次运行的磁盘抖动或 GPU 波动只影响 wall-clock，不改变方法比较的逻辑到达条件。FedCompass 可以改变本地步数和分组，因为这是算法本身；其他方法不能通过修改 Plan 偷换系统条件。完整生命周期、特殊方法处理、方法服务器状态和任务耗时校准见 [`docs/async_runtime.md`](docs/async_runtime.md)。

### V100 八卡运行

当前可编辑配置已经设置 `runtime.backend: client_parallel`、`runtime.devices: all`、GPU 服务器聚合和 `model.dtype: fp16`。程序会使用当前进程可见的全部 GPU，不需要在普通运行命令中指定卡号。历史 profile 不会自动继承这些修改，因此先创建新的不可变多卡 profile：

```bash
python tools/generate_system_profiles.py \
  --profile v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42 \
  --reuse_plans_from default_e1_bs1_ga4_r10_s42

bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42
```

每张 V100 会占用一份完整 7B FP16 模型及一个客户端的激活/优化器状态；这和原先 `device_map: auto` 把一个模型切到多卡不同。正常情况下无需设置 `CUDA_VISIBLE_DEVICES`。如果服务器还有其他任务、只想临时使用部分卡，可以选择性地设置该环境变量；程序会自动使用其中所有可见设备。

若某些任务每个 optimizer step 本来就更慢，先完成一轮运行，再执行：

```bash
python tools/estimate_task_compute_factors.py \
  runs/llava/afvlm_cm/profiles/v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42/2clients/fedasync/seed42
```

把输出因子写入 `configs/base.yaml` 后创建不复用旧 Plan的新 profile。任务固有耗时使用 `task_compute_factors`，设备差异使用 `speed_factor`，二者分开保存。

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
- `ours`：敏感性感知异步 LoRA 整合；使用 LoRA group sensitivity、功能性陈旧度、按任务历史敏感性记忆和逐组精度融合。默认 `module_gate`，并支持 `rank_gate`、`adam_v` 消融。

FCIT、C2-AFCL 和 FedSpace 属于联邦持续/任务增量学习，不是当前固定任务客户端的强制 baseline；FedAST 的原问题是并行训练多个联邦模型，也不作为当前主 baseline。注册表保留后续扩展能力。

## 运行单个方法

接口为 `bash scripts/run_one.sh <method> <setting> [profile]`，其中 setting 为 2、5 或 10。省略 profile 时使用原始兼容配置；提供 profile 时使用对应的不可变配置快照。以下是 2 clients/task 的全部正式 baseline 及提出方法命令：

```bash
bash scripts/run_one.sh local 2
bash scripts/run_one.sh fedavg 2
bash scripts/run_one.sh fedprox 2
bash scripts/run_one.sh fedadam 2
bash scripts/run_one.sh fedasync 2
bash scripts/run_one.sh fedbuff 2
bash scripts/run_one.sh fedcompass 2
bash scripts/run_one.sh fedasmu 2
bash scripts/run_one.sh masfl 2
bash scripts/run_one.sh adamasfl 2
bash scripts/run_one.sh pilot 2
bash scripts/run_one.sh unifed_lora 2
bash scripts/run_one.sh ours 2
```

`ours` 的默认主配置使用 `module_gate` sensitivity。推荐为正式实验生成独立的不可变 profile：

```bash
python tools/generate_system_profiles.py \
  --profile ours_module_gate_e1_bs1_ga4_r10_s42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/run_one.sh ours 2 ours_module_gate_e1_bs1_ga4_r10_s42
```

如需 `rank_gate` 或 `adam_v` 消融，修改准备实验所用的 `configs/methods/ours.yaml`，并生成名称中明确包含 estimator 的新 profile；不要修改已保存的 profile。完整参数与公式见 `docs/ours.md`。

使用前文创建的正式 V100 八卡快照：

```bash
bash scripts/run_one.sh fedavg 2 v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42
bash scripts/run_one.sh fedasync 5 v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42
```

profile 结果写入独立目录，例如：

```text
runs/llava/afvlm_cm/profiles/v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42/2clients/fedavg/seed42/
```

将最后一个参数改为 5 或 10 即选择对应数据集；例如：

```bash
bash scripts/run_one.sh fedcompass 5
bash scripts/run_one.sh adamasfl 10
bash scripts/run_one.sh pilot 5
bash scripts/run_one.sh unifed_lora 10
bash scripts/run_one.sh ours 2
```

直接使用完整配置的等价命令为：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/llava/afvlm_cm/2clients/fedasync.yaml
```

批量运行全部 12 个 baseline（`ours` 是待比较的提出方法，因此不加入 baseline 批处理）：

```bash
bash scripts/run_baselines.sh 2
bash scripts/run_baselines.sh 5
bash scripts/run_baselines.sh 10
```

对一个保存的 profile 批量运行：

```bash
bash scripts/run_baselines.sh 2 v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42
```

### 命令行训练进度

训练进度条默认开启，直接使用原有命令即可，不需要重新生成 system profile 或 TrainPlan：

```bash
bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42
```

命令行会依次显示：

- 元数据扫描完成以及所有可见 GPU worker 分别就绪的提示；
- `fedasync virtual arrivals`：当前已应用的虚拟到达数 / TrainPlan 总更新数，并显示服务器版本、当前客户端和 staleness；
- `GPU<n> <client_id>`：每张卡当前客户端的真实 optimizer step / 计划 optimizer step，并动态显示 loss；
- `evaluate <task>`：定期 validation 和最终 test 时各任务已评估样本数。

本地进度按 optimizer step 计数，不按 gradient accumulation 的 micro-batch 计数。因此，若配置为 `gradient_accumulation: 4`，进度条增加 1 代表已经完成 4 个 micro-batch 的梯度累积及 1 次参数更新。进度显示只读取已有训练状态，不会改变 local epoch、TrainPlan、聚合顺序或虚拟时间。

`batch_size` 是每个 GPU worker 的真实多模态 micro-batch 大小：文本在当前 batch 内动态 padding，图像组成同一个 batch tensor，并通过一次 LLaVA forward/backward 处理。每个客户端的有效批量为 `batch_size × gradient_accumulation`；例如 `batch_size: 4`、`gradient_accumulation: 4` 对应有效批量 16。增大 `batch_size` 会提高单卡显存占用，修改后必须创建匹配的新实验 profile 和 TrainPlan。

若需要把终端输出重定向到文件，建议同时打开 Python 非缓冲输出：

```bash
PYTHONUNBUFFERED=1 \
  bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_edf_e1_bs1_ga4_r10_s42 \
  2>&1 | tee fedasync-2clients.log
```

如需关闭进度条，在准备创建新 profile 的基础配置中设置：

```yaml
output:
  progress_bar: false
```

已经保存的不可变 profile 不需要补写该字段：缺省值就是 `true`。若要永久改变某个旧实验的显示选项，应创建新的配置 profile，而不是修改旧快照。

## 评估与输出

项目采用常见异步联邦学习评估协议：以**成功的服务器模型更新数**作为评估间隔。每达到 `evaluation.eval_every_server_updates`，复制当前服务器联邦状态并在公共 validation 集上分别评估六个任务；评估过程不计入虚拟训练时间，也不改变训练状态。全部 TrainPlan 到达事件完成后，先执行方法的结尾处理（例如刷新 FedBuff 残余缓冲），再用最终服务器状态在公共 test 集上评估。

所有存在服务器聚合的算法都以 `server_global` 为主评估结果。Pilot 仍使用任务路由，但使用当前服务器状态及同任务服务器端客户端视觉适配器的均值，不使用个性化 LoRA 替代全局主结果。Local-only 没有全局模型，因此是唯一例外：分别用每个客户端本地模型测试其所属任务的同一公共测试集，再在任务内等权平均，输出协议标记为 `client_local_mean`。

训练完成后可独立复评：

```bash
python scripts/evaluate.py \
  --config configs/experiments/llava/afvlm_cm/2clients/fedasync.yaml \
  --checkpoint runs/llava/afvlm_cm/2clients/fedasync/seed42/checkpoints/final_trainable.pt \
  --output runs/llava/afvlm_cm/2clients/fedasync/seed42/metrics.reproduced.json
```

每个运行目录包含 `resolved_config.yaml`、`system_profile.reference.json`、`train_plan.json`、`events.jsonl`、`updates.jsonl`、`task_metrics.jsonl`、`metrics.json`、`system_stats.json`、`train.log` 与 `checkpoints/final_trainable.pt`。评估文件显式记录 `protocol`、`split`、`server_version`、`virtual_time` 和 `per_task`。并行更新日志还记录 worker/GPU 与真实开始、完成时间。检查点只保存联邦可训练状态、服务器/方法/调度器/随机状态及必要 metadata，不保存冻结的 7B 主干。

系统统计包括 mean/median/max staleness、总/接受更新数、聚合次数、客户端/任务更新分布、虚拟训练时间、真实 wall-clock 时间和 GPU worker 数；FedBuff 额外报告缓冲聚合次数、平均占用和平均等待时间；FedCompass 额外报告本地步数分配、分组完成时间与组内完成跨度。不同任务的量纲不兼容，因此不会把 accuracy、CIDEr 和 IoU 粗暴平均为一个原始分数。

数据接口和每种 baseline 的组件/方程/适配边界另见 `docs/AFVLM_CM.md` 与 `docs/baselines.md`。提出方法的公式、三种 sensitivity estimator、配置和运行流程见 `docs/ours.md`。

## 静态工程检查

以下命令只检查 Python 语法/导入、YAML 解析与继承、注册表、39 个配置、路径以及原始分区完整性；不会加载 LLaVA、下载权重或启动训练：

```bash
python -m compileall -q code scripts tools
python tools/validate_repository.py
python tools/validate_afvlm_cm_data.py --setting 2  # 可选的全量数据/图片检查
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
- `ours` 已实现，但其 sensitivity 是局部二次精度代理，并非完整 Hessian 或真实任务泛化重要性；当前只支持 LoRA-only 联邦状态。`ours` 不在 baseline 批处理里，应使用独立命令运行。

## 主要论文与数据来源

- FedCompass, ICLR 2024: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/a9f3457fa97f106f1756885237787789-Abstract-Conference.html>
- FedASMU, AAAI 2024: <https://ojs.aaai.org/index.php/AAAI/article/view/29297>
- MasFL/AdaMasFL, ICML 2025: <https://proceedings.mlr.press/v267/yan25f.html>
- Pilot, AAAI 2025: <https://ojs.aaai.org/index.php/AAAI/article/view/35476>
- UniFed-LoRA, CVPR Workshops 2026: <https://openaccess.thecvf.com/content/CVPR2026W/FedVision/html/Milasheuski_UniFed-LoRA_Exploiting_Semantic_Task_Correlation_for_Heterogeneous_Multimodal_Federated_Fine-Tuning_CVPRW_2026_paper.html>
- DVQA 官方数据仓库: <https://github.com/kushalkafle/DVQA_dataset>
- FigureQA 官方项目: <https://www.microsoft.com/en-us/research/project/figureqa-dataset/>
