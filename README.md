# AFVLM-CM

新增独立 `aced`：全客户端初始梯度与首步下降、活跃梯度缓存均值、超时筛选及重新参与。
沿用固定任务分区、虚拟 Plan 和全局评估，现支持全部可见 GPU 协同的 DDP 梯度收集；
不改已有 DDP 训练或其他方法。多步 AdamW、额外初始化成本与原文时间戳差异均明确披露，
不声称原论文精确复现。配置和启动见 [ACED 说明](docs/aced.md)。

FedASMU、FedCompass、MasFL/AdaMasFL、ACED、UniFed-LoRA 现默认按能力选择 DDP；
Pilot 因动态训练参数仍使用客户端并行。改动仅在相应方法内，公共训练/评估和 ours/OPCM
不变；UniFed 的超网络不会被 DDP 分摊显存。详见 [方法内 DDP 适配与启动](docs/baseline_ddp.md)。

新增独立异步 baseline `ca2fl`（ICLR 2024）：客户端旧更新缓存差分校准、全局缓存均值、
缓冲聚合，以及可选 4/8-bit MF 缓存。默认 `server_lr=1`、`buffer_size=3`、完整 FP32 缓存；
沿用现有 DDP、本地训练、虚拟 Plan 与全局评估，不修改 `ours` 或其他 baseline。
运行、参数、显存成本和原文适配差异见 [CA²FL 说明](docs/ca2fl.md)。

本次 baseline 功能审查见 [审查报告](docs/baseline_audit.md)：以核心机制完整、功能正确为标准，
不是逐行复制官方框架。按最新要求加入OrthoFL并补全无服务器数据的FedASMU功能适配，
额外目标计算/原文差异见[专门说明](docs/orthofl_fedasmu.md)。Pilot/UniFed-LoRA保持
全局主评估，训练结束另存原生个性化附加结果。
新增 `c2afcl`：保留原文客户端历史差分与跨任务 SVD 正交校准，作为固定任务 LoRA
适配 baseline。实现差异、参数与运行步骤见 [C2-AFCL 说明](docs/c2afcl.md)。
修正后的方法需要新 profile；旧快照不自动更新。未启动训练，不能把非训练检查当作 GPU
训练或原论文精度的保证。

AFVLM-CM 是用于论文实验的异步联邦视觉语言模型指令微调框架，研究固定任务客户端下的任务异构与系统异构：每个客户端永久属于一个任务，客户端速度、可用时间和网络延迟不同，但不存在持续学习或任务增量流。

正式实验只使用 LLaVA-v1.5-7B、CLIP ViT-L/14-336 和 LoRA。默认 LoRA 为 r=8、alpha=16、dropout=0.05、bias=none；默认随机种子为 42。LoRA 仅作用于语言模型侧：即使 `q_proj/k_proj/v_proj` 名称同时匹配 CLIP，冻结的 vision tower/vision resampler 参数也会从优化器、DDP 归约和联邦状态中排除。默认运行时发现全部可见 GPU，按方法能力选择执行器：除 Pilot 外的当前方法支持全部 GPU 对同一客户端做 DDP，方法自身处理 SGD、初始化与中途刷新等特殊需要；Pilot 的动态 adapter 保留“一卡一个客户端”worker pool。普通异步方法仍按虚拟 TrainPlan 冻结 base version 并按计划到达聚合。冻结 7B 主干不会上传、聚合或写入联邦检查点。

## 已检测的数据

框架训练时只读 `data/AFVLM_CM`，不会隐式生成、重分区或改写指令文件。需要重建论文固定分区时，必须显式运行后文的版本化划分工具。

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
| `caption` | Flickr30k | BLEU-1..4、METEOR、ROUGE-L、CIDEr、FCIT average | `Flickr30k/train/` |
| `vqa` | AOKVQA | VQA accuracy | `COCO2014/train2014/`、`val2014/` |
| `chart_vqa` | DVQA | answer accuracy | `DVQA/images/` |
| `visual_reasoning` | FigureQA | answer accuracy | `FigureQA/images/` |
| `grounding` | COCO grounding | mean IoU、IoU@0.5 accuracy | 与 AOKVQA 共用 `COCO2014/` |

每个任务目录包含 `client_N.json`、`statistics.json`、`val.json` 和 `test.json`。当前固定版本为 `fcit_eval_v2_grounding8k_68k`，六任务训练记录数依次为 13,000 / 15,000 / 11,000 / 9,500 / 11,500 / 8,000，总计 68,000。该版本只把 Grounding 训练池由 4,500 条扩展为 8,000 条，保留旧训练池全部记录；其多轮结构、validation/final 以及其他五个任务均保持不变。2/5/10 clients/task 三档复用同一任务池，只改变客户端划分。启动时会再次扫描实际文件数；配置中的 12/30/60 不是客户端清单的替代品。`configs/datasets/afvlm_cm_integrity.json` 固定该版本的静态完整性摘要。

数据读取器兼容该 benchmark 实际存在的三种标注形式：LLaVA `conversations`、扁平的 `text + answer`，以及 Grounding 测试集的 `text + answer_bbox`。Grounding 训练会保留完整多轮对话并让每个 assistant 回答参与 loss；validation/final 会把每个 user/assistant 对展开成独立 query。Caption 的 validation/test 每张图片保留五个参考 caption。`max_text_length` 默认使用 LLaVA 的 2k 上下文；若某个多轮样本仍超长，collator 会明确报错而不会静默截掉后面的 assistant 回答。正常训练和独立评估不会在启动时全量扫描图片；样本真正进入 batch 时才读取对应图片。

### 重建固定 68k 分区

下面命令从 FCIT 指令文件构建一个新候选目录；如果目标目录非空会直接拒绝，不会覆盖当前 `partitioned/`：

```bash
python tools/partition_afvlm_cm.py \
  --source_root data/AFVLM_CM/instruction \
  --output_dir data/AFVLM_CM/partitioned_fcitev2_grounding8k_68k
```

验证候选目录后，再人工备份旧目录并启用新目录。数据目录不上传 GitHub，因此电脑端和服务器端都要保存/同步同一个固定分区：

```bash
# 以下迁移适用于当前仍使用旧 64.5k 分区的服务器；备份目标必须不存在。
test ! -e data/AFVLM_CM/partitioned_fcitev2_grounding4k5_64k5 || exit 1
mv data/AFVLM_CM/partitioned data/AFVLM_CM/partitioned_fcitev2_grounding4k5_64k5
mv data/AFVLM_CM/partitioned_fcitev2_grounding8k_68k data/AFVLM_CM/partitioned
python tools/validate_afvlm_cm_data.py --all
```

划分以问题/指令记录为隔离单位，沿用现有 FCIT 来源及固定验证/测试记录；DVQA、FigureQA、Grounding 中不同问题或区域可以引用同一原图，因此不宣称图片级隔离。训练池使用固定 seed=42 的均匀无放回抽样，再打乱并按不等数量分配客户端；客户端大小限制在同任务平均值的约 0.7～1.3 倍，不额外引入标签 Dirichlet 偏斜。每个任务总量不同，同任务客户端大小也不相等；完整数量保存在各任务的 `statistics.json` 和 `partitioned/metadata/partition_report.md`。更换分区后，旧 system profile/TrainPlan 的 `num_samples` 已失效，必须创建不复用旧 Plan 的新 experiment profile：

```bash
# 先确认 configs/base.yaml 的实际训练参数；名称本身不设置参数。
python tools/generate_system_profiles.py --profile grounding8k_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ours 2 grounding8k_e1_bs4_ga4_r10_s42
```

本地旧 64.5k 分区保存在 `data/AFVLM_CM/partitioned_fcitev2_grounding4k5_64k5`，旧 72k 分区备份也保留。此数据调整不修改训练方法、评估规则或批量设置。

需要检查数据时，显式运行独立工具。它会验证当前设置下的全部 `client_N.json`、`val.json`、`test.json`，并对去重后的图片路径逐一检查：

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

`requirements.txt` 已包含 CUDA 11.8 对应的 PyTorch 2.0.1/TorchVision 0.15.2、LLaVA、Transformers、PEFT、`pycocoevalcap`、数据处理、绘图和工程检查依赖，并以 editable 模式安装当前 AFVLM-CM 包。LLaVA 源码固定到官方 `v1.1.3`，通过体积固定的 release archive 安装而不是执行 `git clone`，可避免训练服务器连接 GitHub git 服务超时。Caption 的 METEOR 评分还要求系统已安装 Java（例如 `openjdk-8-jre-headless`）；环境安装不会下载 7B/CLIP 预训练参数。请在仓库根目录运行命令，并确保宿主机 NVIDIA 驱动兼容 CUDA 11.8。NF4 量化需要 bitsandbytes；Tesla V100 正式配置使用非量化 FP16。正式训练只读取本地模型权重。

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
├── methods/{local,...,unifed_lora,ours,opcm_lora,dop_lora,nufilt_lora}.yaml
└── experiments/llava/afvlm_cm/
    ├── 2clients/    # 20 个常规方法 + 1 个独立 one-shot 入口
    ├── 5clients/    # 20 个常规方法 + 1 个独立 one-shot 入口
    └── 10clients/   # 20 个常规方法 + 1 个独立 one-shot 入口
```

这三类目录职责不同，不能互相替代：

| 目录 | 作用 | 是否应修改/删除 |
|---|---|---|
| `configs/methods/` | 各算法的可编辑参数源；54 个常规入口与3个独立 one-shot 入口通过 `inherits` 读取；两种生成器分别解析对应方法参数 | 可以按实验需要修改；不能删除 |
| `plans/afvlm_cm/` | 默认兼容运行使用的 system profile 与 TrainPlan；保证 `bash scripts/run_one.sh <method> <setting>` 和原始实验 YAML 仍可直接运行 | 不要手工修改；当前不能删除 |
| `experiment_profiles/` | 已冻结的完整配置、system profile 和 TrainPlan 快照，用于正式实验复现和切换旧实验 | 不要修改或覆盖；新参数应创建新 profile |

因此，平时应修改 `configs/base.yaml`、`configs/models/` 或 `configs/methods/`，然后生成新的 `experiment_profiles/<name>/`。正式运行优先指定 profile；`plans/` 仅作为默认兼容入口保留。

同一规模的普通方法继承相同模型、LoRA、客户端 AdamW、学习率、本地 epoch、batch size、随机种子、数据与评估配置。算法必要例外须单独披露：MasFL/AdaMasFL 用 SGD，FedCompass 动态分配本地步数，Pilot 增加视觉适配器，UniFed-LoRA 使用客户端生成参数，FedASMU额外获取新全局状态并计算客户端目标反馈；不能宣称这些方法具有完全相同的参数量或成本。`plans/afvlm_cm/{2,5,10}clients/` 保留最初的默认系统画像和 TrainPlan；新的正式实验应创建不可变的命名 profile。新 profile 保存完全解析后的 54 份实验配置、三档 system profile/TrainPlan、文件哈希和唯一输出路径；历史 profile 仍保留原状，因此以后修改 `base.yaml` 或方法配置不会改变旧实验。除 FedCompass 的本地工作量分配外，同一 profile 内的异步算法使用相同到达条件。

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

## 服务器上第一次启动正式训练（推荐照抄）

下面以服务器目录 `/userhome/bcx/AFVLM_CM`、Conda 环境 `afvlm-cm`、8 张可见
V100、2 clients/task 和 Module-Gate `ours` 为例。命令都在服务器执行。

### 1. 安全拉取电脑端最新代码

`data/`、`pretrained/`、`runs/`、`wandb/`、`outputs/` 和 `checkpoints/` 已被
`.gitignore` 排除，正常 fast-forward 拉取不会覆盖它们。先确认服务器没有未保存的
**tracked 代码/配置修改**：

```bash
cd /userhome/bcx/AFVLM_CM
git status --short
```

如果输出为空，直接拉取：

```bash
git switch main
git fetch origin main
git merge --ff-only origin/main
```

如果只看到类似 `M configs/base.yaml` 的服务器配置修改，先保存它，再拉取：

```bash
git stash push -m "server config before pull" -- configs/base.yaml configs/methods
git switch main
git fetch origin main
git merge --ff-only origin/main
git stash pop
```

若 `stash pop` 报冲突，不要使用 `git reset --hard`；应检查并手工合并配置。服务器上的
大数据和权重属于 ignored 文件，不需要 stash，也不要执行 `git clean -fdx`。

### 2. 激活环境并确认 8 卡、依赖、权重和数据

```bash
conda activate afvlm-cm

nvidia-smi -L
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available(), 'gpus=', torch.cuda.device_count())"

test -f pretrained/llava-v1.5-7b/config.json
test -f pretrained/clip-vit-large-patch14-336/config.json
test -d data/AFVLM_CM/partitioned/2_clients
test -d data/AFVLM_CM/dataset
```

四个 `test` 命令无输出且退出码为 0 才表示路径存在。数据全量格式/图片检查是独立工具，
不会在正常训练启动时自动运行；只有需要时才执行：

```bash
python tools/validate_afvlm_cm_data.py --setting 2
```

### 3. 生成新的不可变实验 profile 和虚拟 Plan

当前默认 `batch_size=4`。batch size 会改变每轮本地 optimizer step 数与计划训练耗时，
因此必须生成新的 system profile 与 TrainPlan，不能复用旧的 batch-size-1 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile module_gate_freqw18_e1_bs4_ga4_r10_s42
```

如果提示同名 profile 已存在，先用下面命令查看；确认它就是本次配置时直接使用，不要覆盖。
参数有变化时换一个能说明参数的新名称。

```bash
python tools/generate_system_profiles.py --list
```

检查 `ours` 实际会选择哪个物理执行器：

```bash
python tools/select_runtime_backend.py \
  --config experiment_profiles/module_gate_freqw18_e1_bs4_ga4_r10_s42/configs/2clients/ours.yaml
```

正常应输出 `client_ddp`，FedASMU、FedCompass、MasFL/AdaMasFL、ACED、UniFed-LoRA 现在也一样。
只有 Pilot 仍输出 `client_parallel`，其动态适配器暂不强接当前固定 DDP reducer。

### 4. 在 8 卡上启动 `ours`

正常情况下不设置 `CUDA_VISIBLE_DEVICES`，程序就会使用当前可见的全部 GPU：

```bash
PYTHONUNBUFFERED=1 \
bash scripts/run_one.sh ours 2 module_gate_freqw18_e1_bs4_ga4_r10_s42 \
  2>&1 | tee ours-module-gate-2clients.log
```

建议在 `tmux` 中运行，避免 SSH 断开终止训练：

```bash
tmux new -s afvlm-ours
conda activate afvlm-cm
cd /userhome/bcx/AFVLM_CM
PYTHONUNBUFFERED=1 bash scripts/run_one.sh ours 2 module_gate_freqw18_e1_bs4_ga4_r10_s42 \
  2>&1 | tee ours-module-gate-2clients.log
```

按 `Ctrl-b` 再按 `d` 可退出但保持训练；重新进入：

```bash
tmux attach -t afvlm-ours
```

`scripts/run_one.sh` 会先打印配置路径和选中的 backend。`client_ddp` 表示 `torchrun`
已启动一个进程/GPU，所有 GPU 同时训练同一个客户端；只有 rank 0 打印进度、执行评估
并写磁盘，各 rank 会确定性重放相同的服务器更新以保持状态一致。不要改成直接运行
`python scripts/run_experiment.py`，否则不会自动创建 DDP 进程。

### 5. 查看训练状态和结果

另开终端观察 GPU：

```bash
watch -n 2 nvidia-smi
```

本次输出目录为：

```text
runs/llava/afvlm_cm/profiles/module_gate_freqw18_e1_bs4_ga4_r10_s42/2clients/ours/seed42/
```

常用文件包括 `train.log`、`events.jsonl`、`updates.jsonl`、`task_metrics.jsonl`、
`system_stats.json`、`metrics.json` 和 `checkpoints/final_trainable.pt`。当前项目没有实现
任意中间异步事件恢复；若手动中断，保留旧输出用于排查，再使用新的 profile 名称重新运行。
不要删除或覆盖已有正式实验目录。

### 6. 运行一个 baseline 或整组 baseline

同一 profile 下运行 FedAsync：

```bash
PYTHONUNBUFFERED=1 \
bash scripts/run_one.sh fedasync 2 module_gate_freqw18_e1_bs4_ga4_r10_s42 \
  2>&1 | tee fedasync-2clients.log
```

依次运行 13 个 baseline（包含FedASMU适配与OrthoFL，不包含 `ours` 和持续融合组；旧profile没有OrthoFL时须先生成新配置快照）：

```bash
bash scripts/run_baselines.sh 2 module_gate_freqw18_e1_bs4_ga4_r10_s42
```

将第二个参数改为 `5` 或 `10` 即选择 5/10 clients per task。base.yaml 无需手填
`batch_size_scope`、`reference_world_size`；生成 profile 时自动识别可见 GPU 数，
并将参考卡数固定到所有方法的快照中。有效批量是 batch × 累积 × 保存的参考卡数。
在八卡服务器生成时，默认128条记录。
8 卡 DDP 保持每卡4条、累积4次；单卡客户端自动使用4条、累积32次。
旧 profile 缺少该字段时保留原来的每卡批量语义，必须创建新 profile 才会启用公平模式。
完整配置、尾批及迁移说明见 [跨执行器批量公平性](docs/batch_fairness.md)。
无 CUDA 的电脑上生成时，可加 `--reference_world_size 8` 指定目标服务器参考卡数，
不需要改 base.yaml；程序不会在检测不到 GPU 时静默采用1卡预算。

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
| 数据分区或任一客户端的样本数 | 否 | 创建新 profile，不传 `--reuse_plans_from` |
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

此时不要传 `--reuse_plans_from`。生成器会读取服务器现有的 `data/AFVLM_CM/partitioned/`，为 2、5、10 clients/task 三档分别生成 system profile 与 TrainPlan，并保存全部 54 份解析后配置；不会改写 `plans/afvlm_cm/` 或任何旧 profile。

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

# 同一 profile 下依次运行全部 11 个可运行的主 baseline
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

本地优化器策略为 **reset per client job**：普通方法创建新的 AdamW，不跨作业保存优化器 moments。
MasFL/AdaMasFL 必须使用无 momentum/weight-decay 的 SGD，保留其控制变量递推；这两个方法
只将联邦可训练参数保持FP32，冻结7B仍使用共同配置精度。算法优化器例外不能伪装成相同AdamW。
FedAdam 等服务器优化器状态独立保存在方法状态中。

## 异步语义

运行器按常见异步联邦框架拆分为权威 Server/Method、虚拟时钟 Scheduler、Dispatcher、Client Trainer 和 Aggregator。TrainPlan 只保存开始时间、到达时间、客户端速度、网络延迟、本地工作量和可选调度组，**不保存全局模型或指定客户端必须下载哪个参数版本**。运行器在 `start_time` 读取当时的 `base_version`，再调用该方法自己的 `prepare_download`：普通方法下载全局 LoRA，Local-only 下载客户端本地状态，Pilot 下载个性化状态，UniFed-LoRA 下载任务条件化状态，MasFL/AdaMasFL 同时冻结本地控制变量。即使服务器随后更新，该作业也仍从已冻结的 dispatch package 训练。

worker 完成后先返回未经服务器处理的原始更新；父进程在计划的 `arrival_time` 调用权威方法实例的 `prepare_upload` 和 `on_arrival`。上传包含 `client_id`、`task`、`dataset`、`num_samples`、`base_version`、`arrival_time`、`local_state` 和 `delta`。服务器在到达时计算 `staleness = server_version - base_version`。FedASMU由学习式时隙选择器插入逻辑refresh，返回当时冻结的全局快照；到达时另做真实客户端训练目标反馈，不改变普通方法路径。

默认 `runtime.arrival_policy: planned` 会按持久化虚拟到达顺序应用更新，`runtime.worker_queue: plan_arrival_edf` 则只优化物理 GPU 执行顺序。逻辑 start 时先冻结 `base_version` 和方法产生的下载状态，在下一次计划 arrival 前把所有已经合法启动的作业放入候选堆，再优先执行计划 `arrival_time` 最早者；提前算完但尚未轮到的结果进入缓存。EDF 不重算下载版本、不提前聚合，也不允许未来 start 事件偷跑，因此某次运行的磁盘抖动或 GPU 波动只影响 wall-clock，不改变方法比较的逻辑到达条件。FedCompass 可以改变本地步数和分组，因为这是算法本身；其他方法不能通过修改 Plan 偷换系统条件。完整生命周期、特殊方法处理、方法服务器状态和任务耗时校准见 [`docs/async_runtime.md`](docs/async_runtime.md)。

物理执行器由 `MethodCapabilities.supports_client_ddp` 选择。普通 local-epoch 方法以及
`ours` 及除 Pilot 外的当前方法使用所有可见 GPU 对同一个客户端执行 DDP。
FedCompass 保留方法分配的 K，MasFL/AdaMasFL 保留同步后 SGD 钩子，ACED 保留真实梯度；
FedASMU 在方法内部归约反馈并广播请求决策。Pilot 的动态训练参数暂保留 worker pool，
不改变公共 DDP 包装。两种执行器都只在 Plan 指定的逻辑 arrival 聚合。
UniFed-LoRA 每 rank 复制服务器超网络，**不会按八卡分摊显存**；原显存限制仍适用。
方法级实现、旧快照批量注意事项和运行命令见 [DDP 适配说明](docs/baseline_ddp.md)。
`ours`/OPCM 的普通 AdamW DDP 训练使用公平性改造前 `ced4779` 的完整本地训练函数，
与单卡批量对齐隔离；不添加静态图或首窗口同步。FedAsync 的 `paper_loss` 正则
在已有 optimizer-boundary hook 解析加入梯度，避免给重入式 checkpoint 增加重复反向路径。
单卡对齐和其他 baseline 必需的 SGD/特殊梯度操作保留。
根因线索、公式和验证边界见[历史 DDP 与 FedAsync 修复说明](docs/ddp_training_compatibility.md)。
新 profile 自动识别并保存参考卡数；八卡服务器生成的快照固定参考预算128，
DDP 保留原每卡4条、累积4次；worker-pool 在单卡复现八个参考分片，累积32次。
两者保留相同的原始分片补齐、尾批和 micro-batch 平均损失规则。
缺少 scope 字段的旧 profile 仍是 `per_device`，其 DDP 有效批量继续乘 GPU 数。
直接运行可编辑模板时，训练入口在启动 workers 前识别卡数并保存到 resolved config；
正式对比建议使用同一份 profile，避免换机器时参考预算随可见卡数改变。
周期验证和最终测试同样由全部 DDP rank 协同执行：每个任务的样本按 rank 做不重复分片，
生成结果按原始样本顺序汇总后再计算完整语料级指标。Caption 的 CIDEr 因而仍使用整个
验证/测试语料计算，而不是错误地平均分片 CIDEr。不存在 rank 0 独自验证、其余 rank 在
NCCL broadcast 中长时间空等的情况。DDP 会先同步 rank 0 模型参数，再创建各 rank 的
联邦服务器初始快照，避免独立 LoRA 初始化造成状态分叉；`runtime.ddp_timeout_seconds`
控制长时间生成后 collective 的超时，默认 7200 秒。
采用一卡一客户端 worker pool 的方法也复用全部常驻 GPU 做数据并行评估，但不会临时建立
DDP 进程组。每张 GPU 生成六个任务各自互不重叠的样本分片，只向主进程返回 CPU 上的
样本序号、预测文本和参考文本。主进程检查无遗漏、无重复并恢复原始语料顺序，再在完整
语料上调用一次任务指标；不会平均分片 CIDEr。触发评估后会暂停新的客户端训练派发，缓存
已经完成的训练结果，所有评估分片结束后再恢复原 Plan。因此全部可见 GPU 都参与评估，且
评估不改变 TrainPlan、服务器版本、虚拟聚合顺序、optimizer 或任何方法内部状态。

### V100 八卡运行

当前可编辑配置已经设置 `runtime.backend: client_parallel`、
`runtime.executor_policy: capability`、`runtime.devices: all`、GPU 服务器聚合和
`model.dtype: fp16`。`scripts/run_one.sh` 会根据方法能力自动启动 DDP 或原 worker pool。
历史 profile 不会自动继承这些修改，因此先创建新的不可变多卡 profile：

```bash
python tools/generate_system_profiles.py \
  --profile v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42

bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42
```

每张 V100 会占用一份完整 7B FP16 模型及一个客户端的激活/优化器状态；这和原先 `device_map: auto` 把一个模型切到多卡不同。正常情况下无需设置 `CUDA_VISIBLE_DEVICES`。如果服务器还有其他任务、只想临时使用部分卡，可以选择性地设置该环境变量；程序会自动使用其中所有可见设备。

若某些任务每个 optimizer step 本来就更慢，先完成一轮运行，再执行：

```bash
python tools/estimate_task_compute_factors.py \
  runs/llava/afvlm_cm/profiles/v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42/2clients/fedasync/seed42
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
- `ca2fl`：ICLR 2024 缓存更新校准；默认 M=3，使用旧全局缓存均值加客户端增量差分，支持 MF 4/8-bit 缓存；[配置与运行](docs/ca2fl.md)。
- `aced`：活跃客户端梯度缓存均值、超时排除及重新参与；真实轨迹平均梯度的 LoRA 适配，独立记录全客户端初始化成本；[配置与运行](docs/aced.md)。
- `fedcompass`：计算能力感知的本地步数分配与分组半异步聚合。
- `fedasmu`：无服务器数据适配，动态server/local系数元梯度、LSTM/Q请求时隙与fresh global本地混合；用客户端训练目标代理原文全局F，额外成本单列。
- `orthofl`：独立客户端轨迹、全局原上传模型插值、逐层正交漂移校准后用于该客户端下一轮训练。
- `c2afcl`：固定任务 LoRA 适配；同客户端原始 delta 历史差分，其他任务历史平均 delta 的联合 SVD 子空间保护，再增量聚合；默认同任务 buffer_size=1，沿用 DDP 与原 Plan。
- `opcm_oneshot`：独立 one-shot 对照，所有客户端从共同初始模型连续训练多个 epoch，恰好上传一次，收齐后离线 OPCM 合并；不在普通 profile 或 baseline 批处理里。
- `masfl`、`adamasfl`：客户端/全局控制变量、历史下降动量；Ada 版本使用归一化局部方向与实际局部位移聚合。
- `pilot`：任务/客户端视觉适配器、CT-MoA 和任务/文本自适应聚合。
- `unifed_lora`：任务、模态、层和模块描述符驱动的服务器 LoRA 超网络。
- `ours`：Module-Gate 敏感度感知异步 LoRA 聚合；在每个真实 optimizer step 读取整个 LoRA 模块的 gate 梯度，使用绝对值时间平均、客户端内均值归一化与均匀下限，结合完整模块 `BA` 功能陈旧度、直接由客户端敏感度更新的分任务历史记忆和成功聚合窗口上的任务频率校正。聚合使用模块级共享 A/B 系数的增量式更新：`theta_server += alpha_module * (theta_client - theta_base)`；`theta_base` 是该客户端开始训练时的不可变下载快照，不是当前服务器参数。不增加训练参数，也不使用 SVD。

仅切换 `ours` 的聚合公式不需要改变 Plan；数据分区、batch、epochs 和系统参数均一致时可以复用同一 Plan，但应使用新 profile/输出目录区分实验。若同时换到 Grounding-8k 数据，则不能复用旧数据分区的 Plan。α 的计算、训练和评估流程不变；历史精度在新公式中约束增量幅度，不再解释为两个模型端点的精度加权融合。详见 [方法说明](docs/ours.md)。

### 观察 `ours` 是否正常工作

`ours` 的每次服务端到达都会在该行 `events.jsonl` 的
`result_metadata[].ours_diagnostics` 中记录完整的版本化诊断信息。记录只由 `ours`
生成，不改变其他方法的日志、训练状态或聚合路径。主要字段包括：

- `sensitivity.module_by_module`：每个 LoRA module 的最终敏感度；
- `sensitivity.transform`：绝对值统计方式、原始/最终分布、均值归一化、均匀混合系数和下限；
- `sensitivity.transform.signed_gate_mean`、`gate_direction`、`gate_direction_abs_mean`：仅用于分析的有符号 Gate 方向；
- `sensitivity.observations_by_module`、分布摘要和 top module；
- `functional_staleness`：版本陈旧度、server drift、local update、相对功能陈旧度、可靠度函数及 gamma；
- `aggregation.module_alpha_by_module`：每个模块实际使用的共享 A/B 聚合系数；
- `aggregation.rule`：新版本记录为 `base_relative_delta`，导出 CSV 时对应 `aggregation_rule`；旧日志缺少标记时留空，不改写旧实验含义；
- `frequency`：聚合前窗口计数、本次任务权重，以及成功聚合后的窗口计数；
- `memory`：当前任务的原始/偏差修正记忆、接受次数和跨任务历史精度在聚合前后的值；
- rejected update 的拒绝原因和未变化的 memory。

训练过程中可以直接查看最后一条记录：

```bash
tail -n 1 runs/llava/afvlm_cm/profiles/<profile>/2clients/ours/seed42/events.jsonl
```

训练中途或结束后可将嵌套记录导出为便于画图的 update/module CSV：

```bash
python tools/export_ours_diagnostics.py \
  runs/llava/afvlm_cm/profiles/<profile>/2clients/ours/seed42
```

默认输出到运行目录下的 `ours_diagnostics/`：
`ours_update_diagnostics.csv` 和 `ours_module_diagnostics.csv`。Module-Gate 不产生
rank 级 CSV。CSV 是派生分析文件，可以随时重新生成；权威原始记录始终是
`events.jsonl`。

FCIT 和 FedSpace 原本属于联邦持续/任务增量学习，目前没有作为 baseline 加入。
C2-AFCL 原本也研究持续学习，但已按要求加入明确标注的固定任务 LoRA 适配版，
并未改变本项目任务场景。FedAST 的原问题是并行训练多个联邦模型，不作为当前主 baseline。

## 运行单个方法

接口为 `bash scripts/run_one.sh <method> <setting> [profile]`，其中 setting 为 2、5 或 10。省略 profile 时使用原始兼容配置；提供 profile 时使用对应的不可变配置快照。以下是 2 clients/task 的常规 baseline 及提出方法命令；独立 one-shot 使用后文专属入口：

```bash
bash scripts/run_one.sh local 2
bash scripts/run_one.sh fedavg 2
bash scripts/run_one.sh fedprox 2
bash scripts/run_one.sh fedadam 2
bash scripts/run_one.sh fedasync 2
bash scripts/run_one.sh fedbuff 2
bash scripts/run_one.sh ca2fl 2
bash scripts/run_one.sh fedcompass 2
bash scripts/run_one.sh fedasmu 2
bash scripts/run_one.sh orthofl 2
bash scripts/run_one.sh c2afcl 2
bash scripts/run_one.sh masfl 2
bash scripts/run_one.sh adamasfl 2
bash scripts/run_one.sh pilot 2
bash scripts/run_one.sh unifed_lora 2
bash scripts/run_one.sh ours 2
```

OrthoFL/FedASMU实现、OPCM恢复 `.5` 的说明及新profile运行流程见
[专门操作说明](docs/orthofl_fedasmu.md)。FedASMU不预留服务器训练数据，但客户端目标代理
不等于原文全局目标；不能声称原始理论或精度被完全复现。Pilot/UniFed-LoRA已有核心功能
与全局/个性化双评估，但新实现需新配置快照，UniFed完整输出头还受显存限制。

### ACED：活跃梯度缓存异步聚合

参数仅修改 `configs/methods/aced.yaml`。默认 `delay_threshold: 10`、`cache_bits: 32`；
`server_lr: null` 时采用 `0.2 * sqrt(N / (Plan上传数+1))`，也可显式指定服务器梯度步长。
本地仍用公共 AdamW/epochs 设置，缓存实际训练梯度均值而非参数差分。

```bash
python tools/generate_system_profiles.py \
  --profile aced_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from e1_bs4_ga4_r10_s42

bash scripts/run_one.sh aced 2 aced_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh aced 5 aced_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh aced 10 aced_e1_bs4_ga4_r10_s42
```

旧 profile 名请替换成服务器真实名称，数据/训练/系统条件匹配才复用 Plan。
新 profile 自动包含 ACED，旧 profile 不改写；新配置来自当前模板，不保证其他方法参数
与旧 profile 相同。默认批处理脚本列表不变，ACED 单独运行。初始化先额外计算每客户端
一个有效批量的梯度并进行一次全局下降，单独记入 `algorithm_initialization.json`；
不把它或缓存复用冒充新的 Plan 上传。普通评估仍按首次纳入聚合的上传预算触发。
本方法现默认使用全部可见 GPU 的 DDP；初始化按真实 rank 分片，参考八卡有效批量仍为 128。
不改变已有 ours/OPCM 的八卡路径；旧缺少参考元数据的 per_device 快照应重新建 profile。
原文差异、活动年龄与真实陈旧度区别、缓存显存和日志见 [完整说明](docs/aced.md)。

### CA²FL：缓存更新校准

先修改 `configs/methods/ca2fl.yaml`，默认参数为：

```yaml
server_lr: 1.0
buffer_size: 3
cache_bits: 32
cache_seed: 42
flush_last_buffer: true
```

`buffer_size=3` 采用原文 BERT-LoRA 语言实验设置；`server_lr=1` 在原文搜索范围内，
不是已验证的 VLM 最优值。`cache_bits=32` 为标准 CA²FL；改成 8 或 4 启用附录的
MF 缓存量化，量化器的具体选择见专门说明。原有本地训练参数统一继承，不改成原文 rank1。

旧 profile 不会自动增加新方法。在服务器上，若数据与本地训练/系统设置不变，可复用旧 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile ca2fl_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from e1_bs4_ga4_r10_s42

bash scripts/run_one.sh ca2fl 2 ca2fl_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ca2fl 5 ca2fl_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ca2fl 10 ca2fl_e1_bs4_ga4_r10_s42
```

请将旧 profile 名替换为服务器真实名称。新 profile 从当前模板生成配置，不改写旧实验；
复用 Plan 并不代表其他方法配置自动保持旧版本，运行前检查新配置是否符合预期。
能力自动选择全部可见 GPU 的 DDP；无需改 `run_one.sh`。默认 baseline 批处理列表保持
原样，CA²FL 单独启动。评估仍按纳入聚合的客户端更新预算执行，不把缓存复用算作新上传。
算法顺序、重复客户端缓冲适配、残余 flush、日志与显存开销见 [CA²FL 详细说明](docs/ca2fl.md)。

### C2-AFCL：固定任务适配

修改参数使用 `configs/methods/c2afcl.yaml`。默认 `server_lr: 0.5`、
`subspace_rank: 5`、`buffer_size: 1` 是本项目适配默认值，不冒充原文公布的数值。
新方法只增加服务器校准，不改客户端训练与评估。通常自动选择全部可见 GPU 的
`client_ddp`，沿用现有有效 batch 与本地训练代码。

旧 profile 没有新增方法的 YAML，需新建配置快照。若数据、训练工作量与系统设置均一致，
可以复用原 Plan，无需重新随机生成系统条件；把下面的 `YOUR_OLD_PROFILE` 替换为你真实的旧快照名：

```bash
python tools/generate_system_profiles.py \
  --profile c2afcl_fixed_task_v1 \
  --reuse_plans_from YOUR_OLD_PROFILE

bash scripts/run_one.sh c2afcl 2 c2afcl_fixed_task_v1
bash scripts/run_one.sh c2afcl 5 c2afcl_fixed_task_v1
bash scripts/run_one.sh c2afcl 10 c2afcl_fixed_task_v1
```

生成器读取当前所有方法模板，而非复制旧方法配置，正式比较前需核对与已有实验共享设置一致。
新增方法已加入 `run_baselines.sh`；使用缺少它的旧 profile 跑批量脚本会在训练前提示缺少配置。
详见 [原文公式、适配边界、诊断日志与显存成本](docs/c2afcl.md)。

`ours` 现在只实现本文的 Module-Gate 版本。推荐为正式实验生成独立的不可变 profile：

```bash
python tools/generate_system_profiles.py \
  --profile module_gate_freqw18_e1_bs4_ga4_r10_s42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/run_one.sh ours 2 module_gate_freqw18_e1_bs4_ga4_r10_s42
```

完整 Module-Gate 统计、功能距离、聚合公式和限制见 `docs/ours.md`。

使用前文创建的正式 V100 八卡快照：

```bash
bash scripts/run_one.sh fedavg 2 v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh fedasync 5 v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42
```

profile 结果写入独立目录，例如：

```text
runs/llava/afvlm_cm/profiles/v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42/2clients/fedavg/seed42/
```

将最后一个参数改为 5 或 10 即选择对应数据集；例如：

```bash
bash scripts/run_one.sh fedcompass 5
bash scripts/run_one.sh adamasfl 10
bash scripts/run_one.sh pilot 5
bash scripts/run_one.sh unifed_lora 10
bash scripts/run_one.sh ours 2
```

直接调用 Python 不会自动派生 torchrun 进程；需要能力选择和 DDP 时应使用
`scripts/run_one.sh`。下面的命令只适合显式的单进程或固定 backend 调试：

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
bash scripts/run_baselines.sh 2 v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42
```

### 命令行训练进度

训练进度条默认开启，直接使用原有命令即可，不需要重新生成 system profile 或 TrainPlan：

```bash
bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42
```

命令行会依次显示：

- 元数据扫描完成以及所有可见 GPU worker 分别就绪的提示；
- `fedasync virtual arrivals`：当前已应用的虚拟到达数 / TrainPlan 总更新数，并显示服务器版本、当前客户端和 staleness；
- `GPU<n> <client_id>`：每张卡当前客户端的真实 optimizer step / 计划 optimizer step，并动态显示 loss；
- DDP 方法只由 rank 0 显示 `local <client_id>` 进度；其余 rank 同步训练同一客户端，不重复打印；
- DDP 评估显示 `evaluate <task> rank 1/N shard`，它是 rank 0 的本地分片进度；独立的
  `DDP evaluation start/complete` 状态行记录该任务的完整样本数、rank 数和全局完成时刻；
- `evaluate <task>`：worker-pool 方法定期 validation 和最终 test 时的全局累计样本数。
  主进程汇总全部 worker 的进度，GPU worker 不直接写进度条。

所有更新条都设置为逐更新刷新；进入周期评估前会再次强制刷新，并输出包含真实
`server_version` 和 `interval` 的独立状态行。本地进度按 optimizer step 计数，不按
gradient accumulation 的 micro-batch 计数。参考八卡模式下 DDP 完整窗口每卡累积4次，
单卡客户端完整窗口累积32次；尾窗口按实际 micro-batch 数处理。
进度条增加 1 始终代表 1 次参数更新。进度显示只读取
已有训练状态，不会改变 local epoch、TrainPlan、聚合顺序或虚拟时间。

公平模式下 `batch_size` 是每卡 micro-batch 大小，
`batch_size × gradient_accumulation × reference_world_size` 是共同完整步骤记录预算。
真实物理 micro-batch 内仍动态 padding 文本、堆叠图像，并调用一次 LLaVA forward/backward。
CE 沿用原 micro-batch 均值再按窗口平均，DDP 沿用原 rank 梯度平均。
尾批和补齐重复样本与原八卡完全一致，不引入零权重占位或新的 token 加权目标。
`updates.jsonl` 的 `batching` 和 `system_stats.json` 记录实际换算和每步记录数。
启用单卡匹配应新建 profile；其他参数相同的旧八卡128结果可保留，旧单卡16结果需重跑。

若需要把终端输出重定向到文件，建议同时打开 Python 非缓冲输出：

```bash
PYTHONUNBUFFERED=1 \
  bash scripts/run_one.sh fedasync 2 v100_fp16_8gpu_edf_e1_bs4_ga4_r10_s42 \
  2>&1 | tee fedasync-2clients.log
```

如需关闭进度条，在准备创建新 profile 的基础配置中设置：

```yaml
output:
  progress_bar: false
```

已经保存的不可变 profile 不需要补写该字段：缺省值就是 `true`。若要永久改变某个旧实验的显示选项，应创建新的配置 profile，而不是修改旧快照。

## 评估与输出

项目以**已真正纳入当前模型的客户端更新数**作为所有方法统一的周期评估预算。默认
`evaluation.interval_unit: incorporated_client_updates` 和
`evaluation.eval_every_client_updates: 10`。FedAsync 与 `ours` 每次有效聚合纳入一个客户端更新，
所以仍然每 10 次有效聚合评估；FedBuff 只有在 buffer flush 后才按该 buffer 的实际更新数推进，
FedCompass/同步方法则按本次分组或轮次包含的客户端数推进。若一次聚合跨过阈值，评估聚合后的
当前全局模型，并同时记录目标阈值和实际计数，不会对同一个模型重复评估。Local-only 按已经
写入各客户端本地状态的完成更新计数。评估过程不计入虚拟训练时间，也不改变训练状态。

每条 `events.jsonl` 会记录 `accepted_update_count_after` 和
`evaluation_update_count_after`；周期记录会在 `task_metrics.jsonl` 中额外保存
`crossed_update_thresholds`、`evaluation_progress_count`、`accepted_client_updates` 和触发位置。
全部 TrainPlan 到达事件完成后，先执行方法的结尾处理；FedBuff/MasFL 等方法在结尾纳入的残余
更新也会推进评估预算，随后再用最终模型在公共 test 集上评估。

旧不可变 profile 中的 `interval_unit: server_updates` 会作为兼容别名，沿用其数值间隔但自动迁移到
新的客户端更新预算，因此现有 profile 拉取新代码后也能使用公平评估。正式论文实验仍建议从更新后的
基础配置创建一个新 profile，使 `resolved_config.yaml` 明确写出新语义；可以使用
`--reuse_plans_from` 原样复用旧 system profile 和 TrainPlan。

所有存在服务器聚合的算法都以 `server_global` 为主评估结果。Pilot 仍使用任务路由，但使用当前服务器状态及同任务服务器端客户端视觉适配器的均值，不使用个性化 LoRA 替代全局主结果。Local-only 没有全局模型，因此是唯一例外：分别用每个客户端本地模型测试其所属任务的同一公共测试集，再在任务内等权平均，输出协议标记为 `client_local_mean`。

所有 baseline 与 `ours` 固定调用同一个 FCIT-compatible evaluator，不允许方法覆盖任务指标：

- ImageNet-R、AOKVQA：去除首尾空白后忽略大小写的 exact match，输出百分比；
- DVQA、FigureQA：使用去除首尾空白、忽略大小写的完整答案 exact match，空白预测始终计错，输出百分比。该规则修正原 FCIT 子串匹配会将空答案和部分答案误计为正确的问题；
- Flickr30k：对每图五个参考 caption 调用 `pycocoevalcap`，报告 BLEU-1..4、METEOR、ROUGE-L、CIDEr 以及七项的 `FCIT_caption_average`，均为原始 scorer 值乘 100；
- Grounding：每个 query 单独解码，严格以 `IoU > 0.5` 计正确并输出百分比，同时额外保留 0--1 范围的 `mean_IoU`。

DVQA、FigureQA 的旧子串匹配结果与新 exact-match 结果不能直接混用；论文比较时所有方法必须使用同一新规则重新评估。此修复只改变判分，不改变训练、数据分区或 TrainPlan，已有最终检查点可复评。

默认 `generation_max_new_tokens: 128` 与 FCIT 推理长度一致，`generate()` 使用确定性 greedy decoding。任务间指标量纲不同，因此不生成没有意义的六任务 raw average。

训练完成后可独立复评：

```bash
python scripts/evaluate.py \
  --config configs/experiments/llava/afvlm_cm/2clients/fedasync.yaml \
  --checkpoint runs/llava/afvlm_cm/2clients/fedasync/seed42/checkpoints/final_trainable.pt \
  --output runs/llava/afvlm_cm/2clients/fedasync/seed42/metrics.reproduced.json
```

每个运行目录包含 `resolved_config.yaml`、`system_profile.reference.json`、`train_plan.json`、`events.jsonl`、`updates.jsonl`、`task_metrics.jsonl`、`metrics.json`、`system_stats.json`、`train.log` 与 `checkpoints/final_trainable.pt`。评估文件显式记录 `protocol`、`split`、`server_version`、`virtual_time` 和 `per_task`。并行更新日志还记录 worker/GPU 与真实开始、完成时间。检查点只保存联邦可训练状态、服务器/方法/调度器/随机状态及必要 metadata，不保存冻结的 7B 主干。

系统统计包括 mean/median/max staleness、总/接受更新数、聚合次数、客户端/任务更新分布、虚拟训练时间、真实 wall-clock 时间和 GPU worker 数；FedBuff 额外报告缓冲聚合次数、平均占用和平均等待时间；FedCompass 额外报告本地步数分配、分组完成时间与组内完成跨度。不同任务的量纲不兼容，因此不会把 accuracy、CIDEr 和 IoU 粗暴平均为一个原始分数。

数据接口和每种 baseline 的组件/方程/适配边界另见 `docs/AFVLM_CM.md` 与 `docs/baselines.md`。提出方法的 Module-Gate 公式、配置和运行流程见 `docs/ours.md`。

## 持续模型融合基线（新增）

新增三个独立的服务器端异步持续融合适配：`opcm_lora`（NeurIPS 2025，奇异方向投影与范数控制）、`dop_lora`（NeurIPS 2025，双投影/MGDA）、`nufilt_lora`（ICLR 2026，零空间过滤与投影感知残差适配）。它们不是普通 FedAvg，也不是原论文独立专家/持续学习场景的精确复现。保留固定任务客户端；按原 Plan 聚合，使用不可变下载快照提取 `B_local A_local - B_base A_base`，融合后压回原 LoRA rank。冻结主干不变，不增加客户端可训练参数或校准数据。

完整公式、选择依据、上游版本、压缩误差日志及适配限制见 [持续融合说明](docs/continual_merging.md)。三个方法的默认能力策略都选择单客户端 DDP；支持共同 server_global 六任务评估。原 `run_baselines.sh` 的方法顺序不改，持续融合组独立执行。

先确认 `configs/base.yaml` 的实际参数及本地数据，再创建新快照（名称不会设置参数）：

```bash
python tools/generate_system_profiles.py --profile merging_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh opcm_lora 2 merging_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh dop_lora 2 merging_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh nufilt_lora 2 merging_e1_bs4_ga4_r10_s42
# 或顺序跑这三个方法；将2替换为5或10可切换客户端数
bash scripts/run_merging_baselines.sh 2 merging_e1_bs4_ga4_r10_s42
```

旧 immutable profile 不会被改写，也不自动具有新增方法配置。新增方法本身不改变Plan，可创建新profile并用 `--reuse_plans_from <旧快照名>` 复用兼容Plan；这仍使用当前配置模板，需自行核对数据、seed、模型和训练设置一致。更改客户端数据量后不能复用旧Plan。同一比较的旧基线与 `ours` 应使用同一快照和实际有效batch。此处profile名称是人工标签，方法支持服务器的实际分区规模，不会修改或同步本地数据。

## 独立 one-shot 联邦 OPCM 对比

`opcm_oneshot` 与原 `opcm_lora` 完全不同的通信协议：每个客户端从相同version0
连续训练多个epoch，全部客户端各上传一次之后才离线合并，最后只评估同一个最终全局
模型。服务器在收集期间不更新参数，也不向客户端发送其他客户端的知识。
客户端任务、数据、LoRA、DDP 和六任务评估代码保持原样；OPCM仅处理LoRA状态。

参数源是 `configs/methods/opcm_oneshot.yaml`。默认本地10epoch、通信1轮；
`merge_weight: 1.0` 是独立实验的项目设置，不改变原 `opcm_lora` 的0.5。
默认 `merge_order: seeded_random`、`merge_seed: 42`，避免把客户端速度顺序混进融合效果。
按不同merge_seed做顺序对照时，须新建快照，不能按测试结果挑顺序。

在服务器根目录创建独立配置快照并运行：

```bash
python tools/generate_oneshot_profiles.py \
  --profile opcm_oneshot_e10_bs4_ga4_s42 \
  --local_epochs 10

bash scripts/run_oneshot.sh 2 opcm_oneshot_e10_bs4_ga4_s42
# 按需要选择5或10 clients/task；每条命令是一个独立实验。
bash scripts/run_oneshot.sh 5 opcm_oneshot_e10_bs4_ga4_s42
bash scripts/run_oneshot.sh 10 opcm_oneshot_e10_bs4_ga4_s42
```

工具默认识别所有可见GPU；DDP沿用现有每卡batch与梯度累积。只调整本实验epoch
可用 `--local_epochs 5`，无需改 `base.yaml`。离线生成可加 `--reference_world_size 8`。
快照放在 `one_shot_profiles/<name>/`，结果放在
`runs/llava/afvlm_cm/oneshot/profiles/<name>/<setting>clients/opcm_oneshot/seed42/`。
profile名称只是标签，bs4/ga4应与实际base配置一致；同名快照和非空结果目录不覆盖。

不能复用多轮异步TrainPlan，但可以只复用相同数据的基础系统画像，生成新的一次上传Plan：

```bash
python tools/generate_oneshot_profiles.py \
  --profile opcm_oneshot_e10_same_system \
  --local_epochs 10 \
  --system_profile_from YOUR_OLD_PROFILE
```

将 `YOUR_OLD_PROFILE` 替换为真实旧快照名。工具会核对客户端ID、任务、样本数、
任务计算成本和种子；不匹配就拒绝复用。现有 `experiment_profiles`、旧Plan和方法
配置不会被改写；当前普通生成器生成20个方法/60份配置（含CA2FL和ACED），不包含此one-shot实验。

上传数量为12/30/60；对外只有1次全局聚合，离线OPCM内部专家处理次数另记。
日志记录 `merge_client_order`、`merge_steps`、`expert_training` 和原OPCM模块诊断。
one-shot10epoch与10轮×1epoch只是相近样本遍历预算，优化器连续性、通信量和模型
交换轨迹不同；论文中必须区分，不能宣称训练轨迹相同。

完整协议、配置路径、评估规则、缓存显存成本和LoRA适配限制见
[one-shot OPCM说明](docs/opcm_oneshot.md)。当前没有实际启动GPU训练，也不保证原论文精度。

## 静态工程检查

以下命令只检查 Python 语法/导入、YAML 解析与继承、注册表、63 个配置（60常规+3个one-shot）、路径以及原始分区完整性；不会加载 LLaVA、下载权重或启动训练：

```bash
python -m compileall -q code scripts tools
python tools/validate_repository.py
python tools/validate_afvlm_cm_data.py --setting 2  # 可选的全量数据/图片检查
ruff check code scripts tools
```

## 与论文原始设置的差异

- FedCompass：保留计算能力感知步数分配和分组半异步核心，复用本项目的持久化虚拟时钟，没有复制原框架进程架构；优化状态限定为 LoRA。
- FedASMU：保留动态server/local系数及LSTM/Q请求机制，以客户端私有训练目标替代全局F；不预留服务器数据，不用validation/test。请求epoch改为optimizer时隙，FP32 LoRA/共同AdamW适配；额外计算及通信必须计成本，不宣称完全原版。
- OrthoFL：全局聚合与客户端校准分离；在同层LoRA坐标投影而非全冻结主干，主评估保持公共全局模型。
- C2-AFCL：缓存原始客户端增量差分、全联邦可训练坐标 SVD 投影，之前顺序任务改为其他固定任务的已聚合历史；默认逐到达增量更新，可选同任务缓冲。原文未具体定义的持续学习正则项不自行补造，客户端仍用共同指令 CE/AdamW；详见专门说明。
- MasFL/AdaMasFL：真实初始梯度控制、SGD、双层动量及Ada归一化；实际eta*K归一位移。到达缓冲与可调参数是Plan适配，不直接宣称论文tuning-free理论设置。
- Pilot：分阶段任务/私有预训练MLP、token CT-MoA、任务及邻近text聚合；私有状态不被平均。stage1_rounds=1后转stage2以匹配预算，差异特征默认归一；原生个性化为附加结果。
- UniFed-LoRA：同构主干下以任务/模态/层/模块描述符生成真实A/B，客户端delta提供VJP，不是标量门控。full-participation meta-batch、共同初始化及全局部署视图是显式适配；完整输出头可能超出32GB显存。

## 当前限制

- 代码未替用户启动 7B 训练；显存、CUDA、原始 LLaVA 版本与权重兼容性需要在目标训练机确认。
- Flickr30k、DVQA、FigureQA 和 COCO 受各自许可与分发条件约束；下载工具不会绕过许可页面。
- 当前训练入口保存可独立复评的最终联邦检查点，但尚未提供从任意中间异步事件恢复并继续训练的 CLI；这需要同时恢复仍在途的客户端作业快照。
- Flickr30k训练每条一个caption，当前验证/测试保留五参考并使用pycocoevalcap；METEOR需Java。
- 详细算法及参数见[baseline审查](docs/baseline_audit.md)；[独立复审](docs/baseline_reaudit_20261011.md)指出 Pilot 半精度优化风险及 UniFed 默认输出头显存限制尚未解决，不应理解为全部配置已真实训练通过。
- 本次上传对 `ours`/OPCM 与已有八卡实验的影响、是否重建 profile/Plan，见[旧实验兼容说明](docs/profile_compatibility_20261011.md)。新 profile 读取当前模板，复用 Plan 不会复制旧方法参数。
- `ours` 已实现，但其 sensitivity 是虚拟 Module-Gate 的局部一阶变化幅度，即各 optimizer step 的 `|dL/dz_l|` 平均，并非完整 Fisher、Hessian 或真实任务泛化重要性；当前只支持 LoRA-only 联邦状态。任务频率校正基于最近成功聚合标签，而不是对真实设备速度的直接估计。`ours` 不在 baseline 批处理里，应使用独立命令运行。

## 主要论文与数据来源

- FedCompass, ICLR 2024: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/a9f3457fa97f106f1756885237787789-Abstract-Conference.html>
- FedASMU, AAAI 2024: <https://ojs.aaai.org/index.php/AAAI/article/view/29297>
- C2-AFCL, ICML 2026: <https://proceedings.mlr.press/v306/li26gx.html>
- MasFL/AdaMasFL, ICML 2025: <https://proceedings.mlr.press/v267/yan25f.html>
- Pilot, AAAI 2025: <https://ojs.aaai.org/index.php/AAAI/article/view/35476>
- UniFed-LoRA, CVPR Workshops 2026: <https://openaccess.thecvf.com/content/CVPR2026W/FedVision/html/Milasheuski_UniFed-LoRA_Exploiting_Semantic_Task_Correlation_for_Heterogeneous_Multimodal_Federated_Fine-Tuning_CVPRW_2026_paper.html>
- DVQA 官方数据仓库: <https://github.com/kushalkafle/DVQA_dataset>
- FigureQA 官方项目: <https://www.microsoft.com/en-us/research/project/figureqa-dataset/>
