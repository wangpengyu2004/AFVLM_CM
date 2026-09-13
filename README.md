# AFVLM-CM

AFVLM-CM 是一个面向**异步联邦视觉语言模型微调**的研究框架。研究场景是：每个客户端长期固定在一个任务上，不发生持续学习式任务切换；不同任务的训练耗时不同，因此更新到达顺序会与任务相关，并同时产生陈旧性和跨任务冲突。

本仓库实现的是 AFVLM-CM 自研方法及其对照实验。FCIT 仅是当前指令数据的来源，项目不实现 FCIT 的持续学习算法、专家路由或任务序列。

## 方法概览

AFVLM-CM 包含两个轻量且可消融的机制：

1. **任务条件下载修正**：服务端为每个任务维护已接受更新的指数移动平均方向。客户端下载时，从最新全局 LoRA 状态出发，加上本任务的小幅记忆修正，缓解统一模型对高频/快速任务的偏置。
2. **陈旧冲突上传修正**：更新到达服务端后，计算客户端训练期间的全局漂移。若更新方向与漂移方向冲突，只删除冲突投影分量，再按陈旧度做多项式衰减；不改变无冲突更新。

对应实现位于 `code/afl_vlm/methods/custom/afvlm_cm.py`，方法 ID 为 `afvlm_cm`。关键参数都在 `configs/run.yaml`：

- `download_strength`：任务记忆在下载模型中的修正强度；
- `memory_momentum`：每任务更新记忆的 EMA 动量；
- `staleness_exponent`：陈旧权重 `(staleness + 1)^(-exponent)` 的指数；
- `server_lr`：服务端应用修正更新的步长；
- `download_correction` / `upload_correction`：两项机制的独立消融开关。

框架记录原始/修正更新范数、冲突余弦、被删除的投影量、陈旧度与实际权重，便于直接验证机制是否发挥作用。

## 安装

推荐 Linux、CUDA GPU 和 Python 3.10+。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,qwen,plots]"
```

Windows PowerShell 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

默认主干是 `Qwen/Qwen2.5-VL-7B-Instruct`，使用语言侧 LoRA/QLoRA，只在联邦通信中传输可训练参数。仓库也提供可选的 `llava15` 适配器；使用它时安装 `pip install -e ".[dev,llava,plots]"`，并在配置中把 `model.adapter` 和 `model.hf_id` 改为 `llava15` 与 `llava-hf/llava-1.5-7b-hf`。

## 数据集

AFVLM-CM 固定任务 benchmark 从本地 FCIT 指令标注派生，但会先按任务汇总，再重新划分客户端。每个客户端只拥有一个固定任务，且训练、probe 和 final 集互不重叠。

| 配置值 | 总客户端 | 每任务客户端 | 数据量设置 |
|---|---:|---:|---|
| `small_balanced` | 16 | 2 | 同任务客户端等量 |
| `small_quantity_skew` | 16 | 2 | 同任务客户端不等量 |
| `medium_balanced` | 40 | 5 | 同任务客户端等量 |
| `medium_quantity_skew` | 40 | 5 | 同任务客户端不等量 |
| `large_balanced` | 80 | 10 | 同任务客户端等量 |
| `large_quantity_skew` | 80 | 10 | 同任务客户端不等量 |

在 `configs/run.yaml` 中只改 `dataset.variant` 即可切换。运行时会自动读取该变体的任务、客户端归属、训练文件和任务延迟，不会二次随机划分。

### 图片放置位置

所有图片统一放在 `data/fcit/dataset/` 下，并保持指令中的相对路径：

| 任务 | 应存在的目录/路径示例 |
|---|---|
| ImageNet-R | `data/fcit/dataset/ImageNet-R/train/...` |
| ArxivQA | `data/fcit/dataset/ArxivQA/images/...` |
| IconQA | `data/fcit/dataset/IconQA/iconqa_data/iconqa/...` |
| CLEVR | `data/fcit/dataset/CLEVR/images/...` |
| OCR-VQA | `data/fcit/dataset/OCR-VQA/images/...` |
| Flickr30k | `data/fcit/dataset/Flickr30k/train/...` |
| FigureQA | `data/fcit/dataset/FigureQA/images/...` |
| super-CLEVR | `data/fcit/dataset/super-CLEVR/images/...` |

不要把图片平铺到同一目录。正式训练设置 `require_images: true`，缺少任意被引用图片都会在加载数据时明确失败。

重新构建或检查指令数据：

```bash
python -m scripts.prepare_fixed_task_benchmark --force
python -m scripts.prepare_fixed_task_benchmark --validate-only
python -m scripts.preflight_benchmark_images
```

若图片还未全部下载，但想先生成缺失清单：

```bash
python -m scripts.preflight_benchmark_images --allow-missing
```

检查结果写入 `data/fcit/fixed_task_benchmark/image_preflight.json`。

## 如何运行

所有正式实验统一使用 `configs/run.yaml`。先验证数据、配置和展开后的调度，不加载模型：

```bash
python -m scripts.prepare_fixed_task_benchmark --validate-only
python -m scripts.preflight_benchmark_images
python -m scripts.validate_config --config configs/run.yaml
python -m scripts.run_experiment --config configs/run.yaml --methods afvlm_cm --dry-run
```

### 运行自研方法

```bash
python -m scripts.run_experiment \
  --config configs/run.yaml \
  --methods afvlm_cm \
  --output-root runs/afvlm_cm/ours
```

### 运行单个 baseline

以 FedAsync 为例：

```bash
python -m scripts.run_experiment \
  --config configs/run.yaml \
  --methods fedasync \
  --output-root runs/afvlm_cm/baselines/fedasync
```

把 `fedasync` 换成下表任一方法 ID 即可。`--methods` 接收的是 `configs/run.yaml` 中的 ID，不需要修改多个 `enabled` 字段。

### 一次运行全部 baseline

```bash
python -m scripts.run_experiment \
  --config configs/run.yaml \
  --methods async_sgd async_decay fedavg fedasync fedbuff fedadam fedyogi fedcompass_sim \
  --output-root runs/afvlm_cm/baselines/all
```

### 自研方法与全部 baseline 使用相同调度运行

```bash
python -m scripts.run_experiment \
  --config configs/run.yaml \
  --methods afvlm_cm async_sgd async_decay fedavg fedasync fedbuff fedadam fedyogi fedcompass_sim \
  --output-root runs/afvlm_cm/full_comparison
```

已有输出默认不覆盖；确认要重跑同一路径时显式加 `--overwrite`。多随机种子实验在配置中设置 `run.seeds: [42, 43, 44]`。两个延迟场景会在同一份任务/客户端数据上运行：`task_correlated` 保留任务相关速度，`task_shuffled` 打乱客户端速度映射，作为到达偏置控制组。

这里的异步协议与 FLGo/FedAsync 属于同一类“客户端基于旧版本训练、服务端按到达顺序更新”的范式，但并非逐行复现 FLGo。AFVLM-CM 采用确定性事件流和每客户端等上传配额，因为它能在比较任务相关延迟时固定各客户端参与量；`task_shuffled` 则专门检验收益是否只是由速度分布造成。

汇总结果：

```bash
python -m scripts.summarize_results runs/afvlm_cm/full_comparison
```

每个运行目录包含解析后的配置、确定性调度、事件/更新/probe 日志、最终指标、LoRA 状态和方法状态。不同方法、种子和延迟场景均从相同初始化独立开始。

## Baseline

仓库提供 8 个可运行对照：

| 方法 ID | 类型 | 作用 |
|---|---|---|
| `async_sgd` | 经典异步加法 | 无陈旧补偿的下界 |
| `async_decay` | 陈旧衰减 | 隔离单纯 staleness weighting 的收益 |
| `fedavg` | 同步 FedAvg | 无异步到达偏置的参照 |
| `fedasync` | FedAsync | 经典异步模型插值 |
| `fedbuff` | FedBuff | 缓冲式异步聚合 |
| `fedadam` | FedAdam | 自适应服务端优化 |
| `fedyogi` | FedYogi | 更稳健的自适应服务端优化 |
| `fedcompass_sim` | FedCompass 风格调度 | 用客户端步数分配缓解系统异构 |

方法选择依据和论文链接见 `claudedocs/research_2026-09-13_fixed_task_async_baselines.md`。

## 项目结构

```text
code/afl_vlm/
  models/                 # Qwen2.5-VL、LLaVA-1.5、测试模型与状态代数
  data/                   # 固定任务数据预设、任务适配和图片解析
  scheduling/             # 确定性虚拟时钟、延迟模型和调度
  federation/             # 客户端、版本化服务端和状态存储
  methods/baselines/      # 8 个 baseline
  methods/custom/         # AFVLM-CM 自研方法
  experiments/            # 端到端与机制实验协议
  analysis/               # 指标与结果汇总
configs/run.yaml          # 唯一正式实验配置
scripts/                  # 数据构建、图片检查、运行与汇总入口
tests/                    # 单元与流水线测试（不作为研究实验）
```

## 引用与来源

- 数据标注来源：[MLLM-CL/FCIT](https://huggingface.co/datasets/MLLM-CL/FCIT)
- FCIT 原论文与代码：[ICCV 2025 paper](https://www.openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html)、[official repository](https://github.com/Ghy0501/FCIT)

使用派生数据时请同时遵守原始数据集及其组成数据集的许可证和引用要求。
