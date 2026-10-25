# 独立实验：客户端的 LoRA rank—验证性能曲线

本实验观察所选任务客户端在相同训练预算下，对 LoRA 容量的响应是否不同。
每次八张 GPU 通过 DDP 共同训练同一个客户端；不上传、不聚合、不使用异步模拟。
没有注册新的联邦方法，也不读取 `ours`、其他 baseline 或 experiment profile 的配置。

## 实验矩阵与固定训练预算

读取现有 `data/AFVLM_CM/partitioned/2_clients`，不重新划分数据：

| 任务 | 数据集 | client_0 / client_1 训练记录 | 主验证指标 | 每客户端训练轮数 |
|---|---|---:|---|---:|
| cls | ImageNet-R | 7006 / 5994 | accuracy (%) | 5 |
| caption | Flickr30k | 7796 / 7204 | CIDEr，另保存全部原评估指标 | 5 |
| vqa | AOKVQA | 5435 / 5565 | 当前统一评估器的 VQA_accuracy (%) | 5 |

六个客户端分别训练 rank 2、4、8，默认 seed 42，共 **18 个独立训练任务**。
执行顺序为任务 `cls → caption → vqa`，每任务 `client_0 → client_1`，
每客户端 `rank 2 → 4 → 8`。每个任务启用新的八进程组，从本地基础权重重新加载，
重新初始化 LoRA 和 optimizer；不会继承前一个客户端或 rank 的参数。
这会产生 18 次模型加载成本，换取各次训练的进程和状态隔离。

唯一配置为 `configs/probes/local_rank_probe.yaml`，**不继承 `configs/base.yaml`**：

```yaml
training:
  local_epochs: 5
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 0.00002
  max_text_length: 2048
  optimizer: adamw
```

要求恰好八张可见 CUDA GPU，保持你之前的每卡 batch 4、累积 4，完整窗口有效批量
`8 × 4 × 4 = 128`。沿用原 DDP 的 shuffle、补齐和 epoch 尾部窗口处理；不足一个完整
窗口的部分仍按原实现更新，不丢弃。五轮的 optimizer steps 分别为：

| 客户端 | cls/0 | cls/1 | caption/0 | caption/1 | vqa/0 | vqa/1 |
|---|---:|---:|---:|---:|---:|---:|
| 每个 rank 训练任务的总步数 | 275 | 235 | 305 | 285 | 215 | 220 |

同一客户端的三种 rank 拥有相同样本顺序、步数、学习率和种子；不同客户端样本量不同，
因此它们之间步数不同。AdamW 沿用原训练器默认 betas=(0.9, 0.999)、eps=1e-8、
weight_decay=0.01，无新增学习率调度或梯度裁剪。

LoRA 只训练语言模型 Q/K/V/O 投影，保持 dropout=0.05、bias=none。
rank/alpha 配对为 **2/4、4/8、8/16**，固定 alpha/r=2，避免缩放强度随 rank 改变。
LLaVA、视觉塔和投影连接器保持冻结。权重只从以下本地目录读取：

```text
pretrained/llava-v1.5-7b/
pretrained/clip-vit-large-patch14-336/
```

## 训练、评估如何衔接

每个客户端/rank 只调用一次现有 `Llava15Adapter.local_train`，连续训练五个 epoch。
它走原有 `_local_train_historical_ddp`，不会把五轮拆成五次 optimizer 重置的训练。
独立入口通过已有 step hook，在每轮最后一次 optimizer step 完成、梯度清零后评估。
评估结束恢复各模块的 train/eval 状态、Python/NumPy/PyTorch/CUDA RNG 和 use_cache，
然后继续下一轮，不用验证结果改变训练。

每轮评估当前客户端自己的模型、自己的任务验证集。八卡各生成约 1/8 的预测，
不为评估补齐重复样本；收集完整预测并检查每条恰好出现一次后，才计算完整语料指标。
预测仅在内存中用于计算指标，计算完成后释放，不写入磁盘；LoRA 参数也不保存。
**不平均八卡上的局部指标。** Caption 保留每图五个参考，使用项目既有 pycocoevalcap
计算 BLEU、METEOR、ROUGE-L、CIDEr 等；需要现有环境中的 Java。
生成固定 greedy、num_beams=1、max_new_tokens=128。
VQA 沿用项目现有答案匹配规则，不声称改成 AOKVQA 官方 soft-score。

当前数据有 34 条 VQA validation 记录的图像出现在两个 VQA 客户端训练图像的并集中。
为观察容量需求，实验在启动时只根据图像路径元数据固定两个协议：

- 主结果 `validation_image_disjoint`：966 条未见图像的原验证记录；两个客户端、所有
  rank/seed 始终使用同一个子集。
- 附加 `validation`：原 1000 条记录，方便与原数据协议对照。

每轮只生成一次原 1000 条预测，再从这些预测计算两套指标，不重复生成。
不改动任何源标注文件，不读取 test，不改其他实验的评估规则。
cls/caption 继续使用原 1000 条 validation。实际数量与 ID 哈希保存到数据清单；
若以后换数据，按实际文件重新计算，不写死 966。

主比较固定使用 **第 5 个 epoch** 的性能，不为不同 rank 分别挑最好的一轮。
第 1～5 轮的指标都保存，以便查看学习曲线。默认只有一个 seed，可先看趋势；正式扩大复验
时可把独立配置的 `experiment.seeds` 改成 `[42, 43, 44]`，运行数相应变成 54。
不同 rank 参数维度不同，同种子控制随机实验条件，并不意味着逐参数初始化完全相同。

## 启动

在服务器仓库根目录、已安装项目 requirements 且数据/图片/权重就绪后：

```bash
bash scripts/run_local_rank_probe.sh --run-name rank_probe_e5_seed42
```

也可直接使用同一入口：

```bash
python scripts/run_local_rank_probe.py --run-name rank_probe_e5_seed42
```

启动器自己调用 torchrun，**不要在外面再套一次 torchrun**。默认使用全部可见八卡，
不足八卡或多于八卡都会明确报错，不会退回单卡。若机器超过八卡，可通过
`CUDA_VISIBLE_DEVICES` 选出八卡。同一时间只运行一个客户端/rank，完成后进入下一个。
无需生成 Plan、system profile 或 experiment profile。

终端只由 rank 0 显示进度，非主进程不显示进度条，也关闭各进程的权重加载库进度条。
固定两行：上方 `rank probe total: 0/18` 是整个实验已完成的训练任务数，下方是当前
客户端的训练步数或验证样本进度。每个任务五轮训练和评估全部完成后，总数才加 1；
选择任务/客户端/rank 后，总数按实际任务列表计算。评估显示主进程自己的验证分片，
其余七卡仍正常评估，最终仍汇总八卡完整预测计算指标。
启动器按原始字节转发日志，保留原地刷新的回车和光标控制符；不会再把每次刷新变成
新的一行。`launcher.log` 保留同样的原始输出，因此可能包含终端控制符。

输出目录已存在时拒绝覆盖；要重跑请换 `--run-name`。省略该参数会自动生成时间戳名称。
若需要分开执行完整训练任务，可选择任务/客户端；每个被选任务仍完整训练五轮：

```bash
bash scripts/run_local_rank_probe.sh --run-name rank_probe_vqa_c0_e5 \
  --task vqa --client client_0
```

上述命令训练这个客户端默认的三种 rank。也支持 `--rank 2`、`--seed 42`；`--seed`
必须已在配置种子列表里。画图需要所选客户端/seed 在所选 rank 上形成完整配对矩阵。

## 已完成 2/4/8，只补跑 16/32

**保留原目录，不修改或覆盖它。** 推荐从原运行的 `resolved_config.yaml` 读取冻结
配方，而不是修改 `base.yaml` 或把原实验名称重新使用。假设原目录名称是
`rank_probe_e5_seed42`，在仓库根目录执行：

```bash
# 如果原实验用了其他名称，只替换这里的路径。
OLD_RUN=runs/local_rank_probe/rank_probe_e5_seed42

bash scripts/run_local_rank_probe.sh \
  --extend-from "$OLD_RUN" \
  --ranks 16 32 \
  --run-name rank_probe_r16_r32_e5_seed42
```

这次只生成 6 个客户端 × 2 个新增 rank = **12 次训练**，总进度为 `0/12`。
保持原八卡 DDP、每卡 batch、梯度累积、epoch、学习率、种子、目标模块和验证协议。
alpha 按原 `lora_alpha_per_rank` 自动设置；原配方 alpha/r=2 时，rank16/32 对应
alpha32/64。每次仍从基础模型与新 LoRA 初始化，不加载 rank8 的训练后参数。

补跑入口要求旧实验已经完整完成，检查标注/验证 ID 和共享模型、数据、指标实现的
哈希一致，并拒绝重跑旧目录里已经完成的同一客户端/rank/seed。
旧实验若只选择了部分客户端或 seed，补跑也只扩展这些已存在的组合。
如果需要单独跑一个新增 rank，可用 `--rank 16`；它与 `--ranks` 不能同时使用。
也可在普通独立运行中使用 `--ranks 2 4 8 16 32`；省略时仍保持原默认 2/4/8。
不需要重新生成 Plan，也不需要手改原配置文件。

新增数据写入独立目录：

```text
runs/local_rank_probe/rank_probe_e5_seed42/              # 原2/4/8，保持原样
runs/local_rank_probe/rank_probe_r16_r32_e5_seed42/       # 新16/32
```

补跑全部完成后，复制小型结果记录到一个**新的汇总目录**，一起分析五种 rank：

```bash
python tools/merge_local_rank_probe.py \
  "$OLD_RUN" \
  runs/local_rank_probe/rank_probe_r16_r32_e5_seed42 \
  --output-directory runs/local_rank_probe/rank_probe_r2_4_8_16_32_e5_seed42

python tools/plot_local_rank_probe.py \
  runs/local_rank_probe/rank_probe_r2_4_8_16_32_e5_seed42
```

汇总工具**不在任何源目录重新导出 CSV，不移动、不删除、不覆盖原文件**。
它复制各 job 的数值指标、loss（如有）、配置/采样元信息和完成记录；不复制大日志、
预测文本或 LoRA 参数。汇总目录已存在时拒绝覆盖，重复操作需换一个新目录名。
即使没有模型/图片，已经完成的两套结果也可以离线合并。

合并前检查训练与评估配方、数据哈希、共享代码、软件版本、逐轮 optimizer steps、
验证 ID、样本数及完整配对关系。重复 rank、缺少轮次、只为部分客户端补齐 rank 或
条件不一致会明确报错，不会静默平均。这里能核对本地权重路径及软件版本，不能从
旧记录证明大权重文件内容从未变动；请保持原 LLaVA/CLIP 权重不变。

五 rank 汇总保留所有客户端、每个 epoch 的原始数值；默认画固定第 5 轮的曲线，
横轴为 **2、4、8、16、32**。`rank_gains.csv` 会包含四段增益：
`gain_2_to_4`、`gain_4_to_8`、`gain_8_to_16`、`gain_16_to_32`，以及各段每新增
rank 的增益。比较增益饱和时同时看这两套数值，不能忽略各段增加的 rank 数不同。
旧三 rank 实验的导出与画图命令继续可用。

## 保存内容

```text
runs/local_rank_probe/rank_probe_e5_seed42/
├── manifest.json                    # 运行状态、18个任务、配置/源代码哈希
├── resolved_config.yaml             # 本次独立配置快照
├── data_manifest.json               # 源标注哈希、验证ID、固定子集
├── metrics_long.csv                 # 每客户端/rank/epoch/protocol/metric，含状态
├── final_epoch_metrics.csv          # 第5轮主协议结果，仍保留每个客户端
├── task_rank_curves.csv             # 每任务/seed/rank/epoch的客户端等权均值
├── rank_gains.csv                   # 每客户端/seed/epoch的2→4和4→8增益
├── training_steps.csv               # 所有任务的逐optimizer-step loss
├── epoch_training.csv               # 每轮loss、时间、峰值显存
└── cls/client_0/rank2/seed42/        # 其余17个任务使用相同目录结构
    ├── job_config.yaml
    ├── metadata.json                # trainable数量、alpha/r、版本、初始化哈希等
    ├── expected_sampler.json        # 原采样规则推导的顺序哈希，非额外训练采样器
    ├── launcher.log
    ├── training_steps.jsonl
    ├── epoch_training.jsonl
    ├── metrics.jsonl
    └── result.json                 # 五轮均完成才写completed
```

CSV 使用 UTF-8 BOM，可直接用 Excel 打开。`metrics_long.csv` 一行对应一个原始指标，
不会把不同任务的 accuracy、CIDEr 合成一个数。步级 loss 是各 DDP rank 的本地窗口 loss
等权均值；epoch loss 是这些 optimizer 窗口的均值，**不是全语料 token 加权困惑度**。
完整窗口为 128 样本，尾部窗口和原采样补齐重复数可从配置/采样清单确认。

`rank_gains.csv` 保存所选相邻 rank 的增益和每新增 rank 的增益；包含 2/4/8 时也保留
原来的 Δ(4→8)/Δ(2→4) 比值。
当 Δ(2→4)≤0 时不计算增益比，避免误导。任务曲线对两个客户端等权，客户端原始曲线
始终保留；不把两个客户端差异当作独立随机种子置信区间。

逐步 loss、元数据与 CSV 导出由本实验独立 hook 处理，不写入 baseline 输出目录。
记录时间包含/排除评估的口径由字段名区分；`epoch_train_seconds` 不含本轮验证和记录
写入时间。仅保留画图与核对实验条件需要的指标、loss、时间/显存统计和配置元数据，
不创建预测文件或 LoRA 检查点，**不提供中途断点续训**。

若某个任务失败，启动器停止后续训练，保留日志及已产生的数据并标记 failed。
汇总 CSV 保留状态，失败任务不加入均值；不完整曲线会标记 `complete=False`。
画图工具要求所选任务全部完成，避免把残缺结果画成完整对比。
运行期间如果标注、配置或核心训练/评估源代码被改动，下一个任务启动会拒绝混用版本。

## 导出与画图

训练每完成一个任务都会自动刷新根目录 CSV。需要重新导出时：

```bash
python tools/export_local_rank_probe.py runs/local_rank_probe/rank_probe_e5_seed42
```

完整实验结束后画图（matplotlib 已列在 requirements 中）：

```bash
python tools/plot_local_rank_probe.py runs/local_rank_probe/rank_probe_e5_seed42
```

默认在本次目录 `figures/` 输出 PNG/SVG：

- `rank_validation_epoch5`：三个任务分别成图面板，显示每个客户端曲线及等权均值。
- `rank_gains_epoch5`：逐客户端比较所选相邻 rank 的收益，五 rank 时包含四段。

可用 `--epoch 1` 查看同一固定轮次的结果。若运行多个 seed，曲线阴影为按 seed
计算的标准差，不是置信区间；单 seed 不画统计误差带。

本实验支持的结论限定为：**所选任务客户端在相同设置与预算下表现出不同容量响应趋势**。
不同任务的数据量、输出复杂度等仍不同，不能仅由三点曲线认定普遍最优 rank，或认定
性能差异全部由任务本质导致。保留这六个客户端原始结果，便于后续复验。

## 与原项目的隔离边界

新入口、配置、运行时和输出均独立；复用现有模型/数据/指标的公开接口。
没有改动模型结构、历史 DDP 函数、batch collator、联邦 runner、方法注册表、
ours、OPCM、其他 baseline、原配置和数据。旧命令照常使用原代码路径。
本地只做语法、配置、数据元信息和非训练工程验证，没有加载 7B、下载权重或启动训练；
八卡 CUDA/NCCL 的真实运行与实际性能仍需在服务器执行上述正式命令验证。
