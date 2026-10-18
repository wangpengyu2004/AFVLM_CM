# 保留八卡实验的跨执行器批量公平性

## 公共配置

所有方法模板继承 `configs/base.yaml`：

```yaml
training:
  local_epochs: 1
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2.0e-05
  max_text_length: 2048
  optimizer: adamw
```

不需要在 base.yaml 手动写 `batch_size_scope` 或 `reference_world_size`。
生成 profile 时根据 `torch.cuda.device_count()` 自动识别可见 GPU 数（遵循
`CUDA_VISIBLE_DEVICES`），将 `batch_size_scope: reference_ddp` 和识别到的参考卡数
写入所有方法的快照及 manifest。训练使用已保存的参考值，不再随所在机器重新识别。
这些是快照内部元数据，请勿删除或手动修改不可变快照。

直接运行可编辑模板而不是 profile 时，在训练入口、启动 client workers 之前识别并
写入本次 resolved config；workers 继承已解析值，不会按自己的单卡可见性再次推断。
论文正式比较建议所有方法使用同一个 profile，不要在不同机器上分别运行未冻结模板。

参考卡数不是并发客户端数。基础 B/G 保留参考 DDP 的每卡含义；单卡自动补足累积量。
以下以你原来的八卡服务器为例：

| 执行方式 | 每卡 micro-batch | 完整窗口物理累积次数 | 有效批量 |
|---|---:|---:|---:|
| 原八卡 DDP、同一个客户端 | 4 | 4 | 128 |
| 单卡客户端 | 4 | 32 | 128 |

不要为单卡方法把基础 `gradient_accumulation` 改成32，否则八卡也会变成32。
保存八卡参考的 profile 允许一个训练 GPU 或八个 DDP ranks，其他 DDP 卡数明确报错，
避免静默改变预算。新 profile 若在四卡机器生成，其参考预算是4*4*4=64，不能据此
声称复现之前的八卡128实验。

## 不只匹配乘积

八卡保留原随机打乱、补齐分片、每卡 micro-batch、尾批、loss 除以当前窗口长度、
DDP 梯度平均以及 optimizer/hook 调用时机。不引入新的 token 加权目标。
单卡复现八个参考分片，将同一步的所有 rank micro-batches 顺序计算后才执行一次
optimizer.step。完整窗口32个 micro-batches，不是32次参数更新。
每个 micro-batch 仍只前向/反向一次，计算图用完即释放。

设一个 rank 当前窗口有 M 个 micro-batches，原梯度为
`(1/8) * sum_r [(1/M) * sum_j g(r,j)]`；单卡为
`(1/(8*M)) * sum_r sum_j g(r,j)`。各 micro-batch 梯度相同时，两者数学相等。
这仍是原 micro-batch 均值目标，不声称等同于直接堆128条计算的 token 均值目标。

尾窗口按实际 micro-batch 数归一化，不固定除以32。保留原参考 DistributedSampler
风格的重复补齐，不丢弃尾记录。日志中的记录贡献数包含补齐，不是唯一记录数。
普通 epoch 方法真实步数一致：`epochs * ceil(ceil(ceil(N/8)/4)/4)`。
例如 N=6071，一个 epoch 都为48步。敏感度/梯度钩子仍每个 optimizer step 调用一次。

匹配不保证位级相同：dropout 随机流、浮点累加/归约顺序及算法特有梯度处理仍有差异。
无需按卡数放大学习率。单卡实际 batch 仍为4，累积32一般不增加同等倍数的激活显存，
但方法自己的 adapter、历史缓存及梯度状态仍可能导致 OOM。单客户端计算可能更慢。

## 方法和虚拟 Plan

能力策略下当前仅 Pilot 保留每卡一个客户端；FedCompass、FedASMU、MasFL/AdaMasFL、
ACED、UniFed-LoRA 已通过方法私有适配接入 DDP，其他方法保持原执行方式。
公共批量/训练/评估代码不改，详见 [DDP 适配与边界](baseline_ddp.md)。

普通方法的原 Plan 保留：Plan 的 `local_steps` 是旧 B*G 单卡预算的名义虚拟时间单位，
并不等于八卡真实步数（上述例子为380和48）。进度/中途刷新使用真实48步；
虚拟 start/arrival、下载版本、聚合顺序仍使用原 Plan。

FedCompass 本来就分配实际 optimizer steps。新分配使用128条的步骤预算，同时将
每步虚拟计算成本乘参考卡数8，换算回原系统成本单位，不改客户端速度或系统随机值。
min/max_local_steps 仍是实际参数更新次数；算法可以改变工作量，不能声称其每次步数
都与固定 epoch 方法相同。

## 旧结果与运行

旧 immutable profiles 不覆盖。缺少 scope 的旧配置保留 `per_device` 行为。
已完成的八卡 B4/G4 结果不必仅因本次匹配重跑，但数据、模型、方法系数、评估、seed
等也必须相同；旧有效批量16的单卡结果需要重跑。

启用单卡匹配应从当前模板生成新快照。数据、B/G、epochs、轮数、seed 和系统条件
未变时，可以原样复用旧普通 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile fair_ref8_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from YOUR_OLD_PROFILE

bash scripts/run_one.sh adamasfl 2 fair_ref8_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ours 2 fair_ref8_e1_bs4_ga4_r10_s42
```

替换真实旧 profile 名称。新快照从当前全部模板生成，不只更新一个方法；检查其他参数
是否与已完成实验相符。数据/B/G等改变时不要复用旧 Plan。名称只是标签，不替代数值。

无需额外卡数参数，在八卡服务器生成就会保存参考8。若在没有 CUDA 的电脑生成，
程序会明确报错，不会静默按1卡生成。此时可以通过命令指定目标服务器的参考卡数，
仍不需要编辑 base.yaml：

```bash
python tools/generate_system_profiles.py \
  --profile fair_ref8_e1_bs4_ga4_r10_s42 \
  --reference_world_size 8 \
  --reuse_plans_from YOUR_OLD_PROFILE
```

启动时的 `freezing batch reference` 显示参考卡数、有效批量和来源。`--list`、静态
配置检查和旧 immutable profiles 不触发 GPU 检测，不要求 CUDA 可用。

## 记录与验证边界

启动打印 scope、effective_batch、training_ranks、per_device_batch 和
physical_accumulation；worker 数不参与一个客户端的批量计算。
`updates.jsonl` 的 `batching` 和 `system_stats.json` 记录参考/实际卡数、批量、累积、
loss/padding 规则及每步/总记录贡献数。profile manifest 记录参考预算和 Plan 名义单位。
仅验证索引、数学权重、步数、配置及代码接口，不加载7B或启动训练。
目标服务器 CUDA/NCCL、显存和数值稳定性仍需正式运行观察。
