# 历史八卡训练恢复与 FedAsync 定向修复

## 实际日志与回归线索

用户日志明确为：

```bash
python tools/generate_system_profiles.py --profile e1_bs4_ga4_r10_s42 --reuse_plans_from e1_bs4_ga4_r10_s42_lb
bash scripts/run_one.sh fedasync 2 e1_bs4_ga4_r10_s42
```

日志显示八卡 `client_ddp`、每卡batch4、累积4、有效批量128、68k训练记录；
首个客户端尚未完成任何更新，重入式 checkpoint backward 报
`Expected to mark a variable ready only once`。它不是 ours/OPCM 失败的证据，
也不是 Plan 到达顺序或评估造成的错误。

更新 `c52a69c` 同时进行了单卡公平性改造和 baseline 修正，并非只修改单卡。
其中 FedAsync 原来没有 local proximal loss，更新后默认 `paper_loss` 直接
在模型forward外读取 LoRA 参数，构成另一条 autograd 路径。该路径与
模型重入式 checkpoint 的反向传播重复引用同一参数，与日志症状一致。
即使正则值最初为0，该图分支仍存在。此前把原因直接归为旧八卡实现
有隐患不够严谨；本次据具体新增路径定向修复，非服务器实测证明。

模型外参数引用/重入反向的限制见
[PyTorch 2.0.1 官方实现](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/nn/parallel/distributed.py)。

## 修复保持同一个 FedAsync 目标

目标仍为

```text
L = L_CE + mu/2 * ||theta - theta_download||²
gradient = gradient_CE + mu * (theta - theta_download)
```

仅 FedAsync `paper_loss` 的执行方式改变：

1. loss日志仍包含正则数值，但计算正则数值时 detach LoRA 参数，
   因而它不额外建立参数反向路径。
2. 按原来的 micro-batch、`loss / len(window)`、DDP/no_sync 规则完成 CE 梯度。
3. 已有的 `transform_gradients` hook 在完整 optimizer window 后、
   `optimizer.step` 前解析加入 `mu*(theta-theta_download)` 一次。
4. AdamW 接收包含正则的完整梯度，再按原规则 step 和 zero_grad。

参数在一个累积窗口内尚未更新，因此每个micro-batch的二次正则梯度相同；
对window平均后就是同一个正则梯度，无须再除以累积次数。
DDP各rank参数/下载锚点一致，rank平均后正则仍是同一个量，
无需除以world_size或新增all-reduce。它不是另做weight decay或参数收缩，
也不增加forward/backward、optimizer步骤或可训练参数。

仅操作联邦可训练参数；FP32计算正则差值后转换为参数梯度dtype，
仍在GPU完成。mu=0和none模式不改梯度；official_contraction保持原实现。
FedProx、其他baseline和ours/OPCM的loss/gradient hook不改变。

FedAsync未声明mean-gradient收集，因此该方法不依赖hook前的梯度快照。
将来若扩展这种能力或使用GradScaler，需重新审查收集/缩放顺序；
当前LLaVA训练不使用GradScaler。本次未改变既有精度策略。
浮点归约/加法顺序不同，不声称与旧FedAsync逐位相同。

## 八卡旧路径与单卡公平性隔离

基准为公平性更新前main的 `ced4779`。
`_local_train_historical_ddp` 是该提交完整的 `local_train` 函数体，仅改函数名。
采样、种子、分片补齐、尾窗口、no_sync、AdamW、loss/window、
梯度/step/progress hook和返回值均与历史函数一致。
DDP包装、模型加载和collator也通过无例外的历史AST指纹比较。
不存在额外static_graph设置或首窗口全同步改动；先前统一兼容草稿已撤回。

普通AdamW DDP（包括ours/OPCM）进入上述历史函数；
特殊SGD/梯度估计等新增baseline保留自己的扩展路径。
单卡客户端保留reference-batch预算对齐：八卡参考B4/G4对应单卡B4/累积32，
包括原分片补齐和尾窗口。没有修改base、方法参数、已有profile/Plan、
数据、评估或调度入口。

未知用户之前实际服务器运行SHA/resolved_config，不能把指定历史源码一致
说成任意旧实验位级复现，也没有将整个仓库回滚。

## 使用与验证边界

本次不要求重建profile/Plan，已生成的上述FedAsync快照可继续用。
但本地补丁上传并被服务器拉取后才生效；修复本身不自动覆盖快照。
失败输出若非空，应先保留/重命名旧运行目录，不自动删除实验结果。

新增检查仅做数值/梯度张量算术、loss图叶节点检查、配置与源码指纹；
不执行新训练forward/backward，不加载LLaVA，不下载权重。
整个仓库既有非训练回归和静态验证也运行。
这些检查验证解析导数与路径隔离，不等于真实服务器八卡已经训练通过。
