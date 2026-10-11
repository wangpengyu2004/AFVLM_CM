# 本次 baseline 更新与旧实验/profile 的兼容性

对比基准是更新前 main 的 `ced4779`。上传包括此前待提交的 baseline 修正、OrthoFL、无服务器数据 FedASMU 适配、跨执行器批量对齐、Grounding 8k 划分工具/摘要及复审报告，不是只上传报告。

## `ours` 和 OPCM 的改动范围

- `code/afl_vlm/methods/ours.py`、`configs/methods/ours.yaml` 与更新前 main 完全相同。
- `configs/methods/opcm_lora.yaml` 完全相同：merge_weight=0.5、retain=0.5、constant staleness、eps=1e-8。
- OPCM 的共享聚合基类与 OPCM 类语法树相同；过滤函数增加 retain 合法性检查，保留之前实际使用的浮点 cutoff 运算。没有替换主融合公式。
- `configs/base.yaml` 本次仅新增解释注释，解析数值与更新前 main 相同：local_epochs=1、batch_size=4、gradient_accumulation=4、lr=2e-5、max_text_length=2048、AdamW。
- 公共采样器/执行入口有重构，以便单卡客户端匹配八卡批量。原八卡 B4/G4、分片补齐、尾窗口、loss 归一、optimizer/hook 时机保留原规则；不是说公共代码一行没改。已有非训练检查逐批比对旧八卡规则。

这些结论只针对本次更新，不等于拿当前模板与用户服务器上的任意旧 profile 比较过。

## 是否需要新 profile / Plan

| 目的 | 操作 |
|---|---|
| 继续运行已有 ours/OPCM 快照，数据与 Plan 一致 | 不必仅因 git pull 重建；旧快照不被覆盖 |
| 使用新增 OrthoFL 或修正后的 baseline 参数 | 创建新 profile；旧快照不会自动新增/更新方法配置 |
| 启用单卡与八卡有效批量对齐 | 创建当前版本新 profile，冻结八卡参考 |
| 数据分区、B/G、epoch、rounds 或系统条件变化 | 从当前数据生成新 Plan，不复用旧 Plan |

若共同训练设置、数据与系统条件仍与旧比较实验一致，在八卡全部可见的服务器运行：

```bash
python tools/generate_system_profiles.py \
  --profile baseline_audit_ref8_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from YOUR_OLD_PROFILE
```

将 `YOUR_OLD_PROFILE` 换成真实名称。**只复用 system profile/Plan，不复用旧 YAML 配置。** 新配置从当前 base/model/dataset/method 模板解析，名称本身不会设置参数。

提示应为 `GPUs=8, effective_batch=128`。自动检测遵循可见 CUDA 卡数，不要只暴露一张卡生成后当作八卡参考。无 CUDA 电脑可显式加 `--reference_world_size 8`；服务器八卡正常可见时不需要该参数。

随后例如运行 2 clients/task：

```bash
bash scripts/run_one.sh ours 2 baseline_audit_ref8_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh opcm_lora 2 baseline_audit_ref8_e1_bs4_ga4_r10_s42
```

如果旧 Plan 对应 Grounding 4,500 条，而当前数据是 8,000 条，不能复用。省略 `--reuse_plans_from` 从实际本地分区创建新 Plan；这种数据/计划变化会改变实验，不应说是复现旧结果。GitHub 只上传工具与摘要，不同步被忽略的实际数据。

## 新快照是否会改变旧两种方法的结果

在旧实验本来就是八卡 DDP B4/G4，数据/模型/LoRA/方法参数/seed/Plan/评估/执行策略相同的前提下，新参考批量元数据不改变这两种方法的名义训练预算，其聚合算法也没有因新增 baseline 而替换。

但不能承诺位级结果相同，也不能把“方法模板没改”当作“新 profile 与任意旧快照相同”。例如旧 ours 快照未启用 history_age，而当前模板启用，新快照会读取当前值；旧 bs1/512 配置也不会被自动保留。

生成前先将当前模板对齐到旧实验。生成后检查差异，例如：

```bash
diff -u \
  experiment_profiles/YOUR_OLD_PROFILE/configs/2clients/ours.yaml \
  experiment_profiles/baseline_audit_ref8_e1_bs4_ga4_r10_s42/configs/2clients/ours.yaml

diff -u \
  experiment_profiles/YOUR_OLD_PROFILE/configs/2clients/opcm_lora.yaml \
  experiment_profiles/baseline_audit_ref8_e1_bs4_ga4_r10_s42/configs/2clients/opcm_lora.yaml
```

路径、profile 名称、输出目录、新增参考批量元数据可以不同；训练、模型、方法、数据、评估和执行设置的实质差异必须核对。各 profile 输出路径独立，不覆盖旧结果。

## 尚未解决的限制

Pilot 的 FP16 AdamW 风险和 UniFed-LoRA 全输出头显存限制只是被查明，**本次没有修复**。上传成功不意味着所有 baseline 已真实训练通过。详见 [独立复审报告](baseline_reaudit_20261011.md)。
