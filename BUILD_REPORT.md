# AFVLM_CM 构建报告

## 交付范围

- 从项目规格书零起点实现 `code/afl_vlm` 完整包。
- 实现 `tiny_mock` 与 Qwen2.5-VL LoRA/QLoRA 模型适配器。
- 实现 FedMLLM 统一数据适配、确定性不重叠划分和 held-out 诊断数据。
- 实现确定性虚拟调度、fixed/per-task/per-client/scripted 延迟和任务延迟重排控制。
- 实现固定加法、staleness decay、FedAvg、FedAsync、FedBuff 及完整方法插件生命周期。
- 实现 E1 stale/fresh twin、E2 biased/balanced start 和等配额端到端 trace。
- 实现全部 JSONL/JSON/CSV 结果、跨 seed 汇总、严格单文件配置及防覆盖策略。
- 提供完整 README、可编辑安装配置、命令行脚本、测试与 GitHub Actions CI。

## 已执行验证

- `python -m scripts.validate_config --config configs/run.yaml`
- `python -m scripts.run_experiment --config configs/run.yaml --dry-run`
- `python -m scripts.run_experiment --config configs/run.yaml`
- `python -m pytest`
- `ruff check .`
- `ruff format --check .`

验证结果：14 个测试通过；默认配置展开 2 个独立运行并生成 3 行机制汇总；最终产物中的 10 个 JSON 与 8 个 JSONL 文件全部通过解析审计。

`tiny_mock` 的 E2 显示偏置起点相对平衡起点有更高的初始/5 步后 S loss、更大纠偏 norm 和更强 F 回退；E1 的 `ExtraHarm` 未呈正向信号。两者都只用于证明流程和判据不会被硬编码为“支持”，不能替代 7B 真模型结果。

默认 `tiny_mock` 运行输出位于被 Git 忽略的 `runs/validation/`。科学结论仍需按 README 接入真实数据并使用 Qwen2.5-VL 7B 复核。

## 发布状态

发布目标为 `https://github.com/wangpengyu2004/AFVLM_CM` 的 `main` 分支；具体提交哈希以仓库历史为准。
