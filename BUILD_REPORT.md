# AFVLM-CM 构建报告

## 当前交付

- 固定任务异步联邦 VLM 数据接入，支持 16/40/80 客户端和等量/不等量两种划分。
- Qwen2.5-VL 默认 LoRA/QLoRA 适配器与可选 LLaVA-1.5 适配器。
- AFVLM-CM 自研方法：任务条件下载记忆与陈旧冲突上传修正，可独立消融。
- 8 个可运行 baseline、确定性异步调度、任务相关/打乱延迟控制和等客户端上传配额。
- 单一正式配置、方法命令行选择、完整运行产物、汇总脚本、单元测试与 CI。

## 正式入口

```bash
python -m scripts.prepare_fixed_task_benchmark --validate-only
python -m scripts.preflight_benchmark_images
python -m scripts.validate_config --config configs/run.yaml
python -m scripts.run_experiment --config configs/run.yaml --methods afvlm_cm --output-root runs/afvlm_cm/ours
```

研究训练需要先补齐 `data/fcit/dataset/` 中被指令引用的全部图片和 Qwen2.5-VL 运行依赖。测试夹具只用于代码正确性检查，不作为论文实验结果。
