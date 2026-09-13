---
license: apache-2.0
---

# AFVLM-CM 固定任务数据

此目录保存 AFVLM-CM 的指令数据与图片。指令标注来源于 [MLLM-CL/FCIT](https://huggingface.co/datasets/MLLM-CL/FCIT)，但这里生成的是固定任务联邦划分：每个客户端永久绑定一个任务，不使用持续任务序列。

默认生成三档客户端规模（16/40/80）与两种数据量分布（`balanced`/`quantity_skew`）。每档都覆盖 8 个任务，共用每任务 12,000 条训练样本、512 条 probe 样本和 1,024 条 final 样本；训练集合在同任务客户端间不重叠。

```text
data/fcit/
  partitioned_data/Task-related/seq/1.0/  # 来源指令
  dataset/                                # 图片统一根目录
  fixed_task_benchmark/
    benchmark_summary.json
    common_eval/<task>/{probe,final}.jsonl
    {small_16_clients,medium_40_clients,large_80_clients}/
      {balanced,quantity_skew}/
        client_manifest.jsonl
        framework_config.yaml
        clients/<client_id>/train.jsonl
```

图片必须按每条记录的 `image` 相对路径放到 `data/fcit/dataset/` 下。数据构建脚本不会复制或下载图片。

```bash
python -m scripts.prepare_fixed_task_benchmark --force
python -m scripts.prepare_fixed_task_benchmark --validate-only
python -m scripts.preflight_benchmark_images
```

正式实验请使用仓库唯一配置 `configs/run.yaml`；其中 `dataset.variant` 负责选择六种划分之一。每个变体内的 `framework_config.yaml` 仅作为自包含的数据审计产物。
