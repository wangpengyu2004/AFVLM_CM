---
license: apache-2.0
---

# FCIT fixed-task federated benchmark

This directory contains the local [FCIT dataset](https://huggingface.co/datasets/MLLM-CL/FCIT)
and a derived benchmark for the AFVLM setting. The derived benchmark removes the continual
task sequence: every generated client is permanently assigned to exactly one of the eight
tasks.

## Experimental design

The default build creates three client scales and two quantity profiles at each scale:

| Tier | Total clients | Clients per task | Balanced samples/client |
|---|---:|---:|---:|
| small | 16 | 2 | 6,000 |
| medium | 40 | 5 | 2,400 |
| large | 80 | 10 | 1,200 |

- `balanced`: every client in the tier has exactly the same number of training records.
- `quantity_skew`: client proportions are sampled deterministically from a Dirichlet
  distribution with concentration 0.5. Every client receives at least 20% of the task-wise
  mean, avoiding empty or unusably tiny clients.

Every task uses a shared pool of 12,000 training records, 512 probe records, and 1,024 final
evaluation records. The six variants reuse the same task pools and evaluation sets. Thus,
changing the client scale or quantity profile changes only the federated partition, not the
underlying examples or total training data (96,000 records per variant).

The canonical source is `partitioned_data/Task-related/seq/1.0`. Its 50 original clients per
task are used only as source shards: records are pooled per task, exact duplicates are removed,
then records are repartitioned. Original continual trajectories are not retained.

## Tasks

| FCIT task | Generated task name | Capability |
|---|---|---|
| Task1 | `imagenet_r_classification` | image classification |
| Task2 | `arxivqa_scientific_vqa` | scientific figure VQA |
| Task3 | `iconqa_diagram_vqa` | diagram VQA |
| Task4 | `clevr_compositional_vqa` | compositional VQA |
| Task5 | `ocr_vqa` | OCR-based VQA |
| Task6 | `flickr30k_captioning` | image captioning |
| Task7 | `figureqa_chart_vqa` | chart VQA |
| Task8 | `super_clevr_compositional_vqa` | compositional VQA |

## Output layout

```text
fixed_task_benchmark/
  benchmark_summary.json
  task_catalog.json
  common_eval/<task_name>/{probe,final}.jsonl
  small_16_clients/{balanced,quantity_skew}/
  medium_40_clients/{balanced,quantity_skew}/
  large_80_clients/{balanced,quantity_skew}/
    summary.json
    client_manifest.jsonl
    framework_config.yaml
    clients/<client_id>/train.jsonl
```

`client_manifest.jsonl` is the entry point for training. Every entry supplies `client_id`, its
single `task_id`/`task_name`, `num_train_samples`, and the relative `train_file`. Both manifest
and sample metadata explicitly set `fixed_task=true` and `continual=false`.

Image paths remain the original FCIT relative references. The preparation script does not
download, extract, or copy image bytes. They are resolved against the single shared root
`data/fcit/dataset`, never against a nested client directory. Each generated
`framework_config.yaml` enables `require_images: true`, so a real run fails before loading the
VLM if any required image is absent.

Current local image coverage is intentionally recorded rather than hidden: all 1,970 unique
FigureQA images required by Task7 are present and readable. Images for Task1-Task6 and Task8 are
not currently present; the unrelated local DVQA folder is not used by this eight-task benchmark.

## Rebuild and validate

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\prepare_fcit_fixed_benchmark.py
```

Replace an earlier generated output:

```powershell
.venv\Scripts\python.exe scripts\prepare_fcit_fixed_benchmark.py --force
```

Validate without rebuilding:

```powershell
.venv\Scripts\python.exe scripts\prepare_fcit_fixed_benchmark.py --validate-only
```

Check image completeness before training (returns a non-zero exit code if anything is missing):

```powershell
.venv\Scripts\python.exe scripts\preflight_fcit_images.py
```

To refresh `image_preflight.json` while images are still being assembled:

```powershell
.venv\Scripts\python.exe scripts\preflight_fcit_images.py --allow-missing
```

Once preflight reports `ready_for_training: true`, run a generated configuration directly:

```powershell
.venv\Scripts\python.exe scripts\run_experiment.py `
  --config data\fcit\fixed_task_benchmark\small_16_clients\balanced\framework_config.yaml `
  --dry-run
```

For normal experiments, prefer the repository-level `configs/fcit.yaml`: changing only
`dataset.variant` selects any of the six variants and automatically hydrates these generated
task/client manifests. The per-variant `framework_config.yaml` files remain useful as explicit,
self-contained audit artifacts.

The script uses only the Python standard library, records all parameters/checksums in
`benchmark_summary.json`, verifies task immutability, sample counts, client disjointness,
train/evaluation separation, and equality of the underlying task pools across all six variants.
