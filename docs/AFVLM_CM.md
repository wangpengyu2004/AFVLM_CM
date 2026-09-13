# AFVLM-CM Data Contract

AFVLM-CM is a fixed-task federated benchmark. A logical client owns exactly
one task for the complete run; the asynchronous dimension is its system
timeline, not a task-incremental stream.

## Detected immutable partitions

| setting | partition root | clients | clients/task |
|---|---|---:|---:|
| 2 | `data/AFVLM_CM/partitioned/2_clients` | 12 | 2 |
| 5 | `data/AFVLM_CM/partitioned/5_clients` | 30 | 5 |
| 10 | `data/AFVLM_CM/partitioned/10_clients` | 60 | 10 |

Every setting contains `cls`, `caption`, `vqa`, `chart_vqa`,
`visual_reasoning`, and `grounding`. Every task directory contains the
existing `client_N.json`, `statistics.json`, `val.json`, and `test.json`.
Training totals per setting are 15,000 / 20,000 / 12,000 / 10,805 / 12,000 /
20,000 in canonical task order. No runtime code repartitions these records.

The integrity manifest in `configs/datasets/afvlm_cm_integrity.json` records
161 files, 164,752,723 bytes, and aggregate SHA-256
`2ebc769eae578da93a7aa0a80ffb5921c0d6e53e7e53db1e1137af50d9116931`.

## Loader interface

`AFVLMDataModule` discovers files and exposes:

- `get_client_ids()`
- `get_client_task(client_id)`
- `get_client_dataset(client_id)`
- `get_client_train_data(client_id)`
- `get_client_num_samples(client_id)`
- `get_task_val_data(task)` and `get_task_test_data(task)`
- `get_global_eval_sets()`

Each normalized sample retains the complete source JSON under
`sample.metadata["raw_record"]`. `resolve_image_path` strips only leading
`./`, normalizes separators, confines paths to `image_root`, and otherwise
preserves annotation-relative paths.

## Image root

All dataset configs use `data/AFVLM_CM/dataset`:

```text
data/AFVLM_CM/dataset/
├── ImageNet-R/train/...
├── Flickr30k/train/...
├── DVQA/images/...
├── FigureQA/images/...
└── COCO2014/{train2014,val2014}/...
```

AOKVQA and Grounding share COCO2014. The downloader materializes these exact
paths from the actual annotations and never rewrites instruction JSON.

## Evaluation

The single task-aware evaluator reports classification accuracy, CIDEr and
ROUGE-L, VQA accuracy, DVQA/FigureQA answer accuracy, and Grounding mean IoU
plus IoU@0.5. It reports a dictionary per task and does not raw-average
incompatible metric scales.

Flickr30k records currently contain one reference caption. The built-in CIDEr
uses the standard 1--4 gram TF-IDF cosine construction, but it is not claimed
to be numerically identical to five-reference COCO-caption evaluation.
