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
Training totals per setting are 13,000 / 15,000 / 11,000 / 9,500 / 11,500 /
12,000 in canonical task order, or 72,000 stored records in total. Every
client-count setting reuses exactly the same task pool; only its client
partition changes. Runtime code never repartitions these records.

The active partition version is `fcit_eval_v1_72k`. The integrity manifest in
`configs/datasets/afvlm_cm_integrity.json` records 158 files, 108,209,141
bytes, and aggregate SHA-256
`12acaf7ef39c1b3a1bcb40bcf8f8a11d81beda353b10f77aebaed197915ad74d`.

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

Grounding training keeps each source record as one LLaVA conversation and
supervises every assistant response in that conversation. Grounding
validation/final loading expands every user/assistant pair into a separate
query, so every referring expression receives its own prediction and IoU.
The default 2,048-token context covers the active records; an over-length
multi-turn record raises an error instead of silently truncating later
assistant answers. Caption evaluation samples retain exactly five Flickr30k
references.

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

The primary protocol follows conventional asynchronous FL evaluation:

1. after every configured number of successful server-version increments,
   evaluate a copy of the current server federated state on common validation
   sets;
2. process all scheduled arrivals and run method finalization, including a
   residual FedBuff flush;
3. evaluate the final server federated state on common test sets.

Evaluation does not mutate training state or consume virtual training time.
Every aggregated method uses `server_global` as its primary scope. Pilot keeps
task routing but uses the server state and the mean output of server-held
client visual adapters for the selected task. Local-only has no server model;
its explicitly labelled `client_local_mean` result averages client models
within each fixed task on the same common held-out split.

All methods call one shared FCIT-compatible evaluator:

- ImageNet-R and transformed AOKVQA use stripped, case-insensitive exact match;
- DVQA and FigureQA retain FCIT's stripped, case-insensitive
  `prediction in reference` rule;
- Flickr30k uses `pycocoevalcap` over five references and reports BLEU-1--4,
  METEOR, ROUGE-L, CIDEr, and their FCIT average (all multiplied by 100);
- Grounding reports FCIT's strict `IoU > 0.5` accuracy in percent and the
  additional raw mean IoU requested for AFVLM-CM.

It records protocol, split, server version, virtual time, and a metric
dictionary per task. It never raw-averages incompatible task metrics. Caption
METEOR requires a working Java runtime, as required by `pycocoevalcap`.
