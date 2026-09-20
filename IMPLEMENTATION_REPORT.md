# AFVLM-CM Implementation Report

This report records the implementation scope and engineering validation. No training, model loading, or model/image download was performed as part of the implementation.

## Files added

- `code/afl_vlm/data/afvlm_cm.py`
- `code/afl_vlm/methods/fedasmu.py`
- `code/afl_vlm/methods/fedcompass.py`
- `code/afl_vlm/methods/masfl.py`
- `code/afl_vlm/methods/ours.py`
- `code/afl_vlm/methods/pilot/__init__.py`
- `code/afl_vlm/methods/pilot/adapters.py`
- `code/afl_vlm/methods/pilot/aggregation.py`
- `code/afl_vlm/methods/pilot/method.py`
- `code/afl_vlm/methods/standard.py`
- `code/afl_vlm/methods/unifed_lora/__init__.py`
- `code/afl_vlm/methods/unifed_lora/descriptors.py`
- `code/afl_vlm/methods/unifed_lora/hypernetwork.py`
- `code/afl_vlm/methods/unifed_lora/method.py`
- `code/afl_vlm/scheduling/train_plan.py`
- `configs/base.yaml`
- `configs/datasets/afvlm_cm_2clients.yaml`
- `configs/datasets/afvlm_cm_5clients.yaml`
- `configs/datasets/afvlm_cm_10clients.yaml`
- `configs/datasets/afvlm_cm_integrity.json`
- `configs/models/llava15_7b_lora.yaml`
- `configs/methods/local.yaml`
- `configs/methods/fedavg.yaml`
- `configs/methods/fedprox.yaml`
- `configs/methods/fedadam.yaml`
- `configs/methods/fedasync.yaml`
- `configs/methods/fedbuff.yaml`
- `configs/methods/fedcompass.yaml`
- `configs/methods/fedasmu.yaml`
- `configs/methods/masfl.yaml`
- `configs/methods/adamasfl.yaml`
- `configs/methods/pilot.yaml`
- `configs/methods/unifed_lora.yaml`
- `configs/methods/ours.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/local.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/fedavg.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/fedprox.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/fedadam.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/fedasync.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/fedbuff.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/fedcompass.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/fedasmu.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/masfl.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/adamasfl.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/pilot.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/unifed_lora.yaml`
- `configs/experiments/llava/afvlm_cm/2clients/ours.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/local.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/fedavg.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/fedprox.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/fedadam.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/fedasync.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/fedbuff.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/fedcompass.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/fedasmu.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/masfl.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/adamasfl.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/pilot.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/unifed_lora.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/ours.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/local.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/fedavg.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/fedprox.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/fedadam.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/fedasync.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/fedbuff.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/fedcompass.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/fedasmu.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/masfl.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/adamasfl.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/pilot.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/unifed_lora.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/ours.yaml`
- `plans/afvlm_cm/2clients/system_profile.json`
- `plans/afvlm_cm/2clients/async_train_plan.json`
- `plans/afvlm_cm/5clients/system_profile.json`
- `plans/afvlm_cm/5clients/async_train_plan.json`
- `plans/afvlm_cm/10clients/system_profile.json`
- `plans/afvlm_cm/10clients/async_train_plan.json`
- `experiment_profiles/README.md`
- `experiment_profiles/default_e1_bs1_ga4_r10_s42/` (39 resolved configs,
  three system profiles, three TrainPlans, and a hash manifest)
- `scripts/evaluate.py`
- `scripts/run_baselines.sh`
- `scripts/run_one.sh`
- `tools/download_afvlm_cm_images.py`
- `tools/download_models.py`
- `tools/generate_system_profiles.py`
- `tools/validate_afvlm_cm_data.py`
- `tools/validate_repository.py`
- `docs/AFVLM_CM.md`
- `docs/baselines.md`
- `docs/ours.md`
- `tests/test_ours_method.py`
- `IMPLEMENTATION_REPORT.md`

## Files modified

- `.github/workflows/ci.yml`
- `.gitignore`
- `README.md`
- `pyproject.toml`
- `code/afl_vlm/__init__.py`
- `code/afl_vlm/config.py`
- `code/afl_vlm/data/partition.py`
- `code/afl_vlm/data/registry.py`
- `code/afl_vlm/federation/client.py`
- `code/afl_vlm/federation/server.py`
- `code/afl_vlm/federation/state_store.py`
- `code/afl_vlm/federation/types.py`
- `code/afl_vlm/methods/base.py`
- `code/afl_vlm/methods/registry.py`
- `code/afl_vlm/models/base.py`
- `code/afl_vlm/models/llava15.py`
- `code/afl_vlm/models/registry.py`
- `code/afl_vlm/runner.py`
- `scripts/run_experiment.py`

## Files deleted

- `BUILD_REPORT.md`
- `claudedocs/research_2026-09-13_fixed_task_async_baselines.md`
- `code/afl_vlm/aggregation/__init__.py`
- `code/afl_vlm/aggregation/base.py`
- `code/afl_vlm/aggregation/fedasync.py`
- `code/afl_vlm/aggregation/fedbuff.py`
- `code/afl_vlm/aggregation/immediate.py`
- `code/afl_vlm/data/collators.py`
- `code/afl_vlm/data/fedmllm.py`
- `code/afl_vlm/data/fixed_task_preset.py`
- `code/afl_vlm/evaluation/__init__.py`
- `code/afl_vlm/evaluation/branch.py`
- `code/afl_vlm/evaluation/paired_metrics.py`
- `code/afl_vlm/evaluation/probes.py`
- `code/afl_vlm/evaluation/task_metrics.py`
- `code/afl_vlm/experiments/__init__.py`
- `code/afl_vlm/experiments/biased_start.py`
- `code/afl_vlm/experiments/common.py`
- `code/afl_vlm/experiments/context.py`
- `code/afl_vlm/experiments/end_to_end.py`
- `code/afl_vlm/experiments/stale_twin.py`
- `code/afl_vlm/methods/baselines/__init__.py`
- `code/afl_vlm/methods/baselines/async_additive.py`
- `code/afl_vlm/methods/baselines/fedasync.py`
- `code/afl_vlm/methods/baselines/fedavg_sync.py`
- `code/afl_vlm/methods/baselines/fedbuff.py`
- `code/afl_vlm/methods/baselines/fedcompass_sim.py`
- `code/afl_vlm/methods/baselines/fedopt_sync.py`
- `code/afl_vlm/methods/baselines/staleness_decay.py`
- `code/afl_vlm/methods/custom/__init__.py`
- `code/afl_vlm/methods/custom/afvlm_cm.py`
- `code/afl_vlm/models/qwen25_vl.py`
- `code/afl_vlm/models/tiny_mock.py`
- `code/afl_vlm/scheduling/base.py`
- `code/afl_vlm/scheduling/delay_models.py`
- `code/afl_vlm/scheduling/virtual_event.py`
- `configs/run.yaml`
- `scripts/preflight_benchmark_images.py`
- `scripts/prepare_fixed_task_benchmark.py`
- `scripts/summarize_results.py`
- `scripts/validate_config.py`
- `tests/fixtures/tiny_config.yaml`
- `tests/test_additive_commutativity.py`
- `tests/test_branch_restore.py`
- `tests/test_config.py`
- `tests/test_end_to_end.py`
- `tests/test_equal_quota.py`
- `tests/test_fixed_task_prepartitioned.py`
- `tests/test_methods.py`
- `tests/test_paired_metrics.py`
- `tests/test_project_runtime.py`
- `tests/test_schedule.py`
- `tests/test_tiny_pipeline.py`

## Detected AFVLM-CM structure

- `data/AFVLM_CM/partitioned/2_clients`: 12 clients (2 per task).
- `data/AFVLM_CM/partitioned/5_clients`: 30 clients (5 per task).
- `data/AFVLM_CM/partitioned/10_clients`: 60 clients (10 per task).
- Each setting contains `cls`, `caption`, `vqa`, `chart_vqa`, `visual_reasoning`, and `grounding`.
- Each task contains the existing `client_N.json`, `statistics.json`, `val.json`, and `test.json` files.
- Per-setting training totals remain: cls 15,000; caption 20,000; vqa 12,000; chart_vqa 10,805; visual_reasoning 12,000; grounding 20,000.
- Integrity: 161 files, 164,752,723 bytes, aggregate SHA-256 `2ebc769eae578da93a7aa0a80ffb5921c0d6e53e7e53db1e1137af50d9116931` under the documented validator convention.

## Required paths

- Dataset configs: `configs/datasets/afvlm_cm_2clients.yaml`, `configs/datasets/afvlm_cm_5clients.yaml`, `configs/datasets/afvlm_cm_10clients.yaml`.
- LLaVA: `pretrained/llava-v1.5-7b`.
- CLIP: `pretrained/clip-vit-large-patch14-336`.
- Images: `data/AFVLM_CM/dataset` with the source subdirectories documented in README.
- Default federated scope: LoRA r=8, alpha=16, dropout=0.05, bias=none; mm_projector disabled unless the shared model config is deliberately changed for every comparison.
- Default experiment/system-profile seed: 42.

## Download commands

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/download_models.py --resume --skip_existing
HF_ENDPOINT=https://hf-mirror.com python tools/download_afvlm_cm_images.py --all --resume --skip_existing
```

The image downloader resolves ImageNet-R/Flickr30k from the live
`HaiyangGuo/UCIT` file tree and DVQA/FigureQA from the live `MLLM-CL/FCIT`
`dataset/` tree. It downloads only selected files and fails explicitly on an
ambiguous or changed repository structure. COCO2014 comes only from the
official COCO HTTP host and is shared by AOKVQA and Grounding.

## Supported methods and commands

The registered methods are `local`, `fedavg`, `fedprox`, `fedadam`, `fedasync`, `fedbuff`, `fedcompass`, `fedasmu`, `masfl`, `adamasfl`, `pilot`, `unifed_lora`, and the implemented proposed method `ours`.

- `local`: `bash scripts/run_one.sh local 2` (replace 2 with 5 or 10).
- `fedavg`: `bash scripts/run_one.sh fedavg 2` (replace 2 with 5 or 10).
- `fedprox`: `bash scripts/run_one.sh fedprox 2` (replace 2 with 5 or 10).
- `fedadam`: `bash scripts/run_one.sh fedadam 2` (replace 2 with 5 or 10).
- `fedasync`: `bash scripts/run_one.sh fedasync 2` (replace 2 with 5 or 10).
- `fedbuff`: `bash scripts/run_one.sh fedbuff 2` (replace 2 with 5 or 10).
- `fedcompass`: `bash scripts/run_one.sh fedcompass 2` (replace 2 with 5 or 10).
- `fedasmu`: `bash scripts/run_one.sh fedasmu 2` (replace 2 with 5 or 10).
- `masfl`: `bash scripts/run_one.sh masfl 2` (replace 2 with 5 or 10).
- `adamasfl`: `bash scripts/run_one.sh adamasfl 2` (replace 2 with 5 or 10).
- `pilot`: `bash scripts/run_one.sh pilot 2` (replace 2 with 5 or 10).
- `unifed_lora`: `bash scripts/run_one.sh unifed_lora 2` (replace 2 with 5 or 10).

- `ours`: `bash scripts/run_one.sh ours 2` (replace 2 with 5 or 10). It is excluded from `scripts/run_baselines.sh` because that script runs comparison baselines only.

## Versioned parameter and TrainPlan profiles

`tools/generate_system_profiles.py --profile <name>` creates a new immutable
profile instead of overwriting the canonical plans. `--reuse_plans_from <name>`
copies a compatible old plan when only non-scheduling parameters change. Both
single and batch runners accept the profile as an optional final argument, for
example `bash scripts/run_one.sh fedasync 2 default_e1_bs1_ga4_r10_s42`.

## Paper adaptations

- FedCompass retains computing-power-aware bounded local-step assignment and grouped semi-asynchronous aggregation inside the shared virtual clock; only federated trainable state is optimized.
- FedASMU retains dynamic staleness-aware server mixing and a mid-local fresher-global interpolation. A deterministic configurable refresh position and online coefficient updates replace the original full RL request controller.
- MasFL/AdaMasFL implement client/global controls and historical descent momentum; AdaMasFL normalizes local directions and aggregates observed local displacement. States are LoRA/configured trainables.
- Pilot isolates task/client visual adapters and CT-MoA in a Pilot-only connector and implements task visual plus nearest-client text-LoRA aggregation. Its two paper stages are jointly optimized here.
- UniFed-LoRA disables backbone heterogeneity but retains configurable task and modality/architecture/layer/module descriptor conditioning through a server hypernetwork.

## Limitations

- No 7B job has been launched; target-machine CUDA, memory, dependency, and checkpoint compatibility still require real-environment confirmation.
- Current read-only image audit finds 64,934 unique references and 61,304 missing files.
- Flickr30k records have one reference caption in this partition; the built-in CIDEr construction cannot be claimed numerically identical to five-reference official evaluation.
- Pilot joint-stage training and FedASMU deterministic refresh are disclosed adaptations, not exact original-framework reproductions.
- The final checkpoint contains server/method/scheduler/random state for reproducible evaluation, but the current CLI does not resume training from an arbitrary in-flight asynchronous event.
- Bash is unavailable on the current Windows host, so shell entrypoints were statically inspected but not executed by Bash.
- The repository-local verification environment does not currently have the newly declared NumPy/Hugging Face runtime dependencies installed; CLI parsing was checked where imports permit, and the documented editable install is required before downloads/training.
- The proposed method is intentionally not invented.

## Evaluation protocol

- Periodic validation is triggered by successful server model updates, using
  `evaluation.eval_every_server_updates` and the current server federated state.
- Final test evaluation runs only after all scheduled arrivals and method
  finalization, including a configured residual-buffer flush.
- Aggregated methods report `server_global`; Local-only reports the explicitly
  labelled `client_local_mean` because it has no global model.
- Pilot uses its current server state with task routing and a task-wise mean of
  server-held client visual adapters, rather than arbitrarily selecting one
  client or promoting personalized LoRA states to the primary result.
- Evaluation artifacts record protocol, split, server version, virtual time,
  and independent per-task metrics. Incompatible task metrics are not raw-averaged.

## Uncertain files intentionally kept

- `Codex项目规格_两项机制最小验证.md`: historical user-authored guidance; retained because it is not safe to treat user documentation as disposable.
- `literature-review.md` and `research-question-card.md`: pre-existing untracked user research notes; left untouched.
- `data/AFVLM_CM/`: all instructions, partitions, existing images, metadata, and utility material are user data and remain untouched.

## Engineering validation performed

- Python compile/import consistency.
- Ruff lint and formatting.
- YAML duplicate-key detection, inheritance, and 39 resolved experiment configs.
- Exact method registry and placeholder error.
- Shared asynchronous TrainPlan paths and system-profile/data matching.
- Immutable partition file count, byte count, and digest.
- The currently installed Python packages report no broken requirements; TOML parsing succeeds.
- No training, 7B loading, model/image download, commit, or push was performed.
