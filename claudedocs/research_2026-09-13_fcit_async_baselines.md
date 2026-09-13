# FCIT fixed-task asynchronous FL: source audit and baseline selection

Date: 2026-09-13

## Executive summary

The project's event protocol and FLGo FedAsync are in the same asynchronous
family but are not identical. FLGo periodically samples idle clients and
aggregates all models received in the same clock tick after staleness-aware
interpolation. This project continuously cycles every client, processes a
deterministic ordered event stream, and enforces equal upload quota. The latter
is the better primary protocol for the stated causal question because it holds
total participation fixed while varying task-correlated latency. It should be
reported under its own name, not as an exact FLGo reproduction.

FCIT's official Task-Heterogeneous training script uses LLaVA-1.5-7B with CLIP
ViT-L/14-336, LoRA rank 64/alpha 128, a separately trained multimodal projector,
one local epoch, 10 communication rounds, and 10% client selection. The local
adapter added here mirrors the backbone and trainable parameter classes through
the Transformers-converted LLaVA checkpoint. It intentionally excludes FCIT's
continual task sequence and expert-routing algorithm.

The runnable baseline suite contains eight controls: AsyncSGD, polynomial
staleness decay, FedAvg, FedAsync, FedBuff, FedAdam, FedYogi, and a clearly
labeled FedCompass simulator realization. FedQS (NeurIPS 2025), FedASMU (AAAI
2024), and AsyPFL (ACML 2023) are relevant follow-on comparisons, but their full
algorithms require protocol or persistent personalization state not present in
the current single-global-state runtime.

## Evidence and decisions

| Question | Evidence | Decision | Confidence |
|---|---|---|---|
| Which FCIT model? | Official `task_1.sh` specifies `llava-v1.5-7b`, CLIP-L/14-336, LoRA 64/128, projector LR 2e-5 | Add `llava15_fcit` with those tuning defaults | High |
| Is the runtime the same as FLGo? | Official FLGo `fedasync.py` samples idle clients by `period` and averages simultaneous interpolated models | Document non-equivalence; keep equal-quota event runtime | High |
| Which classic baselines? | FedAvg, FedAsync, FedBuff and FedOpt are canonical controls for sync, immediate async, buffered async and heterogeneous server optimization | Make all four families runnable | High |
| Which recent systems baseline? | FedCompass (ICLR 2024) assigns varying local work to align heterogeneous client arrivals and groups updates | Implement its core virtual-scheduler mechanisms as `fedcompass_sim`, not an official reproduction | Medium |
| Which newer work should be acknowledged? | FedQS is a NeurIPS 2025 semi-async method; FedASMU adds dynamic server weighting plus mid-training fresh-model pulls | Document as next-phase exact integrations, not approximate baselines | High |
| What about client adaptation? | AsyPFL explicitly maintains personalized models under irregular/asynchronous clients | Keep global aggregation as primary axis; add equal-budget post-global personalization in a later protocol | Medium-high |

## Baseline rationale

1. **AsyncSGD/raw-delta** is a mechanism ablation, not a claimed named-paper
   reproduction. It shows the cost of applying every arrival without staleness
   protection.
2. **Polynomial staleness decay** isolates scalar-age correction. It is crucial
   because the proposed research argues that scalar staleness cannot represent
   task-conditioned harm.
3. **FedAvg** provides the synchronous performance/time reference and uses
   client sample counts under quantity skew.
4. **FedAsync** performs model interpolation, not naive delta scaling.
5. **FedBuff** separates the effect of buffering from immediate aggregation.
6. **FedAdam** and **FedYogi** test whether server adaptivity alone addresses
   extreme task heterogeneity.
7. **FedCompass-sim** tests whether computing-power-aware workload assignment
   and grouped arrivals remove the observed task-delay effect.

Primary reported comparisons should keep model, data pools, client upload
quota, seeds, local optimizer, and evaluation checkpoints fixed. Synchronous
methods should additionally report virtual wall-clock time because equal update
counts alone hide their barrier cost.

## Sources

- FCIT paper: https://www.openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html
- FCIT official repository and script: https://github.com/Ghy0501/FCIT and https://github.com/Ghy0501/FCIT/blob/main/scripts/LLaVA/Train_FCIT_task_het/task_1.sh
- FLGo official FedAsync: https://github.com/WwZzz/FLGo/blob/main/flgo/algorithm/fedasync.py
- FedAvg: https://proceedings.mlr.press/v54/mcmahan17a.html
- FedAsync: https://arxiv.org/abs/1903.03934
- FedBuff: https://proceedings.mlr.press/v151/nguyen22b.html
- Adaptive Federated Optimization: https://research.google/pubs/adaptive-federated-optimization/
- FedCompass: https://proceedings.iclr.cc/paper_files/paper/2024/file/a9f3457fa97f106f1756885237787789-Paper-Conference.pdf
- FedCompass official code: https://github.com/APPFL/FedCompass
- FedASMU: https://ojs.aaai.org/index.php/AAAI/article/view/29297
- FedQS: https://papers.nips.cc/paper_files/paper/2025/hash/7d3f1540bb7af35813578db81d03b0dd-Abstract-Conference.html
- AsyPFL: https://proceedings.mlr.press/v189/ma23b.html

## Limitations

- No 7B training was run locally because seven of eight image collections are
  absent and the current Windows environment is not a target CUDA/bitsandbytes
  training host.
- `fedcompass_sim` has known virtual client speeds and omits APPFL's online
  estimator and deployment transport. Results must use that exact method label.
- FCIT's expert routing is part of its continual algorithm rather than the
  LLaVA backbone; it is intentionally excluded from the fixed-task benchmark.
