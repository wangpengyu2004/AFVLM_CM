# Baseline Implementation Notes

All methods exchange only LLaVA LoRA/configured trainable tensors. Frozen
LLaVA-1.5-7B parameters are never placed in an update or federated checkpoint.
For fair comparison, every client job creates a fresh AdamW optimizer; local
optimizer moments do not persist across jobs.

For all ordinary methods, `training.local_epochs` is the sole local-work
controller. The TrainPlan's `local_steps` value is derived from client sample
count, batch size, gradient accumulation, and local epochs; it estimates virtual
duration but does not truncate training. FedCompass is the intentional exception
because bounded local-iteration allocation is part of its scheduler.

Primary evaluation is always the current server federated state at a common
server-update checkpoint and the final server state after method finalization.
Local-only is the sole exception because no server model exists. Pilot's
server-global evaluation uses task routing with a task-wise mean of its
server-held client visual adapters; personalized LoRA states are not used as
the primary score.

## Reference and classical FL

- **Local-only:** persistent per-client LoRA states, no global mutation.
- **FedAvg:** synchronous sample-count weighted local-state average.
- **FedProx:** FedAvg plus `mu/2 ||theta-theta_snapshot||²` over federated
  trainables only.
- **FedAdam:** synchronous sample-count weighted client deltas with bias-
  corrected server Adam moments and configurable server learning rate.

## Asynchronous FL

- **FedAsync:** each job records `base_version`; arrival applies
  `theta <- theta + alpha*s(staleness)*(theta_client-theta)` with constant,
  polynomial, or hinge decay.
- **FedBuff:** maintains an arrival buffer, aggregates client deltas with
  optional staleness weights, and explicitly flushes or errors on a residual
  buffer.

## System-heterogeneity methods

### FedCompass (ICLR 2024)

The shared scheduler derives bounded `local_steps_k` from each client's
epoch-derived nominal work, persisted speed, and base training cost, then groups near-simultaneous completions for
semi-asynchronous aggregation. It is therefore a compute-aware scheduler, not
a renamed FedAsync decay. The implementation reuses AFVLM-CM's virtual clock
and adapts the optimized state from a full model to LoRA.

Paper: [FedCompass: Efficient Cross-Silo Federated Learning on Heterogeneous Client Devices Using a Computing Power-Aware Scheduler](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a9f3457fa97f106f1756885237787789-Abstract-Conference.html).

### FedASMU (AAAI 2024)

`methods/fedasmu.py` implements dynamic per-client staleness factor `xi`, the
bounded server interpolation coefficient, and a configurable mid-local
interpolation with fresher global state. Online loss-improvement signals adapt
the exposed server/local coefficients. The deterministic refresh point and
online coefficient rule replace the original full device request/RL control
path, so this is an explicit LoRA/AFVLM-CM adaptation rather than an exact
framework reproduction.

Paper: [FedASMU: Efficient Asynchronous Federated Learning with Dynamic Staleness-Aware Model Update](https://ojs.aaai.org/index.php/AAAI/article/download/29297/30446).

### MasFL and AdaMasFL (ICML 2025)

Both variants maintain per-client controls, a global control, and historical
descent momentum. MasFL corrects local gradients with those controls and uses
the two-level momentum recursion. AdaMasFL additionally normalizes the local
direction and aggregates observed normalized displacement. These are not
generic server-momentum aliases; every vector is the federated LoRA state.

Paper: [Momentum-Driven Adaptivity: Towards Tuning-Free Asynchronous Federated Learning](https://proceedings.mlr.press/v267/yan25f.html).

## Task-heterogeneous multimodal methods

### Pilot (AAAI 2025)

The Pilot-only model wrapper contains task/client visual adapters, CT-MoA
cross-task routing, difference/load-balance/router losses, task-wise visual
aggregation, and nearest-client personalized text-LoRA aggregation. Other
methods retain the common LLaVA model. The current implementation jointly
optimizes Pilot losses in one run instead of executing a separate paper-style
pretraining stage; the immutable AFVLM-CM partition remains unchanged.

Paper: [Pilot: Building the Federated Multimodal Instruction Tuning Framework](https://ojs.aaai.org/index.php/AAAI/article/download/35476/37631).

### UniFed-LoRA (CVPR Workshops 2026)

The server hypernetwork conditions LoRA gates on configurable task semantics,
modality, architecture, layer, and module descriptors. Client indices are not
descriptors. In this experiment `backbone heterogeneity = disabled` and
`task heterogeneity = enabled`; adding new architecture/modality descriptor
dimensions does not require changing the dataset.

Paper: [UniFed-LoRA: Exploiting Semantic Task Correlation for Heterogeneous Multimodal Federated Fine-Tuning](https://openaccess.thecvf.com/content/CVPR2026W/FedVision/html/Milasheuski_UniFed-LoRA_Exploiting_Semantic_Task_Correlation_for_Heterogeneous_Multimodal_Federated_Fine-Tuning_CVPRW_2026_paper.html).

## Proposed method

`ours` implements sensitivity-aware asynchronous LoRA consolidation. Clients
estimate module/rank/Adam-v LoRA sensitivity during normal backward passes;
the server uses functional staleness, task-balanced historical sensitivity,
and group-wise precision fusion. The default paper setting is Module-Gate.
It is excluded from baseline batches because it is the proposed method, not
because it is unavailable. Full equations and commands are in `docs/ours.md`.
