# Proposed method: Module-Gate Sensitivity-Aware Asynchronous LoRA Updates

The registered method `ours` implements the Module-Gate design for fixed-task
asynchronous federated LLaVA instruction tuning. Its main line is:

```text
module importance + functional staleness + task-specific historical memory
+ sliding-window task-frequency correction
```

It does not add a trainable gate, change the PEFT structure, perform SVD, or
align LoRA ranks across clients. Only
LoRA A/B tensors and one sensitivity scalar per LoRA module are communicated.

## 1. LoRA scope and validation

For module `l`, the effective LoRA update is:

```text
DeltaW_l = scaling_l B_l A_l
A_l: [rank, in_features]
B_l: [out_features, rank]
```

The client and server validate module names, A/B state keys, shapes, rank, and
LoRA scaling. Frozen LLaVA and vision-tower parameters are never included in
the federated state.

## 2. Client Module-Gate sensitivity

For an imaginary scalar gate `z_l`:

```text
DeltaW_l(z_l) = scaling_l z_l B_l A_l
```

At one real optimizer step, after gradient accumulation and DDP gradient
synchronization, the implementation reads the existing gradients:

```text
h_A[l] = <grad(A_l), A_l>_F
h_B[l] = <grad(B_l), B_l>_F
h[l]   = 0.5 (h_A[l] + h_B[l])
```

This is the derivative with respect to the virtual module gate at `z_l=1`.
The operation runs under `torch.no_grad()`, does not change `.grad`, does not
call backward again, and introduces no optimizer parameter.

The first `sensitivity_warmup_steps` optimizer steps are skipped. For the
remaining observed steps, the raw statistic is the mean absolute gate:

```text
C_raw[k,l] = mean_m |h[k,l,m]|
```

Absolute value is applied before averaging, so positive and negative steps do
not cancel. Unlike a squared statistic, it does not quadratically amplify an
occasional large gate response.

For diagnostics only, the client also retains the signed sum and reports:

```text
signed_gate_mean[k,l] = mean_m h[k,l,m]
gate_direction[k,l]   = -sum_m h[k,l,m]
                         / (sum_m |h[k,l,m]| + eps)
```

`gate_direction_abs_mean` is the mean absolute direction over observed
modules. These values never enter sensitivity, staleness, memory, or fusion.

The current version performs no sqrt transform, quantile cap, or sensitivity
clipping. It first normalizes across all modules of the same client and then
mixes in a uniform prior:

```text
C_norm[k,l] = C_raw[k,l] / (mean_j C_raw[k,j] + eps)
C[k,l]      = (1-eta) + eta C_norm[k,l]
```

The default `eta=0.5` gives a floor of 0.5 while preserving an approximately
unit client-wide mean. If no module has a valid post-warm-up observation, all
module sensitivities fall back to one and the diagnostic validity flag is
false.

## 3. Complete-module functional distance

Parameter distance in A/B space is not used because LoRA has scale ambiguity.
For two versions `x` and `y` of one module:

```text
d_l(x,y) = ||scaling_l (B_x A_x - B_y A_y)||_F^2
           / (out_features_l * in_features_l)
```

The implementation never constructs the dense `B @ A`. It uses rank-by-rank
Gram products:

```text
||B_x A_x||_F^2
  = tr[(B_x^T B_x)(A_x A_x^T)]

<B_x A_x, B_y A_y>_F
  = tr[(B_x^T B_y)(A_y A_x^T)]
```

This includes cross-rank terms while constructing only rank-by-rank
intermediates. Small negative distances caused by floating-point cancellation
are clamped to zero.

## 4. Functional staleness and reliability

Let `b_k` be the client's immutable base snapshot, `k` its local endpoint, and
`t` the current server state:

```text
D[k,t] = sum_l C[k,l] d_l(t,b_k)
U[k]   = sum_l C[k,l] d_l(k,b_k)

delta[k,t] = sqrt(D[k,t] / (U[k] + eps))
rho[k,t]   = exp(-gamma delta[k,t])
```

The default is `gamma=1.0`; `gamma=0.5` is a direct configuration ablation.
An update with `U[k] < min_update_energy` is rejected as functionally empty,
even if A and B changed through an equivalent LoRA reparameterization.

## 5. First-observation per-task historical memory (current template)

The current template uses `history_update_mode: first_observation_ema` and
`base_precision: 0.0`. Every unobserved task/module starts with H=1. The first
successfully accepted upload REPLACES that task's prior, and subsequent accepted
uploads use ordinary EMA. This is not zero-initialized EMA with bias correction:

```text
H[q,l] = 1                                  while n[q] == 0
H[q_k,l] <- C[k,l]                           on its first accepted upload
H[q_k,l] <- beta H[q_k,l] + (1-beta) C[k,l]    on later accepted uploads
n[q_k]   <- n[q_k] + 1
```

The default `history_beta=0.9` is unchanged. Do NOT divide H by `(1-beta^n)`.
Each task's accepted count is independent; a rejected upload changes neither
its unit prior nor its EMA/count/window. Other tasks' memories are not updated.

With uniform task prior `pi[q]=1/Q` and default history_strength=1:

```text
Omega_base[t,l] = history_strength sum_q pi[q] H[q,l]
```

There is no outer fixed1. Initially all six H values are1, so Omega_base=1;
only the arriving task's placeholder is replaced by real sensitivity. For
example C=1,2,3 produces H=1,1.1,1.29, not corrected-EMA 1,1.526,2.070.
The initial observation has weight beta^(n-1); this statistical change does
not promise better accuracy or bitwise equivalence with earlier experiments.

The memory used for one arrival is the state before fusing that arrival. The
source task memory is updated only after fusion. Reliability and frequency do
not scale this EMA, so stale or infrequent tasks do not receive weaker future
protection solely because of their system behavior.

Compatibility: a profile missing `history_update_mode` retains
`zero_initialized_corrected_ema`: stored m starts0, H=m/(1-beta^n) for n>0,
unobserved H=0, plus its original positive base_precision (default1).
Do not add the new flag to an old checkpoint: its memory has different semantics.
New history requires base_precision=0 and explicit mode in new snapshots.

### Finished-task selected-layer cumulative protection (current template)

The template enables `history_age_enabled` and `history_age_mode: selected_layers_sum`.
Only the composition of finished-task extra demands changes from MAX to SUM.
Selection, frozen histories, age clock and fixed task priors remain unchanged.
Functional staleness, frequency, age coefficients and LoRA increment are unchanged.
All configured clients must finish all unique planned jobs by successful incorporation.
Rejected jobs remain pending and do not retire a task; duplicate accepted jobs fail.
After the last successful update's EMA, freeze its effective `H_end` and counter.
For first_observation_ema this is the stored H without any bias correction;
legacy profiles still freeze their corrected EMA.

```text
u          = number of successfully aggregated client updates so far
u_finish[q]= u when every planned (client_id, local_round) job of task q is accepted
age[q,t]   = u - u_finish[q]
E[q,m]     = max(H_end[q,m] - 0.5, 0)
I[q,layer] = mean_QKVO(E[q,layer,projection])
p[q,layer] = I[q,layer] / sum_layers I[q,layer]
mask[q,layer] = p[q,layer] > 1/L + 1e-8
b[q,t]     = history_age_boost * (1-exp(-age[q,t]/history_age_time_scale))
P[q,m]     = pi[q] * b[q,t] * mask[q,layer(m)] * H_end[q,m]
Omega[t,m] = Omega_base[t,m] + history_strength * sum_finished_q P[q,m]
```

L is inferred from complete decoder Q/K/V/O metadata (32 for LLaVA1.5-7B), not
chosen from observed experimental layer lists. The floor is `1-uniform_mix`,
currently0.5, subtracted only for selection. Zero/indistinguishable scores select
no layers. H_end/mask are immutable after completion. No fixed K, L/K compensation,
normalization of extra mass or additive full-layer age term is used.
Four projections retain their own H_end. Each selected task contributes its own
demand; overlapping tasks now add protection. Keep pi=1/6 for the six-task setup,
not 1/(number of finished tasks). Empty sums are zero. Base history still sums
all tasks. Selection union and positive-extra support differ.

Default boost2/scale18 are initial experimental settings, not proven optimal.
Current aggregation reads pre-update age: first upload after completion has age0,
second has age1. No parameter anchor, synthetic upload or extra server update.
Sparsification is not budget-matched relocation. Sum is at least as protective as
max for identical histories/ages; it can reduce later alpha. This is not evidence
of better accuracy without a new run.

Compatibility: absent `history_age_mode` retains the original `all_modules_sum`
rule for old profiles, including its old resolved-job retirement semantics.
Explicit `selected_layers_max` retains MAX and its original source-task logs.
The strict successful-completion and frozen selection rules apply to both
selected-layer modes; do not stack the old all-module age term with either one.
Set `history_age_enabled: false` for base-only protection. Old profiles/checkpoints
are not silently rewritten. See [full implementation and logs](ours_selected_layer_protection.md).

No parameter anchor, old-update replay, extra forward/backward or client upload
is introduced. Increased Omega only damps subsequent movements in important
modules; it cannot restore already-lost knowledge or distinguish harmful updates
from beneficial transfer. Excessive protection may slow the remaining tasks.
Compare disabled, zero-boost and positive-boost runs under the same Plan and
training budget; tune on validation, never by selecting a test-set result.

## 6. Sliding-window task-frequency correction

The server retains task labels for the latest `W=18` successfully aggregated
updates. Rejected updates are not inserted. Before the current update is
aggregated, let `n[q,t]` be its task count in the current window and `Q` the
number of tasks:

```text
w[q,t] = clip((((|window|/Q)+1)/(n[q,t]+1))^frequency_power,
              frequency_min, frequency_max)
```

Defaults are power `0.5` and bounds `[0.5, 1.5]`. The current task enters the
FIFO window only after successful fusion, so its own label cannot change the
weight used for that fusion. The window is method state and is checkpointed.

## 7. Module-wise base-relative LoRA increment aggregation

Client precision and server protection are:

```text
P_client[k,l] = rho[k,t] w[q_k,t] C[k,l]
P_server[t,l] = history_lambda Omega[t,l]
```

```text
alpha[k,l] = P_client[k,l]
             / (P_client[k,l] + P_server[t,l] + eps)
```

The current template uses `history_lambda=1` and `alpha_max: null`: the exact
default is alpha=rwC/(rwC+Omega+eps), with no additional0.5 clipping. For valid
non-negative precisions the fraction itself lies in[0,1]; it is NOT N/Omega.
Current C stays in the numerator and denominator, not its reciprocal or a
constant replacement. C denotes this arriving client's normalized Module-Gate
sensitivity, not task history H. Its128-module mean is approximately1, but
individual modules differ. Positive C expresses importance, not proven update quality.

Old snapshots retaining numeric alpha_max still clip to that limit. Missing
alpha_max retains the historical0.5 default; only explicit YAML null disables
clipping. One coefficient is shared by the complete A and B of the same module:

```text
A[t+1,l] = A[t,l] + alpha[k,l] (A[k,l] - A[b_k,l])
B[t+1,l] = B[t,l] + alpha[k,l] (B[k,l] - B[b_k,l])
```

`b_k` is the exact immutable snapshot downloaded when this client started,
not the current global model or an inferred version. The server adds the
client's local-training increment to its current A/B tensors. It does not
interpolate toward the stale endpoint or subtract a fraction of the server
drift. A fresh update (`t=b_k`) is algebraically identical to endpoint mixing.
Sensitivity, functional reliability, frequency weight, and
the post-success history/window updates are otherwise unchanged.

The history precision now damps the added increment; it is not an exact
precision-weighted fusion of two model endpoints. An unconstrained quadratic
surrogate in parameter movement `s_l` gives the same uncapped coefficient:

```text
min_s P_client[k,l] ||s - (theta[k,l]-theta[b_k,l])||^2
      + P_server[t,l] ||s||^2
```

The actual coefficient includes `eps`; a numeric cap is used only if configured. This is a
step-size interpretation, not a convergence or task-preservation guarantee.
The increment remains stale and can conflict with other tasks. Factor-space
A/B increments are not identical to adding a dense `Delta(BA)`; they preserve
the existing LoRA rank and PEFT state interface without SVD.

## 8. Configuration

`configs/methods/ours.yaml` contains:

```yaml
method:
  name: ours
  params:
    sensitivity_warmup_steps: 1
    sensitivity_uniform_mix: 0.5
    gamma: 1.0
    frequency_window_size: 18
    frequency_power: 0.5
    frequency_min: 0.5
    frequency_max: 1.5
    history_beta: 0.9
    history_update_mode: first_observation_ema
    history_lambda: 1.0
    history_strength: 1.0
    history_age_enabled: true
    history_age_mode: selected_layers_sum
    history_age_boost: 2.0
    history_age_time_scale: 18.0
    base_precision: 0.0
    alpha_max: null
    eps: 0.000000000001
    min_update_energy: 0.0000000000000001
```

There are intentionally no parameters for rank sensitivity or sensitivity
power/quantile compression.

## 9. DDP and optimizer-step semantics

`ours` remains DDP-capable. All visible GPUs train the same client. The profile
generator auto-detects visible GPUs and freezes reference_ddp metadata; users
need not put these fields in base.yaml. With reference8 the budget is128.
Eight ranks preserve existing B4/G4; one-GPU methods
use B4/G32 and the same reference shard windows. Original padded sampler tails,
micro-batch loss means and rank averaging are retained, without new token
weighting. Hooks still run once per optimizer step after accumulation/sync.
Legacy profiles retain per-device semantics: old eight-rank B4/G4 runs have
budget128, but old one-GPU B4/G4 runs have budget16.
See [batch fairness and migration](batch_fairness.md).

The training hook order is:

```text
accumulated forward/backward
-> final DDP gradient synchronization
-> Module-Gate read under no_grad
-> optimizer.step
-> zero_grad
```

The current LLaVA training path does not use GradScaler. If AMP scaling is
added later, gradients must be unscaled before the Module-Gate hook. The
virtual TrainPlan, base version, planned arrival, and aggregation order are
unchanged by this method revision.

## 10. Diagnostics and checkpoints

First-observation history writes schema-version-6 `ours_diagnostics`;
older selected-layer corrected EMA remains5, original legacy mode remains4. Fields include:

- raw and final module-sensitivity summaries;
- per-module sensitivity and observation count;
- module-functional server drift and local update energy;
- relative staleness, reliability, and gamma;
- pre-update window counts, frequency weight, and post-success window state;
- per-module signed gate mean, direction, and mean absolute direction;
- per-module alpha;
- `aggregation.alpha_rule=normalized_precision_ratio`, configured alpha_max
  (null means uncapped), and client_precision_by_module=rwC;
- `aggregation.rule = base_relative_delta`, identifying this aggregation revision;
- stored/effective task memory and accepted-update counts (equal in the new mode);
- history_update_mode, history_beta, history_bias_correction=false,
  unobserved_task_sensitivity=1 and base_precision=0 for the new template;
- temporal-history enable flag, boost, time scale and accepted-update clock;
- each task's pending Plan jobs, completion counter, age and protection multiplier,
  before/after the arrival;
- per-module historical precision both with and without temporal weighting;
- immutable task-end H, layer scores/shares/masks and client completion counts;
- selected-layer union and actual positive-extra support;
- summed age mode: each finished task's extra precision, all positive contributors
  and their count per module; old max mode retains its max-source field instead;

`tools/export_ours_diagnostics.py` exports update-level and module-level CSV
files. Both include `aggregation_rule`. Legacy events without a rule label
export an empty field, rather than being relabelled as incremental updates.
Temporal fields in legacy events also remain empty. The update CSV contains JSON
maps `history_task_weights_before/after`, `history_task_ages_before/after`,
`history_tasks_finished_at_before/after`, and `history_pending_updates_before/after`;
the module CSV adds `historical_precision_unweighted_before/after` for comparison
with actual `historical_precision_before/after`.
Selected-layer records additionally export extra precision and layer. Sum-mode
records list all contributors and per-task demands; max-source CSV columns are
empty for sum mode, and new sum-only columns are empty for older logs.
`ours_task_completion_diagnostics.csv` is created when task-end snapshots are present.
There is no rank-level CSV because Module-Gate does not produce or upload
rank sensitivity.

First-observation EMA uses state schema5 with an explicit history-mode marker.
Legacy zero-initialized corrected EMA keeps schema4. Loading restores the saved
mode/parameters, not a new statistical reinterpretation of old stored values;
schema/mode mismatch and invalid unobserved priors are rejected. Pre-v4 states
cannot reconstruct the ordered accepted-task frequency window. Use a separate
output/profile and a fresh experiment for the new history variant; do not rename
old results, remove bias correction on an old memory or continue an old endpoint-
mixing run as the same experiment.
Temporal-history runs additionally checkpoint `history_age_state` (legacy sub-schema1;
selected-layer sub-schema2 additionally requires frozen H/masks):
the exact Plan jobs, resolved jobs, successful-update counter and task retirement
times. Loading it validates completion consistency. A legacy checkpoint without
these fields is accepted only with temporal history disabled; it is never assigned
invented completion times. This does not add interrupted-training resume support.

## 11. Running

Create a new immutable profile so an old Rank-Gate snapshot is not reused. The
current batch size is 4, so generate a new virtual system and TrainPlan rather
than reusing a profile created for a different batch size:

```bash
python tools/generate_system_profiles.py \
  --profile module_gate_delta_grounding8k_e1_bs4_ga4_r10_s42
```

Run on all visible GPUs:

```bash
bash scripts/run_one.sh ours 2 module_gate_delta_grounding8k_e1_bs4_ga4_r10_s42
```

Use setting `5` or `10` for the corresponding existing AFVLM-CM partition.
Do not resume an old Rank-Gate output directory or checkpoint.

Changing only the aggregation formula does not require a different system
profile or TrainPlan. When the data partition and scheduling/training settings
are unchanged, add `--reuse_plans_from <compatible_profile>` to the generation
command to copy the identical Plan into the new profile/output. Do not reuse
a Plan from the old 64.5k/72k partition with the new Grounding-8k/68k data, or
from a different batch/epoch configuration. Existing immutable configs and
Plans are never rewritten by this aggregation change.

For this cumulative-protection revision, create a separate profile from the current
templates. If data, seed and scheduling/training settings match your existing
profile, reuse its identical Plan (replace `YOUR_EXISTING_PROFILE`):

```bash
python tools/generate_system_profiles.py \
  --profile module_gate_sum_firstema_nocap_b2_t18_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from YOUR_EXISTING_PROFILE
bash scripts/run_one.sh ours 2 module_gate_sum_firstema_nocap_b2_t18_e1_bs4_ga4_r10_s42
```

The profile name is a label, not a parameter override. New snapshots read current
`configs/` values, so check them against the old experiment. If data or the local
training budget changed, omit `--reuse_plans_from`. Old immutable profiles are
unchanged and do not automatically activate temporal protection. For an ablation,
set `history_age_enabled: false` in `configs/methods/ours.yaml` and create another
new profile with the same compatible Plan.
