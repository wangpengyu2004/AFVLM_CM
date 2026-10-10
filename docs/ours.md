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

## 5. Bias-corrected per-task historical memory

For every task `q` and module `l`, the server stores a raw EMA `m[q,l]` and an
accepted-update count `n[q]`. Only an accepted update from its source task
changes that task's memory:

```text
m[q_k,l] <- history_beta m[q_k,l]
            + (1-history_beta) C[k,l]
n[q_k]   <- n[q_k] + 1
```

The default is `history_beta=0.9`. Memory is bias-corrected separately for
each task:

```text
m_hat[q,l] = m[q,l] / (1-history_beta^n[q])    if n[q] > 0
```

With uniform target-task prior `pi[q]=1/Q`, the protected server precision is:

```text
Omega[t,l] = base_precision
             + history_strength sum_q pi[q] a[q,t] m_hat[q,l]
```

The memory used for one arrival is the state before fusing that arrival. The
source task memory is updated only after fusion. Reliability and frequency do
not scale this EMA, so stale or infrequent tasks do not receive weaker future
protection solely because of their system behavior.

### Optional age-weighted protection of finished tasks

The current method template enables `history_age_enabled`. Only historical
precision synthesis changes; the raw EMA, its bias correction, functional
staleness, frequency correction and base-relative A/B increment remain the same.
When disabled (also the default if the flag is absent in a legacy profile),
`a[q,t]=1` and the original historical precision is recovered exactly.

```text
u          = number of successfully aggregated client updates so far
u_finish[q]= u when every planned (client_id, local_round) job of task q is resolved
age[q,t]   = u - u_finish[q]

a[q,t] = 1                                      if q is unfinished
a[q,t] = 1 + history_age_boost
             * (1-exp(-age[q,t]/history_age_time_scale))  otherwise
```

The exact virtual Plan is observed through `configure_schedule`; it is not
modified. A task retires only after all its clients' planned arrivals are handled,
including a rejected terminal arrival. Physical GPU completion, inactivity in
the frequency window, and finishing one client's last round do not retire a task.
Rejected or zero-weight arrivals do not advance `u`; accepted arrivals advance it
once. The current arrival reads the existing weight before advancing the counter.
A newly retired task starts at age zero and weight one.

Defaults are boost `1.0` and time scale `18.0` accepted updates. The multiplier
grows smoothly from one towards two. These are tunable initial settings, not
empirically established optimum values. Tasks with no accepted history still
contribute zero regardless of multiplier. All original task priors remain
`pi[q]=1/Q`; weighted priors are NOT renormalized and completed tasks are NOT
removed. The frequency window and its fixed-Q calculation are unchanged by
this extension. The time-scale value is independent of the frequency-window size.

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
alpha[k,l] = clip(alpha[k,l], 0, alpha_max)
```

The default `alpha_max=0.5`. One coefficient is shared by the complete A and B
of the same module:

```text
A[t+1,l] = A[t,l] + alpha[k,l] (A[k,l] - A[b_k,l])
B[t+1,l] = B[t,l] + alpha[k,l] (B[k,l] - B[b_k,l])
```

`b_k` is the exact immutable snapshot downloaded when this client started,
not the current global model or an inferred version. The server adds the
client's local-training increment to its current A/B tensors. It does not
interpolate toward the stale endpoint or subtract a fraction of the server
drift. A fresh update (`t=b_k`) is algebraically identical to endpoint mixing.
Sensitivity, functional reliability, frequency weight, alpha clipping, and
the post-success history/window updates are otherwise unchanged.

The history precision now damps the added increment; it is not an exact
precision-weighted fusion of two model endpoints. An unconstrained quadratic
surrogate in parameter movement `s_l` gives the same uncapped coefficient:

```text
min_s P_client[k,l] ||s - (theta[k,l]-theta[b_k,l])||^2
      + P_server[t,l] ||s||^2
```

The actual coefficient also includes `eps` and the configured cap. This is a
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
    history_lambda: 1.0
    history_strength: 1.0
    history_age_enabled: true
    history_age_boost: 1.0
    history_age_time_scale: 18.0
    base_precision: 1.0
    alpha_max: 0.5
    eps: 0.000000000001
    min_update_energy: 0.0000000000000001
```

There are intentionally no parameters for rank sensitivity or sensitivity
power/quantile compression.

## 9. DDP and optimizer-step semantics

`ours` remains DDP-capable. All visible GPUs train the same client. With eight
GPUs, per-device batch 16, and gradient accumulation 4, one optimizer step has
effective batch `8 * 16 * 4 = 512`.

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

Every arrival writes schema-version-4 `ours_diagnostics`, including:

- raw and final module-sensitivity summaries;
- per-module sensitivity and observation count;
- module-functional server drift and local update energy;
- relative staleness, reliability, and gamma;
- pre-update window counts, frequency weight, and post-success window state;
- per-module signed gate mean, direction, and mean absolute direction;
- per-module alpha;
- `aggregation.rule = base_relative_delta`, identifying this aggregation revision;
- raw and bias-corrected task memory and accepted-update counts;
- temporal-history enable flag, boost, time scale and accepted-update clock;
- each task's pending Plan jobs, completion counter, age and protection multiplier,
  before/after the arrival;
- per-module historical precision both with and without temporal weighting;

`tools/export_ours_diagnostics.py` exports update-level and module-level CSV
files. Both include `aggregation_rule`. Legacy events without a rule label
export an empty field, rather than being relabelled as incremental updates.
Temporal fields in legacy events also remain empty. The update CSV contains JSON
maps `history_task_weights_before/after`, `history_task_ages_before/after`,
`history_tasks_finished_at_before/after`, and `history_pending_updates_before/after`;
the module CSV adds `historical_precision_unweighted_before/after` for comparison
with actual `historical_precision_before/after`.
There is no rank-level CSV because Module-Gate does not produce or upload
rank sensitivity.

Module-Gate uses state schema version 4. Older checkpoints are rejected because
they cannot reconstruct the ordered accepted-task frequency window.
This aggregation revision keeps the same memory/checkpoint data layout but
changes the update semantics. Do not continue an old endpoint-mixing run as
the same experiment; use a separate output/profile for the new variant.
Temporal-history runs additionally checkpoint `history_age_state` (sub-schema 1):
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

For this temporal-protection revision, create a separate profile from the current
templates. If data, seed and scheduling/training settings match your existing
profile, reuse its identical Plan (replace `YOUR_EXISTING_PROFILE`):

```bash
python tools/generate_system_profiles.py \
  --profile module_gate_age_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from YOUR_EXISTING_PROFILE
bash scripts/run_one.sh ours 2 module_gate_age_e1_bs4_ga4_r10_s42
```

The profile name is a label, not a parameter override. New snapshots read current
`configs/` values, so check them against the old experiment. If data or the local
training budget changed, omit `--reuse_plans_from`. Old immutable profiles are
unchanged and do not automatically activate temporal protection. For an ablation,
set `history_age_enabled: false` in `configs/methods/ours.yaml` and create another
new profile with the same compatible Plan.
