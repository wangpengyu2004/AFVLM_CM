# C2-AFCL: fixed-task LoRA adaptation

Source: **Cross-task Calibration for Asynchronous Federated Continual Learning**,
ICML 2026, [official proceedings](https://proceedings.mlr.press/v306/li26gx.html).
Implementation follows Algorithm 1, Section 4.1 and Section 4.2, Eqs. (4)--(6).
This is **C2-AFCL-LoRA (fixed-task adaptation)**, not exact reproduction of the
ResNet/class-incremental or domain-incremental experiments or their guarantees.

## What is retained

Let `u = theta_local - theta_download`, flattened over the common federated
trainable state. The frozen LLaVA backbone is never transmitted/projected.

1. **Intra-client calibration (Section 4.1)**: cache the last RAW update by
   `(client_id, task)`. Compute `delta = u - previous_raw_u`; missing cache means
   zero previous update. Replace the cache with `u`, not `delta` or a safe update.
   Cache advances on virtual reception, never on early physical completion.
2. **Inter-task calibration (Eqs. 4--5)**: before accepting the current update,
   form columns from other tasks' already incorporated historical mean RAW
   updates. Compute a thin SVD and keep the top `d` nonzero left singular vectors
   `U_d`. The safe update is `delta_safe = delta - U_d @ (U_d.T @ delta)`.
   There is no layer/module-wise independent projection and no `D x D` matrix.
3. **Aggregation (Eq. 6)**: average safe updates in a same-task buffer and apply
   `theta_now += server_lr * mean(delta_safe)`. Default buffer size1 means each
   arrival is aggregated separately. Larger buffers never mix tasks. All residual
   task buffers are flushed at finalization, oldest pending task first.
4. **Online history**: after each aggregation, update that task's equal-weight
   running arithmetic mean of its incorporated RAW updates. This is neither an
   EMA nor a mean of the projected/calibrated updates. Using an online mean is
   an explicit interpretation of the paper's task-average vector, whose exact
   online window policy is unspecified. Current task is excluded regardless.

Every projection uses only causal PRE-arrival history. Safe updates are computed
at their virtual arrival and retained until flush; they are not reprojected using
later information. Accepted-update counters and global versions are handled by
the existing server. A zero safe update still completes a positive-learning-rate
aggregation; no artificial raw-delta fallback is inserted.

## Adaptations and unresolved source details

- All six AFVLM_CM tasks exist from the beginning and client task identity is
  fixed. The paper's **previous sequential tasks** become **other fixed tasks
  with previously incorporated updates**. No invented task order, continual
  repartition or new client task stream is introduced.
- Coordinates are the shared LoRA A/B factors (plus explicitly enabled common
  trainables), not full ResNet weights or a dense `BA` functional direction.
  Factor-space orthogonality is parameterization dependent; it is not a guarantee
  of functional noninterference between tasks.
- Paper Section3/Eq2 describes immediate model interpolation; Section4/Eq6 uses
  same-task buffered calibrated increments. We prioritize the method-specific
  Eq6 and support both `K=1` immediate and `K>1` same-task-buffered execution.
  `K` is the actual buffer occupancy, including residual flushes, NOT the total
  client count. This protocol choice is disclosed, not claimed as author code.
- Eq3 mentions an unspecified continual regularizer. Its importance estimator
  and numeric settings are not defined in Algorithm1/Section4. We do not invent
  Fisher/EWC/replay or a new loss: fixed-task clients use the shared instruction
  CE, AdamW, epochs and batch budget. Hence this is the two-stage calibration
  baseline specialized to this project, not every unspecified FCL component.
- No extra staleness decay, sample-size weight, gradient refresh, sensitivity or
  task-frequency compensation is added. Staleness is recorded and naturally
  reflected in the Plan snapshots, not given an invented scalar multiplier.
- The paper does not specify numeric server mixing rate / retained SVD rank in
  the inspected algorithm/configuration sections. The values below are **project
  defaults**, not verified official defaults. Tune on validation under a disclosed
  equal search budget, never on the final/test set.

## Parameters

Edit `configs/methods/c2afcl.yaml` before creating a new immutable profile:

| Parameter | Default | Meaning |
|---|---:|---|
| `server_lr` | 0.5 | Eq6 server increment coefficient, in `(0,1]` |
| `subspace_rank` | 5 | Maximum number of protected singular directions; at most five other tasks here |
| `buffer_size` | 1 | Per-task number of arrivals before one aggregation;1 means no waiting |
| `svd_rtol` | 1e-6 | Discard singular values below `max(eps, largest_singular * svd_rtol)` |
| `eps` | 1e-12 | Absolute numerical singular-value tolerance |

No new optimizer/loss/model parameters are required. Under the existing
`runtime.executor_policy: capability`, this method selects **client_ddp**:
all visible cards train one client with the same existing local/DDP code. A
fixed client_parallel policy is also supported; the shared reference-batch
fairness implementation remains authoritative. Physical execution does not
change virtual download versions or arrival order.

## Run with the same system conditions as other baselines

Old profiles are immutable and do not contain `c2afcl.yaml`. Create a NEW profile
using the currently edited templates; reuse an old profile's Plans only if its
data/client counts, epochs, batch/accumulation, rounds and system factors match.
The generator reads CURRENT base/model/method templates, not old method configs.
Check shared resolved training settings against the reference experiment if any
templates have changed since its creation. This does not modify the old profile.

On the server, replace `YOUR_OLD_PROFILE` with the actual existing compatible
profile name (the string is a placeholder, not a provided profile):

```bash
python tools/generate_system_profiles.py \
  --profile c2afcl_fixed_task_v1 \
  --reuse_plans_from YOUR_OLD_PROFILE

bash scripts/run_one.sh c2afcl 2 c2afcl_fixed_task_v1
bash scripts/run_one.sh c2afcl 5 c2afcl_fixed_task_v1
bash scripts/run_one.sh c2afcl 10 c2afcl_fixed_task_v1
```

GPU count is detected normally; no manual reference batch fields are needed in
base.yaml. For offline generation only, existing `--reference_world_size` can
freeze the reference machine's actual GPU count. If data or training work changed,
omit `--reuse_plans_from` to create new compatible Plans for ALL comparisons.

The three inherited entrypoints are:

- `configs/experiments/llava/afvlm_cm/2clients/c2afcl.yaml`
- `configs/experiments/llava/afvlm_cm/5clients/c2afcl.yaml`
- `configs/experiments/llava/afvlm_cm/10clients/c2afcl.yaml`

Existing `run_one.sh` accepts the name without changes. `run_baselines.sh` now
includes `c2afcl`; its preflight refuses a historical profile missing the new
config, before running any baseline. Use a new profile for the enlarged batch.

## Evaluation, diagnostics and costs

Use the unchanged single-server-global six-task evaluation. Periodic milestones
count incorporated client updates; with buffer>1 evaluate after an atomic flush
crosses a milestone, just as other buffered methods. There is no average of GPU
metrics and no additional personalized deployment model.

`events.jsonl` records `result_metadata[].c2afcl_diagnostics` keyed by update ID:
client/task, cache hit, download/receive versions, staleness, raw/calibrated/safe
and removed norms, protected task list, effective SVD rank, singular values and
orthogonality residual. Aggregation metadata additionally records task history
counts, buffer task/occupancy/wait time, rule and aggregation count. Pending rows
are reported at their eventual aggregation. Usual system statistics remain intact.

All calibration/SVD arithmetic stays FP32 on the server state's aggregation
device; local parameters and gradients are not modified. Only small diagnostics
are copied to CPU for JSON; final method checkpoint follows existing runtime
serialization. Server state_dict includes raw caches, means, counts, task buffers
and layout; this does not add resume support to the training workflow.

Memory: `O((N_clients + Q + 2*pending_updates)*D)` FP32 persistent vectors;
thin SVD workspace scales with `D*(Q-1)`, never `D^2`. For 7B rank8 q/k/v/o LoRA,
about8.4M coordinates imply about32MiB per vector, so60 client raw caches alone
are about1.9GiB. GPU SVD adds server compute/memory versus FedAsync. Projection
can remove beneficial shared-task directions, and cache subtraction can shrink
stable repeated updates to zero. These are properties to measure, not bypass
with undocumented heuristics. Actual CUDA memory/performance and published
accuracy are not established by non-training algebra/config checks.
