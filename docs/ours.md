# Proposed method: Rank-Gate Sensitivity-Aware Asynchronous LoRA Aggregation

The registered method **ours** implements the supplied Rank-Gate design for
fixed-task asynchronous federated LLaVA instruction tuning. It changes neither
the LLaVA/PEFT model structure nor the LoRA rank, adds no trainable parameter,
and performs no SVD or cross-client rank alignment. Only LoRA A/B tensors and
small sensitivity metadata cross the federation boundary.

## 1. PEFT LoRA discovery and validation

The function **iter_lora_modules(model)** reads the active PEFT adapter and
yields:

~~~text
module_name, A[rank,in], B[out,rank], scaling
~~~

The client validates that the A and B ranks match. Every update also carries
module name, A/B state keys, shapes, rank, and scaling. The authoritative
method validates this schema against the server LoRA state and rejects later
clients whose module layout or scaling differs. Frozen 7B parameters and the
vision backbone are never uploaded.

## 2. Rank sensitivity at the client

For module l, rank r, and the accumulated gradient immediately before one real
optimizer step:

~~~text
h_A[l,r] = sum_i grad(A[l,r,i]) A[l,r,i]
h_B[l,r] = sum_j grad(B[l,j,r]) B[l,j,r]
h[l,r]   = 0.5 (h_A[l,r] + h_B[l,r])
c[l,r]   = h[l,r]^2
~~~

**RankSensitivityTracker** keeps a zero-initialized FP32 EMA:

~~~text
v[l,r] <- beta v[l,r] + (1-beta) c[l,r]
~~~

The hook runs exactly once after all gradient-accumulation micro-batches and
before optimizer.step(). The current LLaVA path has no GradScaler and no
gradient clipping, so gradients are already unscaled. If AMP scaling is added
later, unscale must happen before this hook.

For each module with n_l positive observations:

~~~text
v_hat[l,r] = v[l,r] / (1-beta^n_l)
~~~

All ranks from all modules are normalized by one client-wide mean and clipped:

~~~text
C[k,l,r] = clip(v_hat[l,r] / (mean(v_hat)+eps), 0, clip_max)
C_module[k,l] = mean_r C[k,l,r]
~~~

When the global mean is non-finite or at most eps, every rank falls back to
one and sensitivity_valid is false; the update remains auditable rather than
silently producing zero aggregation.

## 3. Rank-functional staleness

For one LoRA rank, the effective function is:

~~~text
M[l,r] = scaling[l] B[l,:,r] A[l,r,:]
~~~

The implementation never constructs the dense outer product. It uses FP64:

~~~text
||s b1 a1^T - s b0 a0^T||_F^2
= s^2 (
    ||b1||^2 ||a1||^2
  + ||b0||^2 ||a0||^2
  - 2 (b1^T b0)(a1^T a0)
)
~~~

Let s be the immutable client base version, t the current server, and k the
returned client endpoint. After dividing each rank distance by
out_features times in_features:

~~~text
D[k,t] = sum_l,r C[k,l,r] d(M[t,l,r], M[s,l,r])
U[k]   = sum_l,r C[k,l,r] d(M[k,l,r], M[s,l,r])

delta[k,t] = sqrt(D[k,t] / (U[k] + eps))
rho[k,t]   = exp(-gamma min(delta[k,t], delta_max))
~~~

The existing Update.base_state is already an immutable LoRA snapshot of the
client's start version, so no frozen backbone snapshots or duplicate full
version archive is required.

An update is rejected as no_effective_update only when both functional energy
U and the ordinary A/B parameter delta norm are below min_update_energy.
Rejected updates do not change the model, server version, or task memory.

## 4. Task-balanced module memory

For each task q and module l, the server stores one scalar Omega[q,l]. Task
weights are automatically uniform over all task IDs in the current system
profile. Every task/module entry is explicitly initialized to zero:

~~~text
Omega[q,l] = 0
~~~

The protected historical precision is:

~~~text
history_precision[l]
= base_precision + sum_q pi[q] Omega[q,l]
~~~

Only the source task is updated, after the new model endpoint has been
computed:

~~~text
Omega[q_k,l] <- history_beta Omega[q_k,l]
                + (1-history_beta) rho[k,t] C_module[k,l]
~~~

Other tasks are not decayed when task q_k arrives.

## 5. Module-level A/B aggregation

The first implementation uses constant client_weight=1. For each module:

~~~text
p_client[l] = rho[k,t] C_module[k,l]

alpha[k,l] =
  p_client[l]
  / (p_client[l] + history_lambda history_precision[l] + eps)

alpha[k,l] <- clip(alpha[k,l], 0, alpha_max)
~~~

One alpha is shared by the complete A and B tensors of that module:

~~~text
A_new[l] = A_current[l] + alpha[k,l] (A_client[l] - A_current[l])
B_new[l] = B_current[l] + alpha[k,l] (B_client[l] - B_current[l])
~~~

The endpoint is the returned client solution, not current plus alpha times
(client minus base). Rank sensitivity controls functional staleness; it does
not create independent rank-wise fusion coefficients.

## 6. Configuration

The complete first-version configuration is
**configs/methods/ours.yaml**:

~~~yaml
method:
  name: ours
  params:
    sensitivity_beta: 0.95
    sensitivity_warmup_steps: 1
    sensitivity_clip_max: 10.0
    gamma: 1.0
    delta_max: 10.0
    history_beta: 0.95
    history_lambda: 1.0
    base_precision: 1.0
    alpha_max: 0.5
    eps: 0.000000000001
    min_update_energy: 0.0000000000000001
~~~

There are no Module-Gate, Adam-v, sample-weighting, SVD, or rank-alignment
switches in this version.

## 7. DDP execution and virtual semantics

The editable base configuration uses executor policy **capability**. The
proposed method supports client DDP, so **scripts/run_one.sh** launches one
process per visible GPU and all ranks train the same client:

~~~text
per-device batch = training.batch_size
effective batch  = batch_size * gradient_accumulation * DDP world size
~~~

Samples are deterministically shuffled, padded like DistributedSampler, and
sharded across ranks. Non-final accumulation micro-batches use DDP.no_sync();
the final backward synchronizes LoRA gradients before Rank-Gate sensitivity is
sampled.

DDP changes physical optimizer steps and effective batch, but does not change
the persisted virtual Plan's logical start, frozen base version, arrival, or
aggregation order. Output train_plan.json, events.jsonl, and system_stats.json
record physical executor, DDP world size, effective batch, and actual
optimizer steps.

FedCompass, FedASMU, MasFL, AdaMasFL, and Pilot declare that they do not support
client DDP. The first four preserve their algorithm-specific local-step
allocation, mid-training refresh, or optimizer-step gradient trajectory. Pilot
preserves its dynamic task/client adapter unused-parameter topology alongside
re-entrant gradient checkpointing. They continue to use the original
one-GPU-per-client worker pool. This requested mixed executor policy must be
disclosed because local optimization trajectories differ across executors.

## 8. Running

Create a new immutable profile after this method/configuration change while
reusing the previous system profile and virtual TrainPlan:

~~~bash
python tools/generate_system_profiles.py \
  --profile rank_gate_ddp_e1_bs1_ga4_r10_s42 \
  --reuse_plans_from default_e1_bs1_ga4_r10_s42
~~~

Run the proposed method on all visible GPUs:

~~~bash
bash scripts/run_one.sh ours 2 rank_gate_ddp_e1_bs1_ga4_r10_s42
~~~

Restricting visible devices also changes DDP world size and effective batch, so
do it only when the experiment records that choice:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_one.sh ours 2 rank_gate_ddp_e1_bs1_ga4_r10_s42
~~~

Settings 5 and 10 select the other client-count datasets.

## 9. Output and limitations

events.jsonl result metadata records server drift, local functional update,
parameter delta norm, relative staleness, reliability, validity of the
sensitivity estimate, per-module alphas, and rejection reason. The final
checkpoint stores method parameters, task memory, uniform task weights, and
the validated LoRA module/scaling schema.

Known boundaries:

- Only LoRA A/B may be federated; enabling trainable mm_projector is rejected.
- Rank sensitivity is a local gradient/parameter gate statistic, not a full
  Fisher matrix or Hessian.
- Interpolating A and B separately does not equal a linear interpolation of
  dense BA; the implementation intentionally preserves fixed PEFT rank.
- DDP uses a larger global effective batch than one-GPU client training when
  per-device batch is unchanged.
- No 7B training is run by repository validation.
