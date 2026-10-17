# Baseline Implementation Notes

The current authoritative algorithm/source/parameter audit is
[baseline_audit.md](baseline_audit.md). It supersedes the former scalar-gate
UniFed, joint-loss residual Pilot, heuristic FedASMU and static Compass descriptions.
The criterion is complete core functionality with disclosed LoRA/fixed-task
adaptations, not copying source bugs or claiming reproduced paper scores.

## Shared protocol and algorithm exceptions

All ordinary methods use the same LLaVA checkpoint, language LoRA, dataset,
random seed, held-out corpora and reference effective-batch budget. Frozen 7B
parameters never federate. Fresh local optimizers are created per job.
MasFL/AdaMasFL require SGD instead of AdamW and FP32 trainable adapters; their
raw-gradient initialization adds recorded work. Pilot adds trainable visual
connectors and private routing; UniFed adds a server factor hypernetwork.
These are real algorithm costs, not free or identical parameter budgets.

New profiles automatically freeze visible GPU count. Reference eight gives
DDP B4/G4 and worker B4/G32, both128, with identical reference shard/tail
objectives. This is not bitwise/dropout equivalence. Immutable legacy profiles
remain unchanged. See [batch_fairness.md](batch_fairness.md).

Ordinary local work is controlled by local_epochs. FedCompass assigns bounded
physical optimizer steps dynamically. A common system profile fixes speed,
network delay and initial availability independently of aggregation. Other
methods replay the same ordinary arrival Plan; Compass compiles its algorithm's
semi-asynchronous schedule from that same profile.

## Availability and invocation

Main runnable comparisons: local, fedavg, fedprox, fedadam, fedasync, fedbuff, ca2fl,
fedcompass, fedasmu, orthofl, c2afcl, masfl, adamasfl, pilot, unifed_lora. Optional continual **merging**:
opcm_lora, dop_lora, nufilt_lora. Ours remains the proposed method, not a baseline.

Standalone comparison: `opcm_oneshot` trains every client once for multiple epochs
from the same version0, collects every upload before offline OPCM, and evaluates
only the final global model. Use `tools/generate_oneshot_profiles.py` and
`scripts/run_oneshot.sh`; ordinary profile generation/default batch remain
unchanged. This intentionally changes communication/optimizer continuity,
not the original asynchronous OPCM. See [protocol and usage](opcm_oneshot.md).
FedASMU now uses an explicitly disclosed private-client training-objective oracle,
with dynamic server/local coefficients and learned LSTM/Q request slots. No server
dataset is reserved. Extra objective work is reported separately; this is not the
paper's exact global objective. See [implementation and operation](orthofl_fedasmu.md).

C2-AFCL preserves same-client/task raw-update cache differences and joint-coordinate
SVD interference projection, specialized to fixed tasks and federated LoRA states.
Default per-task buffer1 gives immediate incremental aggregation. Other-task
incorporated history replaces previous sequential tasks; numeric settings and
protocol ambiguity are disclosed in [C2-AFCL adaptation and commands](c2afcl.md).
No local training, dataset, Plan or evaluation code is changed for this baseline.

CA2FL (ICLR2024 Algorithm2/Eq4.1) adds an independent client-update cache and
buffered calibration using the previous global cache mean. It also supports
AppendixA.1 4/8bit unbiased packed caches. Defaults follow the language-buffer
setting M3 with server_lr1; local training remains the common VLM configuration.
No author-code equivalence claim is made. Standard profile generation includes
CA2FL; old snapshots and the existing default batch sequence remain unchanged.
See [algorithm, adaptations, memory and launch commands](ca2fl.md).

```bash
bash scripts/run_one.sh METHOD 2 PROFILE
bash scripts/run_baselines.sh 2 PROFILE
bash scripts/run_merging_baselines.sh 2 PROFILE
```

Replace METHOD/PROFILE with actual registry/profile names; setting2 may be5 or10.
Create a new profile to capture corrected templates. Old Pilot bottleneck and
UniFed conditioning_scale profiles are rejected, not silently executed as an
incorrect same-name algorithm. Reuse ordinary Plans only when dataset and all
local-work/system settings match; see the audit's exact generation command.

## Evaluation

All aggregated methods keep global main evaluation on the common six-task
pipeline at incorporated-client-update milestones and finalization. Local-only
has no global model and evaluates per-client models on common task test sets.
No cross-task raw average is reported. Prediction shards are merged before
computing complete-corpus metrics; GPU metric averages are not used.

Pilot's global deployment view uses shared task connectors plus global text
LoRA, not a private-client mean connector. UniFed's global view is mean-generated
LoRA factors. Both views are disclosed adaptations of personalized methods.
Their native client models are additionally evaluated at the final split into
personalized_metrics.json, with within-task client macro-averages. Other methods
do not receive extra evaluations. This secondary file is separate from metrics.json
and does not alter periodic evaluation frequency or accepted-update counts.

## Verification limits

Equation/state/config checks do not load the VLM, launch training, download
weights, establish CUDA/NCCL compatibility, guarantee memory fit or reproduce
original benchmark accuracy. Full client-specific hypernetwork heads and Pilot
private state can be expensive. Original paper convergence guarantees do not
automatically transfer to nonuniform Plan arrivals, variable K, LoRA or compact
server Adam. Sources, numerical boundary fixes and costs are explicitly listed
in [the audit](baseline_audit.md) and [continual merging](continual_merging.md).
