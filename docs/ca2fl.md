# CA2FL: fixed-task asynchronous LoRA baseline

## Source and suitability

**Tackling the Data Heterogeneity in Asynchronous Federated Learning with Cached
Update Calibration**, Yujia Wang, Yuanpu Cao, Jingcheng Wu, Ruoyu Chen and
Jinghui Chen, ICLR 2024.
[Final paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/464c70add1e8119c4301de4e1dde17b4-Paper-Conference.pdf).
Implementation anchors: Algorithm 2, p.5; Eq.(4.1), p.6; language/LoRA settings,
p.7 and Appendix A.3, p.17; memory-friendly extension, Appendix A.1, p.14.

This is directly relevant to asynchronous **data heterogeneity**, not federated
continual or class-incremental learning. Fixed task identities require no task
stream, client reassignment or new server dataset. The paper already includes
LoRA+AdamW language-model experiments. Extending to six VLM instruction tasks
is nevertheless an adaptation, not a reproduction of its reported scores.

No author implementation link was found in the inspected final paper or author
publication pages. This is an equation-based implementation, not a claim that
third-party FL libraries are official author code.

## Algorithm and cache order

Let delta_i be the uploaded **local minus immutable downloaded base** in the
common trainable LoRA A/B coordinates. No frozen backbone/vision weights enter.

All N configured clients start with zero h_i, including clients not yet seen.
The first global mean h is zero. On each virtual arrival:

```text
correction_sum += delta_i - h_i    # read PREVIOUS client cache
h_i = delta_i                    # replace client cache, no EMA or decay
```

On M arrivals, form the direction with the **PRE-buffer global mean**:

```text
v = h + correction_sum / number_of_distinct_clients_in_buffer
theta_new = theta_current + server_lr * v
h = sum_i h_i / N                 # refresh only AFTER forming v
correction_sum = 0; buffer = empty
```

Never replace h with the newly updated mean before computing v; never divide
by only the clients whose caches are nonzero. There are no sample-size,
task-aware, extra optimizer-step or staleness-decay weights. Different client
sizes remain unchanged and uniform-client calibration follows the source.

The common Plan can contain repeated arrivals from one client before a buffer
fills. Algorithm 2 uses a set S but does not separately specify duplicate-client
handling within a buffer. Our disclosed extension replaces the latest cache
per arrival, threshold M counts uploads, and |S| counts distinct clients. In
full precision, same-client corrections telescope to its latest minus pre-buffer
cache, giving Eq.(4.1) with the latest delta per distinct client. This case is
explicitly checked rather than silently discarding an upload or claiming the
paper analyzed this execution. Quantized-cache corrections follow the same
arrival order with quantization noise, without invented error feedback.

Completed tasks/clients remain in the fixed N cache population. There is no
task retirement reweighting or ours-specific history protection. Cached
directions may therefore persist after a client's last upload; that is an
algorithm property to report, not grounds for deleting its cache.

## Configuration and paper differences

Edit `configs/methods/ca2fl.yaml` before generating a new profile:

```yaml
method:
  name: ca2fl
  params:
    server_lr: 1.0
    buffer_size: 3
    cache_bits: 32
    cache_seed: 42
    flush_last_buffer: true
```

- M=3 follows the paper's BERT-LoRA language setting (vision uses M=10;
  GPT2 generation uses M=2). It is configurable, not a universal optimum.
- server_lr=1 is in the paper's learning-rate search grid, not a claim of an
  official VLM optimum. Other baselines' configurations are not changed.
- Local epochs, batch, accumulation, optimizer, learning rate, sequence limit,
  LoRA rank/targets and evaluation all inherit the SAME project configuration.
  We do not replace them with BERT rank1/alpha1 or paper-specific AdamW settings.
- Reuse the project's system profile, virtual arrivals, immutable download
  snapshots and physical capability-driven DDP. Do not recreate paper-specific
  concurrency/client-resampling or latency distributions and invalidate fairness.
- At a finite Plan's end, a partial buffer is flushed with its actual distinct
  contributors. Setting flush_last_buffer=false raises if anything remains;
  residual work is never silently lost. This is a finite-experiment adaptation.
- No local loss/gradient/optimizer/adapter hooks are introduced. Both serial
  and client-parallel overrides remain supported by the same method interface.

## MF-CA2FL: optional packed cache

Set cache_bits=8 or 4 to enable the Appendix A.1 memory-friendly extension.
Only stored per-client h_i is quantized; newly arriving delta stays FP32.
The global mean is computed from quantized caches, as specified in Appendix A.1.

The source requires unbiased quantization but does not specify its exact codec.
This implementation chooses stochastic uniform affine rounding over the flattened
trainable vector's min/max. Each coordinate rounds to the neighbouring levels
with probabilities preserving its expectation (up to floating-point roundoff).
Eight-bit codes use uint8; two four-bit codes are packed into each uint8.
Offset/step metadata stays on the aggregation device. Zero/constant vectors are
represented exactly. Randomness uses a private device generator seeded by
cache_seed and update_id; it never consumes the local-training RNG sequence.
This is NOT NF4, deterministic rounding or an allegedly author-provided codec.

Full cache storage is approximately N*P*4 bytes, where P is the communicated
LoRA parameter count. Packed caches approach N*P bytes (8bit) or N*P/2 (4bit),
plus metadata. Three FP32 vectors, transient tensors and ordinary server/model
snapshots remain; total GPU memory is NOT reduced by exactly 4x/8x. Eight DDP
replicas do not automatically shard server cache memory. Real GPU peak memory
and speed require an actual experiment and have not been measured here.

## Launch

Existing immutable profiles do not acquire a ca2fl.yaml automatically. On the
training server, with unchanged data/local-training/system settings, create a
new ordinary profile while reusing an old ordinary Plan:

```bash
python tools/generate_system_profiles.py \
  --profile ca2fl_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from e1_bs4_ga4_r10_s42

bash scripts/run_one.sh ca2fl 2 ca2fl_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ca2fl 5 ca2fl_e1_bs4_ga4_r10_s42
bash scripts/run_one.sh ca2fl 10 ca2fl_e1_bs4_ga4_r10_s42
```

Replace the old profile name with one that actually exists. The name is a label,
not a parameter override. The generator reads current base/model/method templates;
verify frozen configs' training/evaluation/runtime against your reference. Reusing
a Plan does not freeze other methods' configurations to an earlier code version.
Changing cache_bits/server_lr/M needs no new system conditions; preserve old
results/configs and use a separate profile/output for every ablation.

Without a profile, `bash scripts/run_one.sh ca2fl 2` uses inherited templates and
the default dataset Plan paths, not a custom profile's Plan. The existing default
baseline batch sequence is deliberately unchanged; launch this new method explicitly.

## Evaluation, diagnostics and verification boundary

The common server-global pipeline evaluates the same six complete task splits.
Periodic evaluation follows the configured accepted-client-update budget, not
an invented count of cached clients. A full M=3 buffer advances accepted work
by three while server_version advances once. If cadence is 10, the next boundary
can be reached at accepted update12, then21, rather than cutting a buffer apart.
Cached reuse is not counted as an additional client upload or training work.

Existing events.jsonl records buffer occupancy/distinct clients, flush count,
waiting time, cache bits/population/storage bytes, before/after global-cache
norms and calibrated-direction norm. ca2fl_diagnostics records each upload's
base/receive version, staleness, raw/cached/correction norms and quantization error.
Common summary reports accepted updates, staleness, virtual time and buffer stats.
Method state includes pending corrections, caches, identities and counters in the
existing final checkpoint; this does not add interrupted-training resume support.

Non-training checks cover equation order, unseen/repeated clients, actual packed
storage, stochastic expectation, private RNG, residual work, state serialization,
three config inheritances, DDP selection and shared regression coverage. These
checks do not establish paper-level accuracy, convergence guarantees under
AdamW/fixed-task VLM assumptions, or successful real eight-GPU training.
