# Versioned experiment profiles

Each subdirectory is an immutable AFVLM-CM experiment snapshot. A profile
contains:

- fully resolved configs for every method and client setting;
- method-independent system profiles and asynchronous TrainPlans;
- a manifest recording shared parameters, source commit, and SHA-256 hashes;
- output paths namespaced by the profile name.

Create a new profile from the parameters currently stored under `configs/`:

```bash
python tools/generate_system_profiles.py --profile e2_bs1_ga4_r10_s42
```

Reuse an existing compatible plan when only non-scheduling parameters change:

```bash
python tools/generate_system_profiles.py \
  --profile lr1e5_alpha03 \
  --reuse_plans_from default_e1_bs1_ga4_r10_s42
```

Profile names are never overwritten. Run a saved profile with:

```bash
bash scripts/run_one.sh fedasync 2 default_e1_bs1_ga4_r10_s42
```
