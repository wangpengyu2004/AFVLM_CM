#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_baselines.sh <setting: 2|5|10> [profile]" >&2
  exit 2
fi

setting="$1"
profile="${2:-}"
# FedASMU uses a disclosed client-training feedback specialization (extra work).
methods=(local fedavg fedprox fedadam fedasync fedbuff fedcompass fedasmu orthofl masfl adamasfl pilot unifed_lora)
for method in "${methods[@]}"; do
  if [[ -n "$profile" ]]; then
    config="experiment_profiles/${profile}/configs/${setting}clients/${method}.yaml"
  else
    config="configs/experiments/llava/afvlm_cm/${setting}clients/${method}.yaml"
  fi
  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config. Create a new profile; existing snapshots are immutable." >&2
    exit 2
  fi
done
for method in "${methods[@]}"; do
  if [[ -n "$profile" ]]; then
    bash scripts/run_one.sh "$method" "$setting" "$profile"
  else
    bash scripts/run_one.sh "$method" "$setting"
  fi
done
