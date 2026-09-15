#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_one.sh <method> <setting: 2|5|10> [profile]" >&2
  exit 2
fi

method="$1"
setting="$2"
profile="${3:-}"
case "$setting" in 2|5|10) ;; *) echo "setting must be 2, 5, or 10" >&2; exit 2 ;; esac

if [[ -n "$profile" ]]; then
  if [[ ! "$profile" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ || "$profile" == "current" ]]; then
    echo "invalid experiment profile name: $profile" >&2
    exit 2
  fi
  config="experiment_profiles/${profile}/configs/${setting}clients/${method}.yaml"
else
  config="configs/experiments/llava/afvlm_cm/${setting}clients/${method}.yaml"
fi
if [[ ! -f "$config" ]]; then
  echo "Unknown method or missing experiment config: $config" >&2
  exit 2
fi
python scripts/run_experiment.py --config "$config"
