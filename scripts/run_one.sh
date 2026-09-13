#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_one.sh <method> <setting: 2|5|10>" >&2
  exit 2
fi

method="$1"
setting="$2"
case "$setting" in 2|5|10) ;; *) echo "setting must be 2, 5, or 10" >&2; exit 2 ;; esac

config="configs/experiments/llava/afvlm_cm/${setting}clients/${method}.yaml"
if [[ ! -f "$config" ]]; then
  echo "Unknown method or missing experiment config: $config" >&2
  exit 2
fi
python scripts/run_experiment.py --config "$config"
