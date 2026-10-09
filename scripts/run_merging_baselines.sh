#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_merging_baselines.sh <setting: 2|5|10> [profile]" >&2
  exit 2
fi
case "$1" in 2|5|10) ;; *) echo "setting must be 2, 5, or 10" >&2; exit 2 ;; esac
for method in opcm_lora dop_lora nufilt_lora; do
  if [[ $# -eq 2 ]]; then
    bash scripts/run_one.sh "$method" "$1" "$2"
  else
    bash scripts/run_one.sh "$method" "$1"
  fi
done
