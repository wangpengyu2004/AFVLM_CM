#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_oneshot.sh <setting: 2|5|10> <one-shot-profile>" >&2
  exit 2
fi
setting="$1"
profile="$2"
case "$setting" in 2|5|10) ;; *) echo "setting must be 2, 5, or 10" >&2; exit 2 ;; esac
if [[ ! "$profile" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ || "$profile" == "current" ]]; then
  echo "invalid one-shot profile name" >&2
  exit 2
fi
config="one_shot_profiles/${profile}/configs/${setting}clients/opcm_oneshot.yaml"
if [[ ! -f "$config" ]]; then
  echo "Missing one-shot config: $config. Use tools/generate_oneshot_profiles.py first." >&2
  exit 2
fi
python tools/generate_oneshot_profiles.py --validate "$profile"
backend="$(python tools/select_runtime_backend.py --config "$config")"
echo "[AFVLM-CM] one-shot config: $config"
echo "[AFVLM-CM] selected runtime backend: $backend"
if [[ "$backend" == "client_ddp" ]]; then
  python -m torch.distributed.run --standalone --nproc_per_node=gpu \
    scripts/run_experiment.py --config "$config" --runtime-backend client_ddp
else
  python scripts/run_experiment.py --config "$config" --runtime-backend "$backend"
fi
