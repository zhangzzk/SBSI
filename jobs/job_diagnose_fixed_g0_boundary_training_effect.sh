#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fg0_boundary_train
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/root_cause_radius_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/root_cause_radius_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
v2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
out=$v2/root_cause_radius_v1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L

cases=$(
  "$python" - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path(
        "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
        "fixed_g0_m258_r060_v2/domain/manifest.json"
    ).read_text()
)
selected = sorted(map(int, manifest["split"]["train_cases"]))[:20]
print(" ".join(f"--case {case}" for case in selected))
PY
)

for model in nll staged direct; do
  case "$model" in
    nll) checkpoint=$v2/flow/nll/selected.pt ;;
    staged) checkpoint=$v2/flow/paired/selected.pt ;;
    direct) checkpoint=$v2/flow_direct_response/selected.pt ;;
    *) exit 2 ;;
  esac
  # shellcheck disable=SC2086
  "$python" -u -m scripts.build_flow_self_response_predictions \
    --domain-root "$v2/domain" \
    --flow "$checkpoint" \
    --output "$out/${model}_training20.feather" \
    --draws 64 \
    --sampling-seed 7301 \
    --batch-size 1024 \
    --device cuda \
    $cases
  "$python" -u -m scripts.diagnose_flow_self_response_by_radius \
    --domain-root "$v2/domain" \
    --predictions "$out/${model}_training20.feather" \
    --radius-edge 0.6154 \
    --radius-edge 0.6298 \
    --radius-edge 0.6436 \
    --radius-edge 0.6572 \
    --output "$out/${model}_training20_radius.json" \
    --replicates 10000 \
    --seed 20260916
done
