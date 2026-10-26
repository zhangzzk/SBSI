#!/usr/bin/env bash
# CPU-only operational smoke on one small scene; no calibration acceptance.
#SBATCH --job-name=sbsi_multiaxis_smoke
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-3%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_multiaxis_evaluator_20260918_v1/smoke_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_multiaxis_evaluator_20260918_v1/smoke_%A_%a.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
root=$cache/joint_multiaxis_evaluator_20260918_v1
python=$data/envs/py31/bin/python
axis=$((1 + SLURM_ARRAY_TASK_ID / 2))
steps=(0.02 0.04)
step=${steps[$((SLURM_ARRAY_TASK_ID % 2))]}
flow=$cache/harmonized_secondary_20260917_v1/refinement/selected.pt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" "$flow" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, flow = map(Path, sys.argv[1:])
manifest = json.loads((root/'manifest.json').read_text())
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if (manifest.get('status') != 'validated'
        or digest(root/'evaluate_constgold_truth_parent.py') != manifest['staged_sha256']
        or digest('scripts/joint_response_axes.py') != manifest['axis_helper_sha256']):
    raise RuntimeError('staged evaluator validation mismatch')
if digest(flow) != 'fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4':
    raise RuntimeError('pilot flow differs from frozen corrected checkpoint')
PY
options=()
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
exec "$python" -u -c 'import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name="__main__")' \
  "$root/evaluate_constgold_truth_parent.py" --parent-root "$root/parent" \
  --constgold-root "$data/blendemu_runs/joint_multiaxis_pilot_20260918_v1/images/g${axis}_h${step}" \
  --flow "$flow" --response-candidate "$data/blendemu_runs/disk_response_moment_20260918_v1/model" \
  --response-composition mobius --response-device cpu --response-threads 4 --device cpu \
  --pair-root "$root/pairs" --rblend "$root/pair_lookup.feather" --cohorts truth_parent \
  --h "$step" --axis "$axis" --seed 7301 --draws 64 --batch-size 1024 \
  "${options[@]}" --cases 20000 --operational-single-case --output "$root/smoke_g${axis}_h${step}.json"
