#!/usr/bin/env bash
# Read the FLOW directly, on the primaries where blending cannot hide in it.
#
# The earlier split binned on R_blend, the SIGNED sum over a primary's pairs.
# That sum reaches zero by cancellation, not by an absence of blending, so its
# near-zero bins never licensed a model-free read of R_flow.  R_blend_abs is the
# sum of |per-pair response|: it bounds how far any per-pair emulator error can
# move that primary.  Where it is small, R_measured IS R_self to within a known
# tolerance, and the flow is read against it with no subtraction of a model.

#SBATCH --job-name=sbsi_cgfg0_quiet
#  gpu:a40 matches only the WHOLE cards, which queue for hours.  The idle
#  capacity on cip is vGPU slices under different GRES names, and their nodes
#  carry ~25G of RAM, so a 96G request cannot land on one.  The identical split
#  job peaked at 1.2G and finished in five minutes.
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_flow_quiet_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_flow_quiet_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v2=$cache/fixed_g0_m258_r060_v2
blocks=$v2/constgold_blend_lookup_abs_v1/rblend_v2_baseline/blocks
root=$v2/constgold_flow_quiet_v1

mkdir -p "$root/logs"
#  the A40-16Q slice lacks the CUDA VMM APIs torch's expandable segments need
export PYTORCH_NO_CUDA_MEMORY_CACHING=${PYTORCH_NO_CUDA_MEMORY_CACHING:-0}
export NO_EXPANDABLE_SEGMENTS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
nvidia-smi -L

lookup=$root/rblend_v2_baseline_abs_cases40_89.feather
if [ ! -f "$lookup" ]; then
  "$python" - "$blocks" "$lookup" <<'PY'
import sys, pathlib
import pandas as pd
blocks = sorted(pathlib.Path(sys.argv[1]).glob("cases*.feather"))
if len(blocks) != 5:
    raise SystemExit(f"expected 5 blocks, found {len(blocks)}: {blocks}")
frame = pd.concat([pd.read_feather(b) for b in blocks], ignore_index=True)
if frame.duplicated(["case", "input_index"]).any():
    raise SystemExit("concatenated lookup has duplicate keys")
missing = set(range(40, 90)) - set(frame["case"].unique())
if missing:
    raise SystemExit(f"concatenated lookup misses cases {sorted(missing)}")
#  the absolute sum can never be smaller than the signed one
if (frame["R_blend_abs"] + 1e-12 < frame["R_blend"].abs()).any():
    raise SystemExit("absolute blending sum below the signed sum")
frame.to_feather(sys.argv[2])
print(f"LOOKUP_CONCAT rows={len(frame):,} "
      f"mean_R_blend={frame['R_blend'].mean():.6f} "
      f"mean_R_blend_abs={frame['R_blend_abs'].mean():.6f}")
PY
fi

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done

"$python" -u -m scripts.diagnose_constgold_response_split \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --flow "$v2/flow/paired/selected.pt" \
  --rblend "reject_baseline=$lookup" \
  "${cases[@]}" --bins 12 --bin-column R_blend_abs \
  --output "$root/result.json" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260915 --device cuda

echo "CONSTGOLD_FLOW_QUIET_DONE output=$root/result.json"
