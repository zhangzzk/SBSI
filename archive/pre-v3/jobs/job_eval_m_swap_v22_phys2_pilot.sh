#!/bin/bash
#SBATCH --job-name=mswap_v22_phys2_pilot
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/mswap_v22_phys2_pilot_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SHARDS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_shards
PATTERN="$SHARDS/ablate_s2c_lt500_v22_perobj_s*_c040-050.feather"
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
LOOKUP="$ROOT/results/blend_lookup_v22_phys2_c40-49.feather"
CHECKPOINT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
OUT="$ROOT/results/constgold_m_swap_v22_phys2_16seed_c40-49.json"

export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"

cd "$ROOT"
[ -f "$CAT" ] || { echo "MISSING catalogue: $CAT"; exit 1; }
[ -f "$LOOKUP" ] || { echo "MISSING lookup: $LOOKUP"; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "MISSING checkpoint: $CHECKPOINT"; exit 1; }
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
mapfile -t dumps < <(compgen -G "$PATTERN" | sort)
[ "${#dumps[@]}" -eq 16 ] || { echo "EXPECTED 16 dumps, found ${#dumps[@]}"; exit 1; }

echo "### V2.2 PHYS2 CONSTGOLD M PILOT job=$SLURM_JOB_ID cases=40-49 seeds=16 ###"
date
"$PY" -u scripts/eval_m_swap_emulator_v22.py \
  --dump-glob "$PATTERN" \
  --catalogue "$CAT" \
  --lookup "$LOOKUP" \
  --candidate-tag lsst_r_extnbr_v22_phys2 \
  --flow-checkpoint "$CHECKPOINT" \
  --min-case 40 \
  --max-case 50 \
  --output "$OUT"
[ -f "$OUT" ] || { echo "M_SWAP_FAILED: no output"; exit 1; }
echo "M_SWAP_DONE: $OUT"
date
