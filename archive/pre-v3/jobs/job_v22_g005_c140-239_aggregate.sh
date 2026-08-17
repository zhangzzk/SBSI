#!/bin/bash
#SBATCH --job-name=v22g5_result
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22g5_result_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_g005_c140-239/constant_response_catalogue_train.feather
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_g005_c140-239
RESULT=results/v22_constgold_g005_c140-239.txt
TMP="${RESULT}.tmp.${SLURM_JOB_ID}"
mapfile -t DUMPS < <(find "$RUN/dumps" -maxdepth 1 -type f -name 'ablate_s2c_lt500_v22_perobj_s*.feather' | sort)
[ "${#DUMPS[@]}" -eq 16 ] || { echo "expected 16 seed dumps, found ${#DUMPS[@]}"; exit 1; }
[ ! -e "$RESULT" ] || { echo "REFUSING to overwrite $RESULT"; exit 1; }

echo "### FROZEN V2.2 ON FRESH g=0.05 CONSTGOLD CASES 140--239 ###"
date
"$PY" -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$RUN/dumps/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 140 --re-min 0.5 --mag-max 25.8 \
  --emulator-tag lsst_r_extnbr_v22 --check-emulator-coverage | tee "$TMP"
mv "$TMP" "$RESULT"
echo "V22_G005_RESULT_DONE -> $RESULT"
date
