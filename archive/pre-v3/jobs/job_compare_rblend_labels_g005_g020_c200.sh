#!/bin/bash
#SBATCH --job-name=rblab_amp200
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblab_amp200_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/rblab_amp200_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
OUT=results/rblend_labels_g005_vs_g020_v22_c0-199.json
CASECSV=results/rblend_labels_g005_vs_g020_v22_c0-199_cases.csv
BINCSV=results/rblend_labels_g005_vs_g020_v22_c0-199_bins.csv
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$ROOT"

for path in "$OUT" "$CASECSV" "$BINCSV"; do
  [ ! -e "$path" ] || { echo "REFUSING existing $path"; exit 1; }
done
"$PY" -u scripts/compare_rblend_labels_g005_g020.py \
  --g005 "$BASE/response_catalogue_g005_train.feather" \
         "$BASE/response_catalogue_g005_c100-199.feather" \
  --g020 "$BASE/response_catalogue_train.feather" --max-case 199 \
  --output "$OUT" --case-csv "$CASECSV" --bin-csv "$BINCSV"
echo RBLEND_LABEL_AMPLITUDE_C200_JOB_DONE; date
