#!/bin/bash
#SBATCH --job-name=ab_cpaircap
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-19%20
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_cpaircap_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_cpaircap_%A_%a.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_local10_c200-299
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_coherent_paircap64_c200-299
REFERENCE=$ROOT/results/anchorblend_g005_response_v22_c100-299.feather
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$PARTS"
cd "$ROOT"
START=$((200 + 5 * SLURM_ARRAY_TASK_ID))
END=$((START + 4))
for case in $(seq "$START" "$END"); do
  env SLURM_PROCID=0 python -u scripts/rescore_anchor_coherent_pair_cap.py \
    --base "$BASE" --reference "$REFERENCE" --output-dir "$PARTS" \
    --case-offset "$case" --n-cases 1 --wide-k 64
done
echo "ANCHORBLEND_COHERENT_PAIRCAP_PART_DONE cases=$START-$END"
date
