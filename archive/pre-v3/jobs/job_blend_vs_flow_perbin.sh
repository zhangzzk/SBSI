#!/bin/bash
#SBATCH --job-name=bl_vs_fl
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bl_vs_fl_%j.out
# Per-bin attribution: is the response error the FLOW's self response or the EMULATOR's R_blend?
# CPU only -- reads existing dumps, scores nothing. 13G of per-object dumps x 16 seeds.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### BLEND-VS-FLOW PER-BIN job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_blend_vs_flow_perbin.py "$@" || { echo "BL_VS_FL_FAILED"; exit 1; }
echo "BL_VS_FL_DONE"; date
