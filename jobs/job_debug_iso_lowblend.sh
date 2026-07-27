#!/bin/bash
#SBATCH --job-name=cg_dbgiso
#SBATCH --time=00:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_dbgiso_%j.out

# DEBUG: why isolated det-response > lightly-blended? flag leg-consistency + mag/size composition.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG DBGISO job=$SLURM_JOB_ID ###"; date
python -u scripts/debug_isolated_lowblend.py --min-case 40 --beta-thr 1e-3
echo "CG_DBGISO_JOB_DONE"; date
