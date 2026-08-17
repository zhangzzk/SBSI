#!/bin/bash
#SBATCH --job-name=cg_selbias
#SBATCH --time=00:30:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_selbias_%j.out

# Truth-only constgold SELECTION-bias confirmation (intrinsic + measured shapes, per-leg S/N cut).
# CPU-only; no GPU. See scripts/eval_selection_constgold.py header.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG SELBIAS job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_selection_constgold.py \
  --min-case 40 --sn-cuts 5 7 10 15 20 --re-null 0.3 0.4 0.5
echo "CG_SELBIAS_JOB_DONE"; date
