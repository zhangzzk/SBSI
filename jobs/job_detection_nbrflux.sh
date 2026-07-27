#!/bin/bash
#SBATCH --job-name=cg_detnf
#SBATCH --time=00:40:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_detnf_%j.out

# Detection bias vs the FLOW's neighbour-flux axis nbr_flux_near (size-decoupled severity).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG DETNF job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_detection_nbrflux.py --col nbr_flux_near --min-case 40 --nbins 6 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detection_nbrflux_v1.npz
echo "CG_DETNF_JOB_DONE"; date
