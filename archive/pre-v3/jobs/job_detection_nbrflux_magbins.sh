#!/bin/bash
#SBATCH --job-name=cg_nfmag
#SBATCH --time=00:40:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_nfmag_%j.out

# Detection bias vs nbr_flux_near in 4 true-mag bins (24.5-26.5), for the mag-curve plot.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG NFMAG job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_detection_nbrflux.py --col nbr_flux_near --min-case 40 --nbins 8 \
  --mag-edges 24.5 25.0 25.5 26.0 26.5 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detection_nbrflux_magbins_v1.npz
echo "CG_NFMAG_JOB_DONE"; date
