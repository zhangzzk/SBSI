#!/bin/bash
#SBATCH --job-name=v2_selffig
#SBATCH --time=00:40:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2_selffig_%j.out

# Fig2/Fig5 (V2) data: per-object flow self-response R_flow (pinned lt500 8-seed ensemble, mean-head
# finite diff) vs the half-shear truth R_hs, with mag/size/nbr_flux/iso -> npz for plotting.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selfresp_gap_v2_lt500.npz
echo "### V2 SELFRESP FIG job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selfresp_gap.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s50*_swaavg.pt" \
  --max-case 39 --output $OUT
echo "V2_SELFRESP_FIG_DONE"; date
