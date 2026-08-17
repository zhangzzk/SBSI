#!/bin/bash
#SBATCH --job-name=s2_selresp
#SBATCH --time=03:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2_selresp_%j.out

# Stage-2 selection-response gate: push moving MEASURED size/mag cuts through the pinned 4D flow,
# compare the selected-catalogue shear response to the det_meas half-shear truth (ISOLATED set,
# R_blend~0). m_C = R_sim,C/R_model,C - 1 must stay within target across the cut suite; the TRUE-Re
# cuts must show the moving-boundary term ~ 0 (null test).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
echo "### S2 SELECTION RESPONSE job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_response.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s50*_swaavg.pt" \
  --max-case 39 --n-samples 128 --batch-size 16384 --all-too \
  --output "$D/selection_response_s2c_lt500.npz"
echo "S2_SELRESP_DONE"; date
