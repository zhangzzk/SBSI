#!/bin/bash
#SBATCH --job-name=s2_fxsz
#SBATCH --time=02:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2_fluxsize_%j.out

# Flux/size shear-response of the 4D true-cond flow (S2), 3-seed ensemble, full ISO ruler.
# Harvests mean-head dims 2,3 vs matched-pair truth; ghat-leg SHAPE control must be ~0%.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
GLOB='/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2_true4d_s50*_swaavg.pt'
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/fluxsize_resp_s2_3seed.npz
echo "### S2 FLUXSIZE EVAL job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_fluxsize_response.py --ckpt-glob "$GLOB" --max-case 39 --output "$OUT" \
  || { echo "FLUXSIZE_EVAL FAILED"; exit 1; }
echo "DONE"; date
