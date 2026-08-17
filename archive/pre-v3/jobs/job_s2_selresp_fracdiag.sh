#!/bin/bash
#SBATCH --job-name=s2_fracdiag
#SBATCH --time=01:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2_fracdiag_%j.out

# size>4.4 gap diagnosis: model kept-fraction (fracM) vs truth (fracS). ISOLATED only, 1 seed.
# If fracM != fracS(~0.43) at size>4.4 -> flow size-MARGINAL miscalibration; else covariance issue.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
echo "### S2 FRAC DIAG job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_response.py \
  --ckpt $D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s501_swaavg.pt \
  --max-case 39 --n-samples 128 --batch-size 16384 \
  --size-cuts 2.5 2.9 3.5 4.4 5.0 5.5 --mag-cuts 24.0 24.5 25.0
echo "S2_FRACDIAG_DONE"; date
