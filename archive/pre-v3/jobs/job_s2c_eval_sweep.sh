#!/bin/bash
#SBATCH --job-name=s2c_eval
#SBATCH --time=02:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2c_eval_%j.out

# Per-lam_theta flux/size response + ghat-leg SHAPE control for the coupling sweep.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
echo "### S2C EVAL SWEEP job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
# baseline S2 (no coupling) for reference, then each lam
for CK in \
  $D/measurement_flow_g0_ngmix_ablate_s2_true4d_s501_swaavg.pt \
  $D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt20_s501_swaavg.pt \
  $D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt100_s501_swaavg.pt \
  $D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s501_swaavg.pt ; do
  echo ""; echo "##### $(basename $CK) #####"
  python -u scripts/eval_fluxsize_response.py --ckpt "$CK" --max-case 39 \
    2>&1 | grep -iE "CONTROL|COUPLING b|OVERALL|b_mag=|ISOLATED|R_size |R_mag "
done
echo "DONE"; date
