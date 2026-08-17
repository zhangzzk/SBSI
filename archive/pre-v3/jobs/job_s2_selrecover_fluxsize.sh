#!/bin/bash
#SBATCH --job-name=s2_selrec
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2_selrec_%j.out

# Stage-2 model-recovery on the FLUX (mag) and SIZE (flux_radius) axes: does the pinned flow reproduce
# the truth selection bias when the SAME moving measured cut is applied to the flow's own outputs?
# det_meas g0 <-> g0.02 matched both-detected pairs (+/-0.02-magnitude via per-object ghat isotropy).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "### S2 SELREC (flux/size) job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_response.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s50*_swaavg.pt" \
  --g0-leg $CAT/det_meas_ngmix_g0.0_train.feather \
  --gS-leg $CAT/det_meas_ngmix_g0.02_test.feather \
  --max-case 39 --n-samples 128 --batch-size 16384 \
  --size-cuts 2.9 3.5 4.4 --mag-cuts 24.5 25.0 25.5 26.0 \
  --true-size-cuts 0.4 0.5 --all-too
echo "S2_SELREC_DONE"; date
