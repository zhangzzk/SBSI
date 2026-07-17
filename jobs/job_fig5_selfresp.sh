#!/bin/bash
#SBATCH --job-name=fig5_selfresp
#SBATCH --time=00:40:00
#SBATCH --mem=22G
#SBATCH --cpus-per-task=6
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --output=/home/z/Zekang.Zhang/logs/fig5_selfresp_%j.out

# Figure 5: flow SELF-response R_model per flux/size bin on the g=0 training
# catalogue (s501), vs the truth R_sim target. Bounded a40 job (cip has idle
# a40 GPUs; inter a40 queue is ~2 days backed up). Writes
# results/fig5_selfresp_bins_s501.npz -> plotting/plot_flow_figures.py figure5().
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

echo "### FIG5_SELFRESP job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
stdbuf -oL -eL python -u scripts/eval_self_response_bins.py \
  --measurement-model models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt \
  --output results/fig5_selfresp_bins_s501.npz \
  --max-rows 2000000 --n-samples 32 --batch-size 32768 --delta 0.02 --flow-seed 12345 \
  2>&1 | grep --line-buffered -vE "module command"
echo "FIG5_SELFRESP_DONE"; date
