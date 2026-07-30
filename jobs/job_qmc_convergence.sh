#!/bin/bash
#SBATCH --job-name=qmc_conv
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/qmc_conv_%j.out

# Can randomized QMC cut n_samples in the selection harness? ONE seed (s501), ISOLATED.
#
# n_samples is the dominant cost: the 16-seed run is 16 ckpt x 2 legs x 1.02M gal x 128 draws =
# 4.16 BILLION forward passes, ~11.5 PFLOP. The per-galaxy quantity is a Monte Carlo integral with
# no closed form (the coupling layers mix all 4 dims, so there is no marginal CDF to integrate), so
# the lever is a better sampler, not analytics.
#
# Measures the SCATTER of R_model across independent sampling seeds at each (mode, n) -- an
# assumption-free read of estimator noise, needing no assumed ground truth. Also reports each mode's
# MEAN: RQMC is unbiased only because every object gets its own random shift, so if the means
# separate the construction is wrong and the speedup is not real.
#
# Cost: sum over n of repeats*2 modes*n draw-units. With n=16,32,64,128 and 3 repeats that is
# 6*(16+32+64+128) = 1440 units against 128 units ~ 208s on this data => ~40 min plus one data load.
#
# GPU: any card. TF32 is irrelevant here (both modes run the same precision, so it cancels), and
# the last two runs landed on a V100 and a 2080 Ti, both pre-Ampere.
#
# FIREWALL: half-shear legs, ISOLATED (R_blend ~ 0). No emulator, no constgold. Trains nothing.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CK=$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt
echo "### QMC CONVERGENCE (1 seed, ISO)  job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

python -u scripts/eval_qmc_convergence.py --ckpt "$CK" \
  --n-list 16 32 64 128 --repeats 3 --batch-size 16384 \
  --size-cuts 3.5 4.4 --mag-cuts 24.0 \
  --output "$D/qmc_convergence_s501.npz" || { echo QMC_CONV_FAILED; exit 1; }
echo QMC_CONV_ALL_DONE; date
