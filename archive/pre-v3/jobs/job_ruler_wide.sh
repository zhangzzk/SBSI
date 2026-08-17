#!/bin/bash
#SBATCH --job-name=rulerwide
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rulerwide_%j.out

# Is the -3.3% smallest-size gap REAL, or a ~2 sigma fluctuation in the truth sample?
#
# Every variant trained so far shares ONE truth sample (cases 0-39), so their mutual agreement says
# nothing about whether that truth value is right -- the +-1.4% truth error in [0.30,0.38) is COMMON
# to all of them. This re-measures the ruler with cases 0-99 (~2.5x the galaxies, ~1.6x smaller truth
# error) on the unchanged 8-seed dom6x9lows ensemble.
#
# NOTE: --max-case 39 is the default because the nn-isolation lookup only covers 0-39, so the
# ISOLATED table below is INVALID at max-case 99 and must be ignored. Only the "ALL objects" table
# is meaningful here -- it needs no isolation lookup. FIREWALL-CLEAN: half-shear truth only.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
echo "### WIDE RULER (max-case ${MC:-99}) job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selfresp_gap.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_${TAG:-ablate_s2c_lt500_dom6x9lows}_s*_swaavg.pt" \
  --true-re-min 0.3 --true-mag-max 26.0 --max-case ${MC:-99} \
  --output "$D/eval/selfresp_${TAG:-ablate_s2c_lt500_dom6x9lows}_mc${MC:-99}.npz"
echo "RULERWIDE_DONE"; date
