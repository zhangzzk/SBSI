#!/bin/bash
#SBATCH --job-name=sel_intr
#SBATCH --time=01:30:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/sel_intr_%j.out

# SELECTION bias on constgold +/-0.02 with the UNSHEARED INTRINSIC shape as the numerator:
#   R_sel = (<e_int>_plus[pass_plus] - <e_int>_minus[pass_minus]) / 0.04
# Both-detected pairs, cut applied separately in each leg on that leg's OWN measured observable,
# so the boundary moves with shear and the whole signal is pure selection (no-cut == 0 exactly).
# Scanned over measured mag 24.5..26.5 and measured size 0.3..0.7". Gold-V2 8-seed flow vs sim.
# Restricted to the FLOW TRAINING DOMAIN (true Re>0.3, true mag<26.0) -- the same domain_cut
# eval_selection_response.py applies. Outside it the mag/size coupling pin is a global-constant fill
# (130 of 270 target cells have counts==0), so the flow is pinned to a constant there, not to data.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selection_intrinsic_domain.npz
echo "### SELECTION INTRINSIC job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_intrinsic.py \
  --min-case 40 --max-case 139 --re-min 0.3 --mag-max 26.0 \
  --mag-cuts 24.5 24.75 25.0 25.25 25.5 25.75 26.0 26.25 26.5 \
  --size-cuts 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --n-samples 128 --batch-size 16384 --chunk 250000 \
  --model-max-rows 1500000 --n-boot 300 \
  --output $OUT
echo "SEL_INTR_DONE"; date
