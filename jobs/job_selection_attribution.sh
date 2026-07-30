#!/bin/bash
#SBATCH --job-name=sel_attrib
#SBATCH --time=00:45:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/sel_attrib_%j.out

# Split the Stage-2 selection residual into SELECTION vs shape bleed-through (owner, 2026-07-30).
#
# The gate's m mixes shape-response error with selection-response error, so the -5.97% at size>4.4
# cannot be attributed. This runs the SAME cuts twice: once as now (project the flow's sampled
# MEASURED shape) and once projecting the EXACT sheared INTRINSIC shape on both sides while still
# SELECTING on measured size/mag. In the second mode both sides average identical noise-free shapes,
# so any sim-vs-model difference is purely WHICH objects got selected.
#
# READ THE CORRECTNESS CHECK FIRST. Intrinsic-mode NO-CUT m must be 0 to float precision (same
# shapes, same objects, no cut). The script prints PASS/FAIL; on FAIL the table means nothing.
#
# ONE SEED on purpose: the 2026-07-30g comparison showed the selection SHIFT is seed-independent
# (+6.34% vs +6.33% at size>4.4 for 1 vs 16 seeds) -- seeds move the response NORMALISATION, which
# is exactly what the intrinsic mode removes. Iterating on 1 seed is ~3.5 min per mode.
#
# FIREWALL: half-shear legs, ISOLATED (R_blend ~ 0). No emulator, no constgold. Trains nothing.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CK=$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt
echo "### SELECTION ATTRIBUTION (1 seed, ISO)  job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

python -u scripts/eval_selection_attribution.py --ckpt "$CK" \
  --max-case 39 --n-samples 128 --batch-size 16384 \
  --size-cuts 2.5 2.9 3.5 4.4 --mag-cuts 24.0 24.5 25.0 \
  --output "$D/selection_attribution_s501.npz" || { echo SEL_ATTRIB_FAILED; exit 1; }
echo SEL_ATTRIB_DONE; date
