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
#
# DEFAULTS: n_samples=32, 4 seeds (owner, 2026-07-30). Both are evidence-backed, not guesses:
#   n_samples=32 -- WORKLOG 30i measured the sampling noise in the units the result is quoted in.
#     At size>4.4 it is 0.025 points of m at n=16 against a 5.48-point effect (224x below), and the
#     raw sd shows NO 1/sqrt(n) trend across 16->128, so extra draws buy nothing. 32 is a 2x margin
#     on the smallest tested value.
#   4 seeds -- WORKLOG 30g showed the selection SHIFT is seed-INDEPENDENT (+6.34% at 1 seed vs
#     +6.33% at 16, size>4.4). Seeds move the response NORMALISATION (seed sd 0.607 -> sem 0.30% at
#     n=4), which the m_sel/m_flow split largely removes anyway.
# Together this is 16x less sampling work than 16 seeds x 128 draws.
#
# SUPERSEDED NOTE (kept for history): this ran on ONE seed because the 2026-07-30g comparison showed the selection SHIFT is seed-independent
# (+6.34% vs +6.33% at size>4.4 for 1 vs 16 seeds) -- seeds move the response NORMALISATION, which
# is exactly what the intrinsic mode removes. Iterating on 1 seed is ~3.5 min per mode.
#
# FIREWALL: half-shear legs, ISOLATED (R_blend ~ 0). No emulator, no constgold. Trains nothing.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505}"   # 4 seeds; s504 absent from the dom6x6 set
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### SELECTION ATTRIBUTION (4 seeds, n=32, ISO)  job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

# Output name is NOT _s501: that file backs WORKLOG 30h/30j and must not be clobbered.
# The `|| { ...; exit 1; }` guard MUST NOT sit after a trailing `#` comment -- it becomes part of the
# comment and the failure check silently disappears. That is what happened here before the fix.
python -u scripts/eval_selection_attribution.py --ckpt $CK \
  --max-case 39 --n-samples 32 --batch-size 16384 \
  --size-cuts 2.5 2.9 3.5 4.4 --mag-cuts 24.0 24.5 25.0 \
  --output "$D/selection_attribution_4seed_n32.npz" || { echo SEL_ATTRIB_FAILED; exit 1; }
echo SEL_ATTRIB_DONE; date
