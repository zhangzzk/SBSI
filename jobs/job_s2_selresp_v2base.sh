#!/bin/bash
#SBATCH --job-name=selresp_v2b
#SBATCH --time=04:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/selresp_v2b_%j.out

# Stage-2 SELECTION-response gate on the NEW BASELINE V2 = dom6x6 flow, 16 seeds (+ the old `_ho`
# emulator, though the emulator does not enter here -- see FIREWALL below).
#
# Supersedes jobs/job_s2_selection_response.sh, which globbed
# `ablate_s2c_coupling_lt500_s50*_swaavg.pt` -- the V2 FULL-RANGE flow, moved to
# _QUARANTINE_2026-07-30 by the 2026-07-30f cleanup. That job can no longer run as written; this one
# replaces it by pointing at the kept dom6x6 checkpoints. Every other parameter is byte-identical to
# the old invocation (--max-case 39 --n-samples 128 --batch-size 16384 --all-too) so the new numbers
# are directly comparable to the 8-seed curve in SBSI/figures/fig_selection_bias_sim_vs_flow.png.
#
# WHAT IT MEASURES. A moving MEASURED cut is pushed through the flow's joint (shape, mag, log-size)
# and the SELECTED-CATALOGUE mean response is compared to the half-shear sim:
#   R_C = [ <e.ghat>_S(g=0.05) - <e.ghat>_S(g=0) ] / g_med ,  m_C = R_sim,C / R_model,C - 1
# The two-means form is required because a measured cut has pass(g0) != pass(gS) -- objects cross the
# boundary as the measured observable itself responds to shear. That crossing IS the selection term.
#
# BUILT-IN NULL TEST: for a TRUE-property cut pass(g0) == pass(gS), so the two-means form collapses
# to the fixed-subset self-response and the selection term must vanish. `--true-size-cuts` supplies
# it for free; if those rows are not ~ the no-cut row, the harness itself is wrong and the measured
# numbers mean nothing. CHECK THE NULL FIRST.
#
# FIREWALL: ISOLATED objects only (no brighter true neighbour within 7"), so R_blend ~ 0 -- no
# emulator, no constgold anywhere in this test. Truth is the det_meas half-shear catalogue on matched
# both-detected pairs. Nothing is trained or fitted; existing checkpoints are only scored.
#
# PRE-REGISTERED EXPECTATION. The 8-seed figure agrees well on the flux axis (sim 13.8% vs flow 11.9%
# at the tightest mag cut) and on size out to 0.7" (both ~9.3%), with ONE divergence at the deep size
# tail, where the sim TURNS OVER (6.8%) while the flow keeps climbing (11.5%). Doubling to 16 seeds
# halves the flow's seed spread but should NOT move the flow's central curve. So the deep-size-tail
# gap is expected to SURVIVE. If it shrinks materially, it was seed noise and the earlier -6.6%
# reading was overstated; if it holds, it is a real deficiency of the joint at the size tail.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
GLOB="$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s*_swaavg.pt"
N=$(ls $GLOB 2>/dev/null | wc -l)
echo "### S2 SELECTION RESPONSE -- BASELINE V2 (dom6x6, ${N} seeds)  job=$SLURM_JOB_ID ###"
nvidia-smi -L; date
[ "$N" -eq 16 ] || echo "WARNING: expected 16 seeds, found $N -- check the 2026-07-30f cleanup"

python -u scripts/eval_selection_response.py \
  --ckpt-glob "$GLOB" \
  --max-case 39 --n-samples 128 --batch-size 16384 --all-too \
  --output "$D/selection_response_v2base_dom6x6_16seed.npz" \
  || { echo "SELRESP_V2B_FAILED"; exit 1; }
echo "SELRESP_V2B_DONE"; date
