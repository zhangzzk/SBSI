#!/bin/bash
#SBATCH --job-name=domv21
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/domv21_%j.out

# STEP 1 of V2.1: fit the two constants of the true-property S/N proxy, and report the population
# the V2.1 domain leaves. Everything else in V2.1 -- the response target, the flow, the emulator,
# the constgold evaluation -- is blocked on this, because `sbs_shear.domain.sn_true` RAISES until
# SN_SKY_VAR / SN_GAIN are set rather than falling back to a default and quietly moving the cut.
#
# V2.1 domain (owner, 2026-08-04): primary true Re > 0.5" (= 2.5 px, resolution R = 0.474) AND
# true S/N > 10. The size unit is PIXELS -- see sbs_shear/domain.py for why that is the only
# reading consistent with "resolution factor roughly 0.5".
#
# CPU only, no GPU: two streaming passes over the 17 GB g=0 training catalogue.
# FIREWALL: constgold is never opened.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### V2.1 DOMAIN CALIBRATION job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_domain_v21.py 2>&1 | grep -v --line-buffered "module command" \
  || { echo "DOMV21_FAILED"; exit 1; }
date
