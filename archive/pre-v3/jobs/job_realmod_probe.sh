#!/bin/bash
#SBATCH --job-name=realmod
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/realmod_%j.out

# STAGE P0 -- the decisive, CPU-only measurement that gates the whole realisation-aware (RA) build.
#
# The RA design exists to fix a defect diagnosed on CONSTGOLD (5.4x response over-prediction at
# measured size < 0.60"; +46.2% at measured mag 26.25-26.5). The firewall forbids training on
# constgold, so the modulation target has to come from the HALF-SHEAR legs. If the same realisation
# dependence is not visible there, every RA design is dead and the correct output is "blocked" --
# NOT a constgold-derived target. This job answers that, for free, from two artifacts that already
# exist: results/halfshear_selfresp.feather (per-object r_sim_self + 16 seeds of R_flow) and the
# matched both-detected base that eval_selection_response.build_base already knows how to make.
#
# PASS (pre-registered, see WORKLOG): count-weighted rms of |rho_sim/rho_model - 1| must exceed
#   (a) 5x its own propagated sem, (b) 3x the 45-degree null, (c) 3x the <e_0.ghat>/g contamination.
# Also required: the faint/small extreme bin depressed by >3 sigma, and FIT vs HELD OUT agreeing.
#
# FIREWALL: half-shear legs only. Nothing is trained, fitted, tuned or selected here.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

OUTD=${OUTD:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/ra}
mkdir -p "$OUTD"

# RUN AT SEVERAL CELL GRANULARITIES. The smoke run (job 15423571) showed that with very coarse
# cells the realisation-BLIND fiducial model already reproduces most of rho -- the measured sub-bin
# is then just proxying for TRUE properties the model does see. The signal
# rho_sim/rho_model - 1 is what isolates the genuinely realisation-driven part, and its size depends
# on how fine the cell is. Deciding the granularity here, on half-shear only, is firewall-clean;
# deciding it later on a constgold number would not be.
echo "### P0 REALISATION-MODULATION PROBE job=$SLURM_JOB_ID ###"; date
for GRID in ${GRIDS:-"6x2" "6x6" "3x1"}; do
  NF=${GRID%x*}; NS=${GRID#*x}
  echo; echo "======== cells ${NF} true-mag x ${NS} true-size ========"
  python -u scripts/eval_realisation_modulation.py \
    --max-case ${MAXCASE:-39} --fit-max-case ${FITCASE:-19} \
    --n-flux "$NF" --n-size "$NS" \
    --n-dmag ${NDMAG:-5} --n-dlogsize ${NDLSZ:-2} \
    --output "$OUTD/realmod_probe_${NF}x${NS}.npz" 2>&1 \
    | grep -v --line-buffered "module command" || { echo REALMOD_FAILED; exit 1; }
done
echo REALMOD_DONE; date
