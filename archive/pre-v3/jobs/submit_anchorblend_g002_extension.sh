#!/bin/bash
set -euo pipefail

# Extend the coherent-anchor block at |g|=0.02 from 100 to 500 cases.
#
# Why: the paired amplitude test (results/anchorblend_amplitude_by_dominance_v22_c400-499.json)
# left the BULK gap at g=0.02 unresolved at 1.4 sigma, which is exactly the
# quantity that decides whether the anchor instrument reproduces constgold's
# bulk-dominated carrier (78% of the constgold deficit sits outside the
# response-dominance tail).  Response noise scales like 1/g, so 100 cases at
# g=0.02 carry about twice the SEM of 100 at g=0.05; 500 cases brings the g=0.02
# bulk SEM to roughly 0.0022, matching the g=0.05 precision.
#
# No g=0.05 partners are rendered.  The measured quantity is
# gap = V2.2 prediction - g=0.02 truth, and the prediction side is a
# deterministic function of the input field, so unpaired g=0.02 cases buy SEM
# directly.  The existing block 400--499 keeps its partners and doubles as the
# replication control inside the combined analysis.
#
# Usage:
#   jobs/submit_anchorblend_g002_extension.sh gate    # cheap path-replication check, run FIRST
#   jobs/submit_anchorblend_g002_extension.sh render  # the four 100-case blocks + combined analysis
#
# `render` is multi-hour and writes tens of GB per block under $DATA_DIR.

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
BLOCKS=("500 599" "600 699" "700 799" "800 899")
MODE=${1:-}
cd "$ROOT"

case "$MODE" in
  gate)
    test ! -e "$ROOT/results/anchorblend_g002_gap_by_dominance_v22_c400-499_newpath_paired.json"
    j=$("$SBATCH" --parsable jobs/job_anchorblend_g002_ext_gate.sh)
    echo "gate=$j"
    ;;
  render)
    test -s "$ROOT/results/anchorblend_g002_gap_by_dominance_v22_c400-499_newpath_paired.json" || {
      echo "run the gate first: jobs/submit_anchorblend_g002_extension.sh gate"; exit 1; }
    test ! -e "$ROOT/results/anchorblend_g002_gap_by_dominance_v22_c400-899.json"
    deps=()
    for block in "${BLOCKS[@]}"; do
      set -- $block
      start=$1; stop=$2
      tag="c${start}-${stop}"
      base=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_${tag}
      test -f "configs/fs2_lsst_r_anchorblend_g002_${tag}.yaml"
      test ! -e "$base"
      test ! -e "$ROOT/results/anchorblend_g002_response_v22_${tag}.feather"
      export AB_START="$start" AB_STOP="$stop"
      common=(--parsable --export=ALL,AB_START="$start",AB_STOP="$stop")
      j1=$("$SBATCH" "${common[@]}" --job-name="abg002x_cat_$tag" \
             jobs/job_anchorblend_g002_ext_catalog.sh)
      j2=$("$SBATCH" "${common[@]}" --job-name="abg002x_sim_$tag" \
             --dependency=afterok:"$j1" jobs/job_anchorblend_g002_ext_sim.sh)
      j3=$("$SBATCH" "${common[@]}" --job-name="abg002x_shape_$tag" \
             --dependency=afterok:"$j2" jobs/job_anchorblend_g002_ext_shape.sh)
      j4=$("$SBATCH" "${common[@]}" --job-name="abg002x_resp_$tag" \
             --dependency=afterok:"$j3" jobs/job_anchorblend_g002_ext_response.sh)
      j5=$("$SBATCH" "${common[@]}" --job-name="abg002x_pairs_$tag" \
             --dependency=afterok:"$j1" jobs/job_anchorblend_g002_ext_pairs.sh)
      j6=$("$SBATCH" "${common[@]}" --job-name="abg002x_dom_$tag" \
             --dependency=afterok:"$j5" jobs/job_anchorblend_g002_ext_dominance.sh)
      deps+=("$j4" "$j6")
      printf '%s\n' "$tag: catalog=$j1 sim=$j2 shape=$j3 response=$j4 pairs=$j5 dominance=$j6"
    done
    all=$(IFS=:; echo "${deps[*]}")
    j7=$("$SBATCH" --parsable --dependency=afterok:"$all" \
           jobs/job_analyze_anchor_g002_gap_by_dominance.sh)
    echo "combined_analysis=$j7"
    ;;
  *)
    echo "usage: $0 {gate|render}" >&2
    exit 2
    ;;
esac
