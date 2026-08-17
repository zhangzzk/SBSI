#!/bin/bash
#SBATCH --job-name=tunedcg
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/tunedcg_%j.out

# Does the Optuna-TUNED blend emulator change the certified constgold m?  NO-SELECTION test:
# global m only, no measured cuts anywhere.
#
# METHOD.  No flow rerun and no GPU.  The 16 CERTIFIED Gold-v1 per-object dumps
# (sbsi_dumps/fig2_perobj_s5*_fixresp.feather, flow `meas_szfl_noz_lam450_fixresp`, seeds 501-516,
# --min-case 40) already carry per-object (case, input_index, r_sim, R_flow, R_blend), and
# validate_constant_with_blend.py uses the lookup R_blend unmodified (rb_add = rb_i), so swapping
# emulators is a JOIN:   m = <r_sim> / ( <R_flow> + <R_blend from the other lookup> ) - 1.
# r_sim and R_flow are emulator-independent.  This is the same eval_swap_lookup.py path that
# produced the published _ho -> wc5 numbers (job 15348008).
#
# CONTROL, printed first and not optional: with the dump's OWN R_blend (= certified `_ho`) the
# 16-seed ensemble must reproduce +0.261% (job 15348008; certified record +0.245%).  If it does
# not, the swapped numbers mean nothing.
#
# THREE EMULATORS, so tuning is separated from the domain narrowing it was built on top of:
#   `_ho`            certified/production baseline          (dump's own R_blend column)
#   `_indom`         = `_ho` config with the PRIMARY training population narrowed to mag<26,
#                      Re>0.3; hyperparameters inherited, untuned
#   `_indom_tuned`   = `_indom` config with model_tag changed ONLY, then Optuna-tuned (73 trials)
# So `_indom` -> `_indom_tuned` is the PURE TUNING effect (identical cuts => identical coverage),
# while `_ho` -> `_indom_tuned` also carries the domain narrowing and its coverage change.
#
# TWO POPULATIONS.  WIDE is the certified convention (mag 18-28, Re 0.1-1.5).  CUT is the
# deliverable TRUE-property population (mag<26, Re>0.3) -- true properties, NOT a measured cut.
# The `_indom*` emulators only score inside their training cuts, so on WIDE ~76% of rows fall back
# to R_blend=0; that column is reported for completeness but is a coverage artifact.
#
# FIREWALL: reads constgold dumps and emulator lookups; trains, fits and selects nothing.
# Promotion of the tuned emulator must be argued on the per-pair ruler (eval_rblend_gap.py),
# NEVER on these numbers.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

R=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results
V1=/project/ls-gruen/users/zekang.zhang/sbsi_dumps
PAT="fig2_perobj_s*_fixresp.feather"
TUNED=$R/blend_lookup_indomtuned_c40-139.feather
INDOM=$R/blend_lookup_indom_c40-139.feather

echo "### TUNED-EMULATOR CONSTGOLD job=$SLURM_JOB_ID ###"; date
[ -f "$TUNED" ] || { echo "MISSING $TUNED -- run jobs/job_build_lookup_indomtuned.sh first"; exit 1; }

python -u scripts/check_emulator_provenance.py --tag lsst_r_extnbr_indom_tuned \
  || { echo "TUNEDCG_ABORTED (provenance guard)"; exit 1; }

run () {  # run <banner> <args...>
  echo; echo "############################################################"
  echo "##  $1"
  echo "############################################################"
  python -u scripts/eval_swap_lookup.py --dump-dir "$V1" --pattern "$PAT" "${@:2}" \
    2>&1 | grep -v --line-buffered "module command" || { echo "FAILED on $1"; exit 1; }
}

# (1) THE REQUESTED COMPARISON, certified WIDE convention. Baseline = dump's own `_ho` R_blend,
#     controlled against the certified +0.245% / measured +0.261%.
run "WIDE  certified _ho (dump column)  vs  _indom_tuned" \
    --lookup "$TUNED" --new-label indom_tuned --old-label ho --certified-m 0.245

# (2) Same comparison on the deliverable TRUE-property population, where `_indom_tuned` is valid.
run "CUT   certified _ho (dump column)  vs  _indom_tuned   (mag<26, Re>0.3)" \
    --lookup "$TUNED" --new-label indom_tuned --old-label ho --apply-cuts

# (3) PURE TUNING effect, in-domain population: same cuts on both sides => same coverage.
run "CUT   _indom (untuned)             vs  _indom_tuned   (mag<26, Re>0.3)" \
    --lookup "$TUNED" --new-label indom_tuned --old-lookup "$INDOM" --old-label indom --apply-cuts

# (4) PURE TUNING effect, wide population (both sides equally coverage-limited, so the SHIFT is
#     still tuning-only even though neither absolute m is usable).
run "WIDE  _indom (untuned)             vs  _indom_tuned" \
    --lookup "$TUNED" --new-label indom_tuned --old-lookup "$INDOM" --old-label indom

echo TUNEDCG_ALL_DONE; date
