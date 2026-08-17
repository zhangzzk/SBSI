#!/bin/bash
#SBATCH --job-name=popwt
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/popwt_%j.out

# THE DECISIVE TEST before spending any GPU slot on a reweighted retrain: does the response pin close
# only under its OWN training population weighting? The trainer's comment says it does; nobody has
# measured it. m is a ratio of means, so both weightings are computable from dumps already on disk.
#
# Run on the two cut-trained flows AND the full-population flow. The full-population flow is the
# control: its training population is much further from the in-domain evaluation population, so if the
# mechanism is real its weighting gap must be LARGER. A mechanism that shows no such ordering is
# probably an artifact of the reweighting arithmetic rather than a real population effect.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

DOM=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/indist_constgold_dumps
T=/home/z/Zekang.Zhang/SBSI/results

echo "### POPULATION REWEIGHT TEST job=$SLURM_JOB_ID ###"; date

# each: <label> <dumpdir> <tag> <the target that flow was actually TRAINED against>
for spec in "dom6x6:$DOM:ablate_s2c_lt500_dom6x6:$T/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz" \
            "dom2:$DOM:ablate_s2c_coupling_lt500_dom2:$T/response_target_crowd_rblend_snc_c0-99_6x3x5_dom.npz" \
            "v2full:$FULL:indist:$T/response_target_crowd_rblend_snc_c0-99_6x9x5.npz"; do
  L=${spec%%:*}; R1=${spec#*:}; DIR=${R1%%:*}; R2=${R1#*:}; TAG=${R2%%:*}; TGT=${R2#*:}
  echo; echo "############################################################"
  echo "##  $L   target=$(basename "$TGT")"
  echo "############################################################"
  [ -f "$TGT" ] || { echo "SKIP $L: no target $TGT"; continue; }
  python -u scripts/eval_population_reweight.py \
    --dump-dir "$DIR" --tag "$TAG" --target "$TGT" \
    2>&1 | grep -v --line-buffered "module command" || { echo "FAILED on $L"; exit 1; }
done

echo POPWT_DONE; date
