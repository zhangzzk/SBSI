#!/bin/bash
#SBATCH --job-name=anchverd
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/anchverd_%j.out

# THE VERDICT on the pre-registered prediction. Enforcing the pin with the global anchor should
# remove term (A) of the decomposition (job 15348680: A=-3.724%, B=+3.212%, total -0.512%), so the
# in-domain m must move to roughly (B) alone = about +3.3%, inside [+2.5,+3.6]. Landing there confirms
# the cancellation end to end; landing near -0.5% falsifies it.
#
# eval_target_vs_constgold re-splits (A) and (B) for the anchored checkpoints, so we see not just the
# new m but whether (A) actually collapsed -- which is the mechanism, not just the outcome.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
T=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz
DUMPS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/indist_constgold_dumps
echo "### ANCHOR VERDICT job=$SLURM_JOB_ID ###"; date
echo "PRE-REGISTERED: in-domain m should move -0.51% -> about +3.3% (window [+2.5,+3.6]),"
echo "and term (A) should collapse from -3.72% toward 0."
for TAG in anch2000 anch10000; do
  n=$(ls $DUMPS/${TAG}_perobj_s*.feather 2>/dev/null | wc -l)
  echo; echo "############ $TAG  ($n dumps) ############"
  [ "$n" -ge 1 ] || { echo "no dumps for $TAG -- skipping"; continue; }
  python -u scripts/eval_target_vs_constgold.py --dump-dir "$DUMPS" --tag "$TAG" --target "$T" \
    2>&1 | grep -v --line-buffered "module command" || echo "FAILED $TAG"
done
echo ANCHVERD_DONE; date
