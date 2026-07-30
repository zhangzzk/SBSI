#!/bin/bash
#SBATCH --job-name=tgtvcg
#SBATCH --time=01:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/tgtvcg_%j.out

# THE FORK IN THE ROAD: split the in-domain m into (A) the flow missing its pin target and (B) the
# target not being the quantity constgold needs. (A) is fixable by training, (B) is not. Deciding
# this before spending GPU hours is the whole point. CPU-only from existing dumps.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
T=/home/z/Zekang.Zhang/SBSI/results
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
echo "### TARGET vs CONSTGOLD job=$SLURM_JOB_ID ###"; date
for spec in "dom6x6:ablate_s2c_lt500_dom6x6:$T/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz" \
            "dom2:ablate_s2c_coupling_lt500_dom2:$T/response_target_crowd_rblend_snc_c0-99_6x3x5_dom.npz"; do
  L=${spec%%:*}; R=${spec#*:}; TAG=${R%%:*}; TGT=${R#*:}
  echo; echo "############ $L ############"
  python -u scripts/eval_target_vs_constgold.py --dump-dir "$D" --tag "$TAG" --target "$TGT" \
    2>&1 | grep -v --line-buffered "module command" || { echo "FAILED $L"; exit 1; }
done
echo TGTVCG_DONE; date
