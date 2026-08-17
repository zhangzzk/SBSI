#!/bin/bash
#SBATCH --job-name=dom6grid
#SBATCH --time=01:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/dom6grid_%j.out

# dom6x6 (the best cut-trained V2 flow) x every available emulator, on both populations.
# CPU-only from the 8 per-object dumps that already exist in v2_domain_dumps -- the GPU array that
# was queued to regenerate them (15348178) was redundant and has been cancelled.
#
# The dumps were written by jobs/job_s2c_domain_eval.sh, which passes
# blend_lookup_extnbrho_c40-139.feather, so the dump's R_blend column is unambiguously the certified
# `_ho` emulator. No --old-lookup override is needed.
#
# `_indom` is the emulator RETRAINED on the cut population; its regression cuts are mag 18-26 /
# Re 0.3-1.5, so it is only valid on the in-domain population and is deliberately NOT run wide.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

DOM=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
R=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results
TAGPAT="ablate_s2c_lt500_dom6x6_perobj_s*.feather"

echo "### DOM6X6 GRID job=$SLURM_JOB_ID ###"; date

run () {  # run <label> <lookup> <extra...>
  echo; echo "############################################################"
  echo "##  dom6x6  vs  $1"
  echo "############################################################"
  python -u scripts/eval_swap_lookup.py --dump-dir "$DOM" --pattern "$TAGPAT" \
    --lookup "$2" --new-label "$1" "${@:3}" 2>&1 | grep -v --line-buffered "module command" \
    || { echo "FAILED on $1"; exit 1; }
}

run wc5-CUT   "$R/blend_lookup_wc5_c40-139.feather"   --apply-cuts
run indom-CUT "$R/blend_lookup_indom_c40-139.feather" --apply-cuts
run wc5-WIDE  "$R/blend_lookup_wc5_c40-139.feather"

echo DOM6GRID_DONE; date
