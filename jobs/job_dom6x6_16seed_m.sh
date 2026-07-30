#!/bin/bash
#SBATCH --job-name=dom6_16s
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/dom6_16s_%j.out
set -o pipefail

# The 16-seed dom6x6 in-domain m, from the per-object dumps. Takes the ensemble from 8 seeds
# (-0.508 +- 0.285%) to 16, halving the seed term of the error bar. Pre-registered expectation: the
# CENTRAL value does not move (seeds are a noise average, not a bias lever); only the bar shrinks to
# about +-0.20%. Stated here so a central-value shift is read as evidence, not as success.
#
# The dumps all carry blend_lookup_extnbrho (the certified `_ho` emulator), written by
# jobs/job_s2c_domain_eval.sh. CPU-only: reads existing dumps, trains nothing.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
N=$(ls $D/ablate_s2c_lt500_dom6x6_perobj_s*.feather 2>/dev/null | wc -l)
echo "### DOM6X6 ${N}-SEED m  job=$SLURM_JOB_ID ###"; date
ls $D/ablate_s2c_lt500_dom6x6_perobj_s*.feather | sed 's|.*_s|  seed |;s|\.feather||'

python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$D/ablate_s2c_lt500_dom6x6_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.3 --mag-max 26.0 2>&1 | grep -v "module command"

echo; echo "### (A)/(B) split at ${N} seeds ###"
python -u scripts/eval_target_vs_constgold.py --tag ablate_s2c_lt500_dom6x6 2>&1 | grep -v "module command"
echo "DOM6_16S_DONE"; date
