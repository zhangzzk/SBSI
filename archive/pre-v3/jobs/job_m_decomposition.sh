#!/bin/bash
#SBATCH --job-name=mdecomp
#SBATCH --time=01:30:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/mdecomp_%j.out

# Localise the in-domain m residual. Pure pandas over per-object dumps that already exist -- no GPU
# and no flow rerun, so this deliberately asks for no --gres and goes to the unsaturated partition.
# Three flows are decomposed side by side so we can see whether the residual STRUCTURE changed when
# the flow was retrained on the cut population, or only its overall level.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

DOM=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps
FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/indist_constgold_dumps

echo "### M DECOMPOSITION job=$SLURM_JOB_ID ###"; date

for spec in "dom6x6:$DOM:ablate_s2c_lt500_dom6x6" \
            "dom2:$DOM:ablate_s2c_coupling_lt500_dom2" \
            "v2full:$FULL:indist"; do
  NAME=${spec%%:*}; REST=${spec#*:}; DIR=${REST%%:*}; TAG=${REST#*:}
  echo; echo "############################################################"
  echo "##  $NAME   (tag=$TAG)"
  echo "############################################################"
  python -u scripts/eval_m_decomposition.py --dump-dir "$DIR" --tag "$TAG" \
    ${EXTRA} 2>&1 | grep -v --line-buffered "module command" \
    || { echo "FAILED on $NAME"; exit 1; }
done

echo MDECOMP_DONE; date
