#!/bin/bash
#SBATCH --job-name=domcomp
#SBATCH --time=00:30:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/domcomp_%j.out

# Why did domain-training make m WORSE (in-domain +9.31% vs baseline +3.49%)? Print R_sim / R_flow /
# R_blend separately per mask for BOTH models on the SAME constgold rows, so the term that moved is
# visible instead of inferred. CPU-only: reads existing per-object dumps, no flow sampling.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather

echo "### DOMAIN-TRAINED (s501) ###"; date
python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps/v2dom_perobj_s501.feather" \
  --catalogue "$CAT" --min-case 40

echo; echo "### BASELINE full-range (s501, same seed) ###"
python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_constgold_dumps/v2_perobj_s501.feather" \
  --catalogue "$CAT" --min-case 40
echo "DOMCOMP_DONE"; date
