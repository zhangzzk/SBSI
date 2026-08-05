#!/bin/bash
#SBATCH --job-name=rflowv21
#SBATCH --time=01:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rflowv21_%j.out
# WHY. 2026-08-05h exonerated the emulator for V2.1's positive m and left the residual on the flow
# (R_flow 1.66% low). This localises that residual against the flow's OWN training box, to test
# whether V2.1's S/N curve -- which has no magnitude ceiling -- is feeding the flow primaries it
# never trained on. LOCALISATION ONLY: only 8 V2.1 checkpoints exist against the 16 AGENTS.md
# requires for m, so the script prints residuals and never an m.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
set -e
echo "seed dumps present:"; ls -1 $D/ablate_s2c_lt500_v21_perobj_s*.feather | wc -l
python -u scripts/diag_rflow_v21.py \
  --dump-glob "$D/ablate_s2c_lt500_v21_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40
echo RFLOWV21_DONE
