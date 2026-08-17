#!/bin/bash
#SBATCH --job-name=eval_v22_s3
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/eval_v22_s3_%j.out
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_scene3/response_catalogue_train.feather
NEW=results/blend_lookup_v22_scene3_c40-139.feather; OLD=results/blend_lookup_v22_c40-139.feather
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_dumps
for f in "$CAT" "$NEW" "$OLD"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
echo "### SCENE3 AGAINST OWN HALF-SHEAR LABELS ###"; date
"$PY" -u scripts/eval_emu_label_gap.py --tag lsst_r_extnbr_v22_scene3 --cat "$CAT" --heldout-min 40 \
  --true-re-min 0.5 --true-mag-max 25.8 --output results/emu_label_gap_v22_scene3.npz
echo "### BLIND CONSTGOLD SWAP, SAME 16 V2.2 FLOW SEEDS ###"; date
"$PY" -u scripts/eval_swap_lookup.py --dump-dir "$D" --pattern 'ablate_s2c_lt500_v22_perobj_s*.feather' \
  --lookup "$NEW" --old-lookup "$OLD" --new-label v22_scene3 --old-label v22 --apply-cuts \
  --cut-catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --true-mag-max 25.8 --true-re-min 0.5
echo EVAL_V22_SCENE3_DONE; date
