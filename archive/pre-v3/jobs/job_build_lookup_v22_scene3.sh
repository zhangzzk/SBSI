#!/bin/bash
#SBATCH --job-name=lk_v22_s3
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v22_s3_%j.out
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT=results/blend_lookup_v22_scene3_c40-139.feather; [ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
echo "### V2.2 SCENE3 CONSTGOLD LOOKUP job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/build_blend_lookup_scene3.py --cases $(seq 40 139) --output "$OUT"
echo LK_V22_SCENE3_DONE; date
