#!/bin/bash
#SBATCH --job-name=cg_nd
#SBATCH --time=10:00:00
#SBATCH --mem=64G          # MEASURED peak: 20.8G (v21 4-seed, job 15527267), 19.6-26.3G on the
                           # fiducial table (15385639/15389640). Memory is dominated by the ONE
                           # catalogue load, not by seed count, so 8 or 16 seeds cost no more
                           # than 4. The old 200G request was ~8x the peak and made the job pend
                           # on Resources for 14h behind FREE GPUs -- cip-cl-nv01 had 3 a40 idle
                           # and only 67G RAM left. Over-requesting memory costs queue time.
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_nd_%j.out

# constgold selection table at NEAR-DOMAIN cuts using the REAL per-leg measured mag/size added by
# job 15366166. Sim and model cut on the SAME quantity at the SAME absolute threshold -- no proxy
# except the (*) real-S/N rows. Reports ALL and ISOLATED; m_flow is only a clean model bias on the
# ISOLATED block (the ALL block's column (3) carries the neighbour term the flow does not model).
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
# 16 seeds: fig4 reports `m`, a bias on the SHAPE (e) response, and the convention sets the
# e-response standard at 16 regardless of what the CUTS are on. The cuts here are flux/size (4-seed
# standard) but they only decide WHICH objects enter the average; the number reported is the shape
# response of that subset. At 4 seeds the m column carried +-0.43%, too coarse for the +-0.3% target.
SEEDS="${SEEDS:-501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517}"
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### CONSTGOLD NEAR-DOMAIN job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_constgold_neardomain.py --ckpt $CK --n-samples 32 --batch-size 16384 --max-rows 0 \
  2>&1 | grep -v --line-buffered "module command" || { echo CG_ND_FAILED; exit 1; }
echo CG_ND_ALL_DONE; date
