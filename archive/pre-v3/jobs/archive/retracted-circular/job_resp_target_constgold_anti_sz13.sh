#!/bin/bash
#SBATCH --job-name=rtgt_cg13
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=10
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/rtgt_cg13_%j.out
# TRACK 1 iter #6: LARGE-SIZE-RESOLVED metric-consistent constgold target. Root cause of the residual
# size1.0-1.5 additive (-1.99% after sz10cg): pure quantile size edges crowd bins at SMALL sizes
# (sz10cg put 9 of 10 edges below 0.76), lumping the whole large-size region 0.76-1.5 -- where R climbs
# steeply from ~1.0 to >1.5 -- into ONE bin, so a size-blind R_blend cannot bridge the within-bin
# gradient. Fix: explicit --size-edges adding 4 bins above 0.78 (constgold has 0.75<size<1.5 counts of
# ~2.7M, bright-dominated, ample for min-count=200). 6 flux x 13 size x 5 blend. Cases 0-99.
# FIREWALL: a-priori physics choice (resolve response where its gradient is steep), OOS by case, not |m|-tuning.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
cd /home/z/Zekang.Zhang/SBSI
CG=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
OUT=results/response_target_constgold_c0-99_6x13x5.npz
EDGES="0.10,0.155,0.20,0.25,0.31,0.38,0.46,0.55,0.66,0.78,0.90,1.05,1.20,1.50"
echo "### RTGT_CG13 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
echo "size edges = $EDGES"
stdbuf -oL -eL python -B -u scripts/compute_response_target_blend.py \
  --catalogue "$CG" --antithetic --nominal-g 0.02 \
  --n-flux 6 --size-edges "$EDGES" --n-dist 4 --max-case 99 --min-count 200 --output "$OUT" \
  || { echo "RTGT_CG13 FAILED"; exit 1; }
echo "### RTGT_CG13_DONE job=$SLURM_JOB_ID ###"; ls -la "$OUT"; date
