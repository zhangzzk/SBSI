#!/bin/bash
#SBATCH --job-name=resp_isoblend_snc
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resp_isoblend_snc_%j.out

# Build the ISOLATED-SEPARATED snc self-response target grid for the joint-flow port (cont.60).
# Unlike the crowd (r_blend) grid, this uses compute_response_target_blend.py's DEFAULT mode:
#   blend bin 0 = ISOLATED (neighbored==False), bins 1..n_dist = distance quantiles for BLENDED.
# So it needs only neighbored + distance (both in det_meas_ngmix_g0.05_val) -- NO r_blend, which
# is unavailable for cases 0-99.  --snc-lookup makes the response BOTH-LEGS-FREE (uses the g=0
# ngmix0 lookup as the noiseless baseline, so only detection at g=0.05 is required -- avoids the
# 22.3% both-legs-finite collapse that biased the cont.59 hs retrain's isolated target).
# Output -> train_forward_prototype.py --target-npz (production-guided isolated recipe).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI

NF=${1:-6}; NS=${2:-3}; NDIST=${3:-4}; MAXCASE=${4:-99}
LK=${5:-results/g0_lookup_c0-99.feather}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=results/response_target_isoblend_snc_c0-${MAXCASE}_${NF}x${NS}x$((NDIST+1)).npz

echo "### ISOBLEND SNC response target (bin0=isolated + $NDIST dist bins), cases <= $MAXCASE ###"
echo "snc_lookup=$LK  out=$OUT"; date
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_ngmix_g0.05_val.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 \
  --n-flux $NF --n-size $NS --n-dist $NDIST \
  --min-count 500 \
  --max-case $MAXCASE \
  --snc-lookup "$LK" --snc-cols ngmix0_g1 ngmix0_g2 \
  --output $OUT 2>&1 | grep -v module
date
echo "RESP_ISOBLEND_SNC_DONE $OUT"
