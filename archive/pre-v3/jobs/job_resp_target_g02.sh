#!/bin/bash
#SBATCH --job-name=resp_g02
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resp_g02_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# STEP 1 of the g=0.02-target experiment (WORKLOG cont.17). Rebuild the flow's response-supervision
# target at the SAME shear amplitude the flow is supervised AND validated at (0.02), instead of the
# current 0.05. Rationale: the target is a secant [e(g)-e(0)]/g (effective slope at g/2), while the
# flow's response term and the constgold validation both probe +/-0.02 about 0 -> a ~0.025-vs-0
# amplitude mismatch that imprints any response curvature as bias. Building the target at g=0.02
# makes target == supervision == validation, removing the linearity assumption in the direction that
# matters. Only --nominal-g and --catalogue (0.02 render) and --max-case (0-199) differ from
# job_resp_target_crowd_snc.sh; binning (6x3x5, r_blend) and estimator (SNC, g0-subtraction) match.
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_g0.02_test_full.feather      # main-sim g=0.02 render (has r_blend + ngmix); cases 0-99 only
LK=results/g0_lookup_c0-99.feather                 # EXISTING g0 SNC lookup (same 100 cases)
OUT=results/response_target_crowd_rblend_snc_g02_c0-99_6x3x5.npz

# NOTE: the g=0.02 render only covers cases 0-99 (preflight), so this target is 100-case like the
# current g=0.05 one -- a clean AMPLITUDE-only A/B on the SAME case set. (A 200-case version would
# first need det+meas on the g=0.02 render for cases 100-199; see job_g0_lookup_0-199.sh / runbook.)
echo "### SNC response target at g=0.02, cases 0-99 -> $OUT ###"
echo "catalogue=$CAT  snc_lookup=$LK"; date
python -u scripts/compute_response_target_blend.py \
  --catalogue $CAT \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.02 \
  --crowd-col r_blend \
  --n-flux 6 --n-size 3 --n-crowd 5 \
  --min-count 500 \
  --max-case 99 \
  --snc-lookup "$LK" \
  --output $OUT 2>&1 | grep -v "module command"
date; echo "RESP_G02_DONE $OUT"
