#!/bin/bash
#SBATCH --job-name=rtsplit
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rtsplit_%j.out
# THE QUESTION. 2026-08-05j/k: the fiducial `m` is a cancellation across the V2.1 resolution cut
# (+1.64% well-resolved, -2.21% on the rest); the emulator is unbiased on BOTH halves, so the split
# is the flow's. The flow is trained against the response target, so: does the TARGET carry the same
# sign flip? Build it on the V2.1 domain and on its complement under otherwise identical settings and
# compare each to what constgold demands (R_sim - R_blend), which is 0.8733 on V2.1 and 0.6020 on the
# complement (2026-08-05j, 16 seeds).
# Settings mirror jobs/job_resp_target_v21_grids.sh exactly except for the domain flag.
# FIREWALL: the target is built from HALF-SHEAR only. The constgold numbers are used for COMPARISON
# in the write-up, never fed into the build, and nothing here is fitted or corrected.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
set -e
build () {   # $1 = domain flag, $2 = label
  OUT=$R/response_target_crowd_rblend_snc_c0-99_6x3x5_${2}.npz
  echo; echo "=================== $2  ($1) ==================="
  python -u scripts/compute_response_target_blend.py \
    --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
    --target-cols measured_ngmix_g1 measured_ngmix_g2 \
    --nominal-g 0.05 --crowd-col r_blend \
    --n-flux 6 --n-size 3 --n-crowd 5 --min-count 500 --max-case 99 \
    $1 \
    --snc-lookup $R/g0_lookup_c0-99.feather \
    --output "$OUT" 2>&1 | grep -v --line-buffered "module command" | tail -8
  python - "$OUT" "$2" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
print(f"  {sys.argv[2]}: global_R = {float(z['global_R']):.4f}   domain stamp = {str(z['domain'])[:70]}")
PY
}
build "--v21-domain"     v21
build "--v21-complement" v21complement
echo
echo "CONSTGOLD DEMANDS (2026-08-05j, 16 seeds, R_sim - R_blend):  V2.1 = 0.8733   COMPLEMENT = 0.6020"
echo "If the target sits BELOW 0.8733 on V2.1 and ABOVE 0.6020 on the complement, the sign flip is"
echo "already in the TARGET and the flow is faithfully reproducing it."
echo RTSPLIT_DONE
