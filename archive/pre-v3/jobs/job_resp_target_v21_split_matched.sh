#!/bin/bash
#SBATCH --job-name=rtsplitm
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rtsplitm_%j.out
# THE QUESTION. 2026-08-05j/k: the fiducial `m` is a cancellation across the V2.1 resolution cut
# (+1.64% well-resolved, -2.21% on the rest); the emulator is unbiased on BOTH halves, so the split
# is the flow's. The flow is trained against the response target, so: does the TARGET carry the same
# sign flip? Build it on the V2.1 domain and on its complement under otherwise identical settings and
# compare each to what constgold demands (R_sim - R_blend), which is 0.8733 on V2.1 and 0.6020 on the
# complement (2026-08-05j, 16 seeds).
# POPULATION MATCHING -- the reason this job exists alongside job_resp_target_v21_split.sh. That
# first version passed no --primary-* cuts, so its COMPLEMENT was the complement of V2.1 within the
# DEFAULT selection cuts: 11.2M pairs against V2.1's 2.6M, a 1:4.3 split where constgold's is 1:1.23.
# It returned global_R = 0.2809 against a constgold demand of 0.6020, but the two describe different
# galaxies and the gap is meaningless. Constgold's complement is the complement WITHIN the flow
# training domain (true mag < 26, Re > 0.3), so those cuts are passed here on BOTH halves. On the
# V2.1 half they should change nothing -- V2.1 lies wholly inside that box (2026-08-05i) -- which
# makes the v21 number a built-in null check on the matching itself.
# Settings otherwise mirror jobs/job_resp_target_v21_grids.sh except for the domain flag.
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
build "--v21-domain --primary-mag-max 26.0 --primary-re-min 0.3"     v21_matched
build "--v21-complement --primary-mag-max 26.0 --primary-re-min 0.3" v21complement_matched
echo
echo "CONSTGOLD DEMANDS (2026-08-05j, 16 seeds, R_sim - R_blend):  V2.1 = 0.8733   COMPLEMENT = 0.6020"
echo "If the target sits BELOW 0.8733 on V2.1 and ABOVE 0.6020 on the complement, the sign flip is"
echo "already in the TARGET and the flow is faithfully reproducing it."
echo RTSPLITM_DONE
