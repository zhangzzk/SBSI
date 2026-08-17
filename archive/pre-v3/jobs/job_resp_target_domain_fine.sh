#!/bin/bash
#SBATCH --job-name=resptgt_fine
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt_fine_%j.out
set -eo pipefail

# FINER in-domain response targets. Justification is the data, not the constgold number:
# the 6x3x5 domain grid still jumps +0.570 -> +0.790 between its first two size bins, and the
# documented limiter (WORKLOG cont.110f) is the flow SMOOTHING a sharp response step in true size.
# In-domain rows ~5.8M, so 6x6x5 (180 cells) and 8x8x5 (320 cells) still leave ~32k / ~18k per cell,
# far above the --min-count 500 floor. Built on the firewall-clean det_meas half-shear legs.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
echo "### FINER DOMAIN RESP TARGETS job=$SLURM_JOB_ID ###"; date

build () {   # $1=n_flux $2=n_size $3=outfile
  echo; echo "--- ${1}x${2}x5 -> $3 ---"
  python -u scripts/compute_response_target_blend.py \
    --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
    --target-cols measured_ngmix_g1 measured_ngmix_g2 \
    --nominal-g 0.05 --crowd-col r_blend \
    --n-flux "$1" --n-size "$2" --n-crowd 5 --min-count 500 --max-case 99 \
    --primary-mag-max 26.0 --primary-re-min 0.3 \
    --snc-lookup $R/g0_lookup_c0-99.feather \
    --output "$3" 2>&1 | grep -v "module command"
  python - "$3" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1]); c = z["counts"]; es, ef = z["edges_size"], z["edges_flux"]
print("  edges_size:", np.round(es, 4))
print(f"  cells={c.size}  empty={int((c==0).sum())}  min/cell={int(c.min()):,}")
w = c.sum(axis=(0,2)); R = z["Rsim"]
Rs = (R*c).sum(axis=(0,2))/np.where(w>0, w, np.nan)
print("  per-size-bin Rsim:", np.round(Rs, 4))
assert es[0] <= 0.3+1e-3 and ef[-1] >= 26.0-1e-3, "grid must align to the domain boundary"
assert int((c==0).sum()) == 0, "empty cells would fall back to a global fill -- reject this grid"
print("  OK aligned, no empty cells")
PY
}

build 6 6 $R/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz
build 8 8 $R/response_target_crowd_rblend_snc_c0-99_8x8x5_dom.npz
echo "RESPTGT_FINE_DONE"; date
