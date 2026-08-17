#!/bin/bash
#SBATCH --job-name=bs_resp
#SBATCH --time=00:40:00
#SBATCH --mem=24G          # the twin build (job_resp_target_v21_g002.sh) measured 869M peak on the
                           # same catalogue and cuts; this one additionally holds the per-row proj /
                           # weight / label arrays (~5 float64 vectors over ~2.6M rows = ~100M) and
                           # a 4,000 x 100 bootstrap multiplier matrix (3M). 24G is generous.
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/bs_resp_%j.out

# HOW WELL IS THE RESPONSE TARGET'S 0.8509 DETERMINED?
#
# WORKLOG 2026-08-04p attributes the whole of V2.1's +0.790% m to a 2.53% gap between the target's
# population-weighted response (0.8509) and what constgold demands (R_sim - R_blend = 0.8724). That
# entry quoted NO ERROR on 0.8509. This closes that hole before any more diagnostics are built on
# top of it: if the target's own sampling error is anywhere near 2.53%, the gap is not a finding.
#
# WHY CLUSTERED, NOT A PLAIN sem. The rows nest -- pair < (case,input_index) noise realisation <
# input_index intrinsic galaxy < case render -- and only the pair level is collapsed by the
# all-pairs 1/n_pairs weighting. The script reports a cluster-robust sem at EVERY level and a
# case-level bootstrap; read the largest. Quoting the row-level number would be the standard way to
# under-report this by the square root of the cluster size.
#
# CPU ONLY, NO GPU. It touches no checkpoint and no flow -- this is a property of the TARGET DATA,
# so it is the same number for every seed and does not follow the 16-seed e-response convention.
# That convention governs quantities that depend on which flow you trained; this one does not.
#
# FIREWALL: read-only. Fits nothing, tunes nothing, writes no target. The `--demanded 0.8724`
# constgold value enters the PRINTOUT of the final gap line only -- never a fit -- and is labelled
# there as carrying no error of its own.
#
# ARGS MUST MIRROR jobs/job_resp_target_v21_grids.sh, which built the target being tested: same
# val catalogue, same snc lookup, same --max-case 99, same V2.1 domain, same nominal g. They are not
# re-typed on trust: the script recomputes global_R and REFUSES unless it reproduces the value
# stored in the npz to 1e-9, so a mismatched argument aborts instead of quietly reporting the error
# bar of a different quantity.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results

echo "### BOOTSTRAP RESPONSE TARGET job=$SLURM_JOB_ID ###"; date
python -u scripts/bootstrap_response_target.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --npz $R/response_target_crowd_rblend_snc_c0-99_5x4x5_v21.npz \
  --snc-lookup $R/g0_lookup_c0-99.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend --max-case 99 \
  --n-boot 4000 --demanded 0.8724 \
  2>&1 | grep -v --line-buffered "module command" || { echo BS_RESP_FAILED; exit 1; }
echo BS_RESP_ALL_DONE; date
