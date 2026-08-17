#!/bin/bash
#SBATCH --job-name=ra_smoke
#SBATCH --time=00:50:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ra_smoke_%j.out

# END-TO-END SMOKE TEST for the realisation-aware (RA) build. Tiny, real data, no synthetic numbers.
#   step 1  build_g0_lookup.py --extra-cols on THREE cases  -> the extended g0 lookup exists
#   step 2  compute_response_target_measbin.py on the same three cases -> a real rho target npz
#   step 3  train_measurement_model_swa_s1_truecond.py for 2 epochs on ~40k rows with
#             --flow-type mean_affine_ra --ra-hidden 32 --ra-weight 1.0 --ra-warm-start <fiducial>
#           exercising the full fiducial loss stack (response pin + coupling pin) plus the RA term
#   step 4  reload the checkpoint and assert the RA head actually makes the response vary per draw
#   step 5  rerun step 3 with --ra-hidden 0 --ra-weight 0 to prove the demoted path still runs
#
# This proves the code path executes; it certifies NOTHING about m and quotes no result.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

R=/home/z/Zekang.Zhang/SBSI/results
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
S=${SMOKE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/ra/smoke}
A=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
rm -rf "$S"; mkdir -p "$S"

echo "### RA SMOKE job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

echo; echo "=== step 1: extended g0 lookup (cases 0-2) ==="
python -u scripts/build_g0_lookup.py --cases 0 1 2 --output "$S/g0_lookup_meas_c0-2.feather" \
  --extra-cols MAG_AUTO:measured_mag_auto_g0 FLUX_RADIUS:measured_flux_radius_g0 \
  || { echo SMOKE_STEP1_FAILED; exit 1; }

echo; echo "=== step 2: rho target on cases 0-2 (coarse, low min-count: SMOKE ONLY) ==="
python -u scripts/compute_response_target_measbin.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --fine-target-npz $R/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz \
  --snc-lookup "$S/g0_lookup_meas_c0-2.feather" \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 --nominal-g 0.05 \
  --merge-flux 3 --merge-size 3 --merge-crowd 5 --n-dmag 3 --n-dlogsize 2 \
  --min-count 50 --max-case 2 --primary-mag-max 26.0 --primary-re-min 0.3 \
  --output "$S/ra_target_smoke.npz" || { echo SMOKE_STEP2_FAILED; exit 1; }

TRAIN () {   # $1=outfile $2=ra_hidden $3=ra_weight
python -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue $D/det_meas_crowd_conc_g0.0_train_full.feather --output "$1" \
  --target-column detected --selection-name sextractor_detected --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine_ra --ra-hidden "$2" --ra-weight "$3" \
  --ra-target-npz "$S/ra_target_smoke.npz" \
  --ra-warm-start $A/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt \
  --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 40000 --max-read-batches 8 --epochs 2 --batch-size 4096 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 5 --weight-decay 1e-5 \
  --seed 501 --num-workers 4 --gpu-resident --swa-last-k 2 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --response-weight 450 --response-delta 0.02 --response-difference central \
  --response-target-npz $R/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz \
  --coupling-weight 500 --coupling-target-npz $R/response_target_theta_coupling_rblend_c0-99_6x9x5.npz
}

echo; echo "=== step 3: 2-epoch RA training (head ON) ==="
TRAIN "$S/flow_ra_smoke.pt" 32 1.0 || { echo SMOKE_STEP3_FAILED; exit 1; }

echo; echo "=== step 4: reload + per-draw response variation ==="
python - "$S/flow_ra_smoke.pt" <<'PY' || { echo SMOKE_STEP4_FAILED; exit 1; }
import sys, numpy as np, torch
from sbs_shear.measurement_model import load_measurement_model, ConditionalMeanFlowRA
b = load_measurement_model(sys.argv[1], device="cpu")
m = b.model
assert isinstance(m, ConditionalMeanFlowRA), type(m)
print("flow_type reloaded as", type(m).__name__, " ra_hidden =", m.ra_hidden,
      " ra_indices =", m.ra_indices.tolist(), " ra_targets =", m.ra_targets.tolist())
print("||A out weight|| =", float(m.ra_net[-1].weight.norm()))
c0 = torch.randn(8, m.context_dim)
c1 = c0.clone(); c1[:, 0] += 0.02
torch.manual_seed(7); s0 = m.sample(c0, n_samples=64)
torch.manual_seed(7); s1 = m.sample(c1, n_samples=64)
d = (s1 - s0)[:, :, 0]
print("per-draw response std within an object: min %.3e  max %.3e" % (float(d.std(1).min()), float(d.std(1).max())))
print("per-draw response mean spread across objects: %.3e" % float(d.mean(1).std()))
assert float(d.std(1).min()) > 0.0, "RA head trained but response is still identical across draws"
# unit-Jacobian consistency on real reloaded weights
x = m.sample(c0, n_samples=1)[:, 0, :]
resid = x - m._mu(c0)
z = resid - m._A(c0, resid.index_select(-1, m.ra_indices))
back = z + m._A(c0, z.index_select(-1, m.ra_indices)) + m._mu(c0)
print("closed-form inverse max|err| =", float((back - x).abs().max()))
assert float((back - x).abs().max()) < 1e-4
print("STEP4_OK")
PY

echo; echo "=== step 5: --ra-hidden 0 demotion still runs (diagnostic-only RA readout) ==="
TRAIN "$S/flow_ra0_smoke.pt" 0 0 || { echo SMOKE_STEP5_FAILED; exit 1; }
python - "$S/flow_ra0_smoke.pt" <<'PY' || { echo SMOKE_STEP5B_FAILED; exit 1; }
import sys, torch
from sbs_shear.measurement_model import load_measurement_model, ConditionalMeanFlow, ConditionalMeanFlowRA
b = load_measurement_model(sys.argv[1], device="cpu")
assert isinstance(b.model, ConditionalMeanFlow) and not isinstance(b.model, ConditionalMeanFlowRA)
print("demoted checkpoint reloads as", type(b.model).__name__, "-- fiducial architecture intact")
PY

echo RA_SMOKE_DONE; date
