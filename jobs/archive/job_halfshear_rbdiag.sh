#!/bin/bash
#SBATCH --job-name=hs_rbdiag
#SBATCH --time=04:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_rbdiag_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

python -u - <<'PY' 2>&1 | grep -v module
import glob, pandas as pd, pyarrow.feather as pf
parts = sorted(glob.glob("results/blend_lookup_hs_part*.feather"))
out = pd.concat([pf.read_table(p).to_pandas() for p in parts], ignore_index=True).drop_duplicates(["case","input_index"])
out.to_feather("results/blend_lookup_hs_c0-39.feather")
print(f"concat {len(parts)} parts -> {len(out):,} rows, {out['case'].nunique()} cases, R_blend mean={out['R_blend'].mean():.4f}")
PY

echo "### HALF-SHEAR g=0.05, NO SNC, binned by emulator R_blend (cases 0-39) ###"
python -u scripts/validate_allpairs_response.py \
  --measurement-model models/measurement_flow_g0_ngmix_np7_respblend_lam300_v1.pt \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g0.05_val.feather \
  --nominal-g 0.05 --max-rows 0 --max-case 39 --n-samples 64 \
  --blend-lookup results/blend_lookup_hs_c0-39.feather 2>&1 \
  | grep -iE "BLEND-BIN|GLOBAL|Rblend-bin|ISO\(|\[0|>=|INCOHERENT binned|N_eff|BLENDED-ONLY" | grep -v module
echo HS_RBDIAG_DONE
