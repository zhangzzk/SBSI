#!/bin/bash
#SBATCH --job-name=cg_np7_blend
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_np7_blend_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# 1) concatenate the per-chunk blend-lookup parts into one c0-39 feather
python -u - <<'PY' 2>&1 | grep -v module
import glob, pandas as pd, pyarrow.feather as pf
parts = sorted(glob.glob("results/blend_lookup_const_part*.feather"))
dfs = [pf.read_table(p).to_pandas() for p in parts]
out = pd.concat(dfs, ignore_index=True)
out = out.drop_duplicates(["case","input_index"])
out.to_feather("results/blend_lookup_const_c0-39.feather")
print(f"concatenated {len(parts)} parts -> {len(out):,} rows, {out['case'].nunique()} cases, "
      f"R_blend mean={out['R_blend'].mean():.4f}")
PY

# 2) constant-gold validation with NP7 flow + full per-(case,input_index) blend lookup
echo "### CONSTANT-GOLD  NP7 flow + per-case blend lookup ###"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_np7_respblend_lam300_v1.pt \
  --blend-lookup results/blend_lookup_const_c0-39.feather \
  --max-rows 6000000 --n-samples 64 2>&1 | grep -v module
echo CG_NP7_BLEND_DONE
