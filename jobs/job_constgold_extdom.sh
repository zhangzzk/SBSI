#!/bin/bash
#SBATCH --job-name=cg_extdom
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_extdom_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
python -u - <<'PY' 2>&1 | grep -v module
import glob, pandas as pd, pyarrow.feather as pf
parts=sorted(glob.glob("results/blend_extdom_part*.feather"))
out=pd.concat([pf.read_table(p).to_pandas() for p in parts],ignore_index=True).drop_duplicates(["case","input_index"])
out.to_feather("results/blend_lookup_const_extdom_c0-39.feather")
print(f"extdom lookup: {len(out):,} rows, {out['case'].nunique()} cases, R_blend mean={out['R_blend'].mean():.4f}")
PY
# crowd_flux flow (feature = nbr_flux, INDEPENDENT of the emulator change) + extdom additive R_blend
echo "### CONSTANT-GOLD crowd_flux + EXTDOM emulator (all 40 + error) ###"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --blend-lookup results/blend_lookup_const_extdom_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 2>&1 | grep -v module
echo CG_EXTDOM_DONE
