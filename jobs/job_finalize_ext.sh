#!/bin/bash
#SBATCH --job-name=finalize_ext
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/finalize_ext_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues

echo "### concat lookups (0-39 + 40-199) ###"
python -u - <<'PY' 2>&1 | grep -v module
import glob, pandas as pd, pyarrow.feather as pf
# shell-flux
fx = [pf.read_table("results/crowd_flux_c0-39.feather").to_pandas()] + \
     [pf.read_table(p).to_pandas() for p in sorted(glob.glob("results/crowd_flux_ext_part*.feather"))]
fxa = pd.concat(fx, ignore_index=True).drop_duplicates(["case","input_index"])
fxa.to_feather("results/crowd_flux_c0-199.feather")
print(f"crowd_flux c0-199: {len(fxa):,} rows, {fxa['case'].nunique()} cases")
# r_blend
bl = [pf.read_table("results/blend_lookup_hs_c0-39.feather").to_pandas()] + \
     [pf.read_table(p).to_pandas() for p in sorted(glob.glob("results/blend_ext_part*.feather"))]
bla = pd.concat(bl, ignore_index=True).drop_duplicates(["case","input_index"])
bla.to_feather("results/blend_lookup_c0-199.feather")
print(f"blend c0-199: {len(bla):,} rows, {bla['case'].nunique()} cases, R_blend mean={bla['R_blend'].mean():.4f}")
PY

FL=results/crowd_flux_c0-199.feather; BL=results/blend_lookup_c0-199.feather
echo "### augment VAL g0.05 (all 200 cases) ###"
python -u scripts/augment_crowding.py --catalogue $D/det_meas_ngmix_np7_g0.05_val.feather \
  --flux-lookup $FL --blend-lookup $BL --output $D/det_meas_crowd_g0.05_val_full.feather 2>&1 | grep -v module
echo "### augment TEST g0.02 (all 100 cases) ###"
python -u scripts/augment_crowding.py --catalogue $D/det_meas_ngmix_np7_g0.02_test.feather \
  --flux-lookup $FL --blend-lookup $BL --output $D/det_meas_crowd_g0.02_test_full.feather 2>&1 | grep -v module
echo FINALIZE_EXT_DONE
