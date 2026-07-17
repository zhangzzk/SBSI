#!/bin/bash
#SBATCH --job-name=crowd_conc_prep
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/crowd_conc_prep_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
FL=results/crowd_flux_conc_c0-199.feather          # near/far/MAX concentration lookup (cases 0-199)
BL=results/blend_lookup_c0-199.feather             # same blend lookup used for the current train cat
CAT_IN=$D/det_meas_ngmix_np7_g0.0_train.feather
CAT_OUT=$D/det_meas_crowd_conc_g0.0_train_full.feather

echo "### STEP 1: build crowding lookup WITH nbr_flux_max (cases 0-199, main sim) ###"; date
python -u scripts/build_crowding_lookup.py --cases $(seq 0 199) \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 --sign 0.0 \
  --output $FL 2>&1 | grep -v module || { echo "BUILD_LOOKUP FAILED"; exit 1; }
date

echo "### STEP 2: augment TRAIN catalogue with near/far/max (+ r_blend) ###"; date
python -u scripts/augment_crowding.py \
  --catalogue $CAT_IN \
  --flux-lookup $FL \
  --blend-lookup $BL \
  --output $CAT_OUT 2>&1 | grep -v module || { echo "AUGMENT FAILED"; exit 1; }
date
echo "### verify nbr_flux_max present in train catalogue ###"
python -c "
import pyarrow.feather as pf
s=pf.read_table('$CAT_OUT', memory_map=True).schema.names
assert 'nbr_flux_max' in s, 'nbr_flux_max MISSING'
print('OK train cat has:', [c for c in s if 'nbr_flux' in c or c=='r_blend'])
"
echo "CROWD_CONC_PREP_DONE"
