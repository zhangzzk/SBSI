#!/usr/bin/env bash
# Frozen two-bin intrinsic-|e| extension to the photometric response target.
#SBATCH --job-name=sbsi_eabs_targets
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/eabs2_full_v1/logs/targets_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/eabs2_full_v1/logs/targets_%j.err

set -euo pipefail
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
root=$base/eabs2_full_v1
train=$root/coupling_train_eabs2.npz
validation=$root/coupling_validation_eabs2.npz
[[ ! -e "$train" && ! -e "$validation" && ! -e "$root/target_audit.json" ]]
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
common=(--g0-catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
        --sheared-catalogue "$cache/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather"
        --response-target "$base/response_target_mixed_train160_6x6x5_matrix_re037.npz"
        --n-e-bins 2 --strict-counts)
"$python" -u -m scripts.build_flow_coupling_target "${common[@]}" \
  --case-role train --min-count 300 --output "$train"
"$python" -u -m scripts.build_flow_coupling_target "${common[@]}" \
  --case-role validation --min-count 100 --e-edges-from "$train" --output "$validation"
"$python" - "$root" <<'PY'
from pathlib import Path
import json
import sys
import numpy as np
root=Path(sys.argv[1])
report={}
edges=None
for role in ('train','validation'):
    with np.load(root/f'coupling_{role}_eabs2.npz') as f:
        assert not f['fallback'].any()
        assert f['coupling_size'].shape==(6,6,5,2)
        if edges is None: edges=f['edges_e_abs'].copy()
        else: np.testing.assert_array_equal(f['edges_e_abs'],edges)
        data={'rows':int(f['counts'].sum()),'cases':len(f['cases']),
              'count_min':int(f['counts'].min()), 'count_median':float(np.median(f['counts'])),
              'case_coverage_min':int(f['case_coverage'].min()),'edges_e_abs':f['edges_e_abs'].tolist()}
        for name in ('mag','size'):
            for statistic in ('case_jackknife_se','case_half_difference'):
                values=np.abs(f[name+'_'+statistic])
                assert np.isfinite(values).all()
                data[name+'_'+statistic+'_median_p95_max']=np.quantile(values,[.5,.95,1]).tolist()
        report[role]=data
report['status']='pass'
(root/'target_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
PY
