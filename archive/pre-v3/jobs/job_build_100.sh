#!/bin/bash
#SBATCH --job-name=build100
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/build100_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
CB=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
date
echo "### blend_lookup 80-139 ###"
python -u scripts/build_blend_lookup.py --cases $(seq 80 139) --tag lsst_r_extnbr_ho \
  --output results/blend_lookup_extnbrho_c80-139.feather 2>&1 | grep -v "module command"
echo "### ood_split 80-139 ###"
python -u scripts/build_ood_split_lookup.py --cases $(seq 80 139) \
  --output results/ood_split_c80-139.feather 2>&1 | grep -v "module command"
echo "### concat catalogues + lookups -> 40-139 (100 cases) ###"
python -u - <<'PY'
import pyarrow.feather as pf, pandas as pd
CB="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
# catalogue: c40-79 + c80-139 -> c40-139
a=pf.read_table(f"{CB}/constant_response_catalogue_c40-79.feather").to_pandas()
b=pf.read_table(f"{CB}/constant_response_catalogue_c80-139.feather").to_pandas()
pd.concat([a,b],ignore_index=True).to_feather(f"{CB}/constant_response_catalogue_c40-139.feather")
print(f"catalogue 40-139: {len(a)+len(b):,} rows ({a.case.nunique()}+{b.case.nunique()} cases)")
# lookups
for name in ["blend_lookup_extnbrho","ood_split"]:
    x=pf.read_table(f"results/{name}_c40-79.feather").to_pandas()
    y=pf.read_table(f"results/{name}_c80-139.feather").to_pandas()
    pd.concat([x,y],ignore_index=True).to_feather(f"results/{name}_c40-139.feather")
    print(f"{name} 40-139: {len(x)+len(y):,} rows")
PY
date; echo BUILD100_DONE
