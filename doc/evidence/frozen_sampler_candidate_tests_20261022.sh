#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=frozen-sampler-tests
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPYCACHEPREFIX=/tmp/frozen-sampler-tests-${SLURM_JOB_ID}
unset PYTHONPATH
cd /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/frozen_sampler_candidate_source_20261022
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u - <<'PY'
from pathlib import Path
import hashlib,json,sys,os,time
source=Path.cwd()
output=Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/frozen_sampler_candidate_validation_20261022')
manifest=source/'SHA256SUMS'
expected='130e9d30fd8ee592c7d934a95202608293a1be4b489e851981d5f8c11e85b26d'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected
entries={name:checksum for checksum,name in (line.split('  ',1) for line in manifest.read_text().splitlines())}
def check():
    assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==h for name,h in entries.items())
check();print('SOURCE_BEFORE_PASS',len(entries),flush=True)
sys.path.insert(0,str(source))
import sbsi,torch
assert Path(sbsi.__file__).resolve().parent==source/'sbsi'
torch.set_num_threads(2)
import pytest
started=time.monotonic(); code=None
try:
    code=int(pytest.main(['tests','-q','-p','no:cacheprovider','--junitxml='+str(output/'junit.xml')]))
finally:
    check();print('SOURCE_AFTER_PASS',len(entries),flush=True)
    (output/'validation.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],source=str(source),manifest_sha256=expected,files=len(entries),source_before_after_pass=True,pytest_exit_code=code,elapsed_seconds=time.monotonic()-started,python=sys.version,torch=str(torch.__version__),cpus=2,gpus=0),indent=2)+'\n')
raise SystemExit(code)
PY
