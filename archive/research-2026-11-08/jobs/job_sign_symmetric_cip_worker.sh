#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --constraint=x86-64-v3
#SBATCH --job-name=signsym500k-cip
set -euo pipefail
H=@STUDY_ROOT@
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PY" "$H/verify.py"
# Test the same frozen sampler on this GPU before claiming production units.
# The original worker will use six single-core slots in this eight-CPU allocation.
SMOKE="$H/cip_smoke_${SLURM_JOB_ID}"
mkdir "$SMOKE"
ln -s "$H/source_snapshot" "$SMOKE/source_snapshot"
readarray -t smoke_ids < <(head -n 2 "$H/additional_faint_ids.txt")
SBSI_TEMPER_ROOT="$SMOKE" bash "$H/source_snapshot/jobs/job_galsbi_tempered.sh" faint \
  --reference "$H/reference" --shear 0 0 --seed 2028500001 --initial-bank-seed 2033500001 \
  --observation-indices "${smoke_ids[@]}" --steps 64 --warmup 32 --profile-stride 16
"$PY" - "$H" "$SMOKE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,smoke=map(Path,sys.argv[1:])
protocol=json.loads((smoke/'faint/protocol.json').read_text())
result=json.loads((smoke/'faint/result.json').read_text())
identity=json.loads((root/'rule.json').read_text())['model_identity']
assert protocol['model_identity']==identity
assert len(result['observations'])==2
receipt=dict(status='integration_passed', model_identity=identity,
             sampler_sha256=hashlib.sha256((root/'worker.sh').read_bytes()).hexdigest(),
             limitation='Short faint-chain code smoke on 16GB A40; not convergence evidence')
(smoke/'passed.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('CIP_INTEGRATION_PASSED',flush=True)
PY
exec bash "$H/worker.sh" "$@"
