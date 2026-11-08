#!/usr/bin/env bash
#SBATCH --job-name=sbsi_geometry_coordinates
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
scope=${1:?smoke or full}
protocol=${2:?frozen protocol required}
output=${3:?new output path required}
args=()
case "$scope" in
 smoke) args+=(--smoke) ;;
 full) test -f /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1/coordinate_smoke.json ;;
 *) exit 2 ;;
esac
"$python" -m pytest tests/test_scene_geometry_coordinates.py -q
exec "$python" -u - "$protocol" "$output" "${args[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path = Path('/home/z/Zekang.Zhang/SBSI/scripts/audit_geometry_coordinate_response.py')
source = path.read_bytes()
protocol, output, *flags = sys.argv[1:]
expected = json.loads(Path(protocol).read_text())['files'][str(path)]
if hashlib.sha256(source).hexdigest() != expected:
    raise RuntimeError('executed source does not match frozen protocol')
sys.argv = [str(path), '--protocol', protocol, '--output', output, *flags]
print('GEOMETRY_EXECUTED_SOURCE', expected, flush=True)
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
PY
