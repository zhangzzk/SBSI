#!/usr/bin/env bash
# CPU smoke by default. Full runs require an explicit one-GPU Slurm override.
#SBATCH --job-name=sbsi_shape_augment
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
domain=$cache/harmonized_secondary_20260917_v1
root=$cache/flow_shape_augmentation_20260918_v1
python=$data/envs/py31/bin/python
arm=${1:?control or augmented}
mode=${2:?smoke or full}
case $arm in control|augmented) ;; *) exit 2 ;; esac
case $mode in
  smoke) options=(--smoke --device cpu); output=$root/smoke_$arm ;;
  full) options=(--device cuda); output=$root/$arm ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$domain" "$root" "$arm" "$mode" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

domain, root = map(Path, sys.argv[1:3])
arm, mode = sys.argv[3:5]
flow = domain/'refinement/selected.pt'
if hashlib.sha256(flow.read_bytes()).hexdigest() != 'fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4':
    raise RuntimeError('corrected initial flow changed')
if mode == 'full':
    smoke = json.loads((root/f'smoke_{arm}/completed.json').read_text())
    protocol = json.loads((root/f'smoke_{arm}/protocol.json').read_text())
    if (smoke['arm'] != arm or smoke['smoke'] is not True
            or protocol['initial_flow_sha256'] != hashlib.sha256(flow.read_bytes()).hexdigest()
            or protocol['script_sha256'] != hashlib.sha256(Path('scripts/refine_flow_shape_augmentation.py').read_bytes()).hexdigest()):
        raise RuntimeError('matching successful CPU smoke required')
PY
exec "$python" -u -m scripts.refine_flow_shape_augmentation \
  --domain-root "$domain/domain" --flow "$domain/refinement/selected.pt" \
  --output-root "$output" --arm "$arm" "${options[@]}"
