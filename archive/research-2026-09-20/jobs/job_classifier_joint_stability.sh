#!/usr/bin/env bash
#SBATCH --job-name=sbsi_classifier_influence
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
mode=${1:?early or full}
case "$mode" in
 early) endpoints=(fresh_g1 fresh_g2); suffix=_cuda_early ;;
 full) endpoints=(fresh_g1 fresh_g2 main independent); suffix=_cuda ;;
 *) exit 2 ;;
esac
"$python" -m pytest tests/test_classifier_joint_stability.py -q
arguments=()
for endpoint in "${endpoints[@]}"; do
 arguments+=(--result "$root/transported_joint_${endpoint}${suffix}.json")
 arguments+=(--receipt "$root/transported_joint_${endpoint}_comparison${suffix}.json")
done
exec "$python" -u -m scripts.summarize_classifier_joint_stability "${arguments[@]}" \
 --output "$root/transported_${mode}_influence.json"
