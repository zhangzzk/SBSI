#!/usr/bin/env bash
#SBATCH --job-name=sbsi_emulator_repeat_check
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
squeue -u "$USER" -o '%.18i %.30j %.8T %.10M %.25R'
sacct -j 16592759,16592811,16592825,16592853,16592812,16592854 --format=JobID,State,Elapsed,ExitCode
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python - <<'PY'
from pathlib import Path
root = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
logs = root/'flow_shape_augmentation_20260918_v1/logs'
for job, prefix in ((16592759, 'blendemu_disk_moment_repeat'),
                    (16592811, 'sbsi_freeze_emulator_repeat'),
                    (16592825, 'sbsi_emulator_repeat_joint'),
                    (16592853, 'sbsi_emulator_repeat_density')):
    for path in sorted(logs.glob(f'{prefix}_{job}*')):
        print('LOG', path)
        print('\n'.join(path.read_text().splitlines()[-8:]))
for path in sorted((root/'scene_classifier_geometry_20260918_v1/emulator_repeat_results').glob('*.json')):
    print('RESULT', path.name, path.stat().st_size)
PY
