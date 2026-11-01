#!/bin/bash
#SBATCH -p cluster
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -J fs2curv
#SBATCH -o /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/fs2/curvature_mock_%j.log
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/fs2/curvature_mock.py /project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2truth_mock_curvature_20261101/curvature_readings.npz 24
