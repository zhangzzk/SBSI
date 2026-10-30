#!/bin/bash
#SBATCH -p cluster
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -J curvature
#SBATCH -o /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/curvature_%j.log
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/curvature.py /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/curvature_readings.npz 24
