#!/bin/bash
#SBATCH --job-name=derivfit2
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/derivfit2_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
date; python -u scripts/derived_fit2.py || { echo ANALYTIC_FAILED; exit 1; }; date
