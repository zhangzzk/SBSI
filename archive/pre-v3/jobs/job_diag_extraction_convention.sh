#!/bin/bash
#SBATCH --job-name=extconv
#SBATCH --time=01:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/extconv_%j.out
# The revised decisive test for the open question "do half-shear and constgold agree on total
# response?" (WORKLOG 2026-08-05n). Rather than comparing two SIMS across two extraction conventions
# -- which CONVENTIONS.md 6c says manufactures a spurious gap -- this measures BOTH conventions on
# ONE sim, identical rows, identical estimator, split by resolution. Estimator match confirmed by
# job 15540936 (see WORKLOG 2026-08-05p).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -u scripts/diag_extraction_convention.py --min-case 40 --max-case 99
