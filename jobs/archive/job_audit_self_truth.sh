#!/bin/bash
#SBATCH --job-name=audit_self
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/audit_self_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# FINAL fork: is the residual q3 deficit flow-underfit (fixable) or anisotropy/cross-term (ceiling)?
# Compares the TRUE self-response R_self_truth (delta_et1_self/gamma, neighbour present) to model R_flow,
# per R_blend quantile, on gold 0-39. q3 R_self_truth >> R_flow => flow over-suppresses (branch A, fixable);
# R_self_truth ~ R_flow => flow fine, deficit is emulator/anisotropy. (Emulator already exonerated by
# audit_blend_truth, so R_self_truth~R_flow would point to anisotropy/cross-term.)
python -u scripts/audit_self_truth.py 2>&1 | grep -vE "module command"
echo AUDIT_SELF_JOB_DONE
