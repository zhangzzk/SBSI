#!/bin/bash
#SBATCH --job-name=audit_blend
#SBATCH --time=03:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/audit_blend_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# GROUNDING: is the in-training/gold q3 deficit an EMULATOR issue or a FLOW issue?
# Compares the ho-emulator's per-pair R_blend prediction to the sim-measured truth (delta_et1/gamma)
# on gold cases 0-39, WITHIN the main sim (same pairs -> no cross-sim confound for emulator accuracy).
# (B) truth_sum - <R_bl_emu> per production R_blend quantile: q3 ~ +0.047 => emulator under-predicts
# (recalibratable emulator issue); ~0 => emulator fine, q3 deficit is R_flow (flow) or anisotropy.
python -u scripts/audit_blend_truth.py 2>&1 | grep -vE "module command"
echo AUDIT_BLEND_JOB_DONE
