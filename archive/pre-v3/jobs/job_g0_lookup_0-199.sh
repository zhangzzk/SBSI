#!/bin/bash
#SBATCH --job-name=g0lk199
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/g0lk199_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# OPTIONAL / 200-case extension only (WORKLOG cont.17). NOT needed for the primary g=0.02 test:
# preflight found the g=0.02 render only covers cases 0-99, so the primary experiment uses the
# EXISTING g0_lookup_c0-99.feather. This job (extend the SNC g=0 shape lookup to 0-199) is only useful
# if you FIRST generate det+meas on the g=0.02 render for cases 100-199 and want a 200-case target.
OUT=results/g0_lookup_c0-199.feather
echo "### build g0 SNC lookup, cases 0-199 -> $OUT ###"; date
python -u scripts/build_g0_lookup.py --cases $(seq 0 199) --output $OUT 2>&1 | grep -v "module command"
date; echo "G0_LOOKUP_199_DONE $OUT"
