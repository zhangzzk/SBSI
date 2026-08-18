#!/bin/bash
#SBATCH --job-name=readme_fig
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/readme_fig_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/readme_fig_%j.err
#
# Rebuild the README front figure: three real constgold galaxies (bright, typical,
# faint) as noisy image stamps, with the release flow's conditional measurement
# distribution for the same truth rows underneath, plus arrows for the blending
# displacement R_blend * gamma from the emulator.
#
#   sbatch jobs/job_constgold_readme_figure.sh
#
# The PNG is written next to the script by default; publish it by copying it into
# the master checkout as examples/measurement_flow_contours.png.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate py31
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
# BlendEMU is a source checkout, not a pip package; the emulator supplies R_blend.
export BLENDEMU_ROOT=${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}
export PYTHONPATH="$REPO:$BLENDEMU_ROOT:$PYTHONPATH"

date
python -u examples/make_constgold_readme_figure.py \
  --output "${OUTPUT:-$REPO/examples/measurement_flow_contours.png}" \
  ${EXTRA:-} 2>&1 | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
