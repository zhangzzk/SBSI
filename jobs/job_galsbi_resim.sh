#!/bin/bash
#SBATCH --job-name=galsbi-resim
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --chdir=/home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/logs/resim_%A_%a.out
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
# GalSBI re-simulation (SBSI scenes, LSST-i, centred 65 px stamps). Chained submissions, small jobs schedule soonest:
#   STEPS=1 CASE0=c NCASES=n                                 catalogues + noise.csv, once
#   STEPS=2-3 CASE0=c NCASES=1 LEGS="0.02 -0.02" --array=0-(2n-1)%500
#                                                            render + measure; task i = case c+i/2, leg LEGS[i%2]
#   STEPS=2-3 CASE0=c NCASES=1 --array=0-(n-1)               (pilot) one case per task, both legs in turn
#   STEPS=4 CASE0=c NCASES=n                                 paired catalogues
# Training (half mode): STEPS=2,3all LEGS="0.0 0.05 0.2" --array=0-(3n-1) renders a leg, then measures primaries and
#   secondaries on it; values with commas go through env (env STEPS=2,3all ... sbatch --export=ALL ...), because
#   --export splits on commas. Then STEPS=4 and STEPS=4b once.
# CONFIG = BlendEMU yaml (the pilot's is doc/evidence/galsbi_resim_pilot_20261101/galsbi_lsst_i_pilot.yaml).
# BlendEMU code: branch galsbi-resim. Renderer: MultiBand_ImSim branch galsbi-resim (local worktree) = the FS2
# renderer plus a retry at a 32768 FFT limit for galaxies GalSim refuses, and a hard failure when an image worker
# crashes.
set -eo pipefail
: "${CONFIG:?}" "${STEPS:?}" "${CASE0:?}" "${NCASES:?}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
LEG_ARGS=()
if [[ -n "${LEGS:-}" ]]; then
  read -ra LEG <<< "$LEGS"
  CASE=$((CASE0 + TASK / ${#LEG[@]}))
  LEG_ARGS=("--shear-label=${LEG[TASK % ${#LEG[@]}]}")
else
  CASE=$((CASE0 + TASK))
fi
echo "START GalSBI resim config=$CONFIG steps=$STEPS cases=$CASE..$((CASE + NCASES - 1)) ${LEG_ARGS[*]}"; date

source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then source /usr/share/Modules/init/bash; fi
module load sextractor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu_galsbi_resim"
export BLENDEMU_SIM_RUN="/home/z/Zekang.Zhang/MultiBand_ImSim_galsbi_resim/modules/Run.py"

python -c "import blendemu, sys; print('blendemu from', blendemu.__file__); assert blendemu.__file__.startswith('/home/z/Zekang.Zhang/blendemu_galsbi_resim/')"
git -C /home/z/Zekang.Zhang/blendemu_galsbi_resim log --oneline -1
git -C /home/z/Zekang.Zhang/MultiBand_ImSim_galsbi_resim log --oneline -1
python run_pipeline.py --config "$CONFIG" --steps "$STEPS" --case-offset "$CASE" --n-cases "$NCASES" --n-mpi 1 \
  "${LEG_ARGS[@]}"

echo "FINISH GalSBI resim"; date
