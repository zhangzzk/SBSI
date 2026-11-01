#!/bin/bash
#SBATCH --job-name=galsbi-pilot
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --chdir=/home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/logs/pilot_%A_%a.out
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
# GalSBI re-simulation pilot: 10 SBSI scenes (cases 9000-9009), LSST-i, +-0.02, centred 65 px stamps.
# Three chained submissions (small jobs schedule far sooner than one 10-core node):
#   STEPS=1   CASE0=9000 NCASES=10                      catalogues + noise.csv, once
#   STEPS=2,3 CASE0=9000 NCASES=1  --array=0-9           render + measure, one case per task
#   STEPS=4   CASE0=9000 NCASES=10                      paired catalogues
# BlendEMU code: branch galsbi-resim. Renderer: MultiBand_ImSim as used for the FS2 sims.
set -eo pipefail
: "${STEPS:?}" "${CASE0:?}" "${NCASES:?}"
CASE=$((CASE0 + ${SLURM_ARRAY_TASK_ID:-0}))
echo "START GalSBI pilot steps=$STEPS cases=$CASE..$((CASE + NCASES - 1))"; date

source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then source /usr/share/Modules/init/bash; fi
module load sextractor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu_galsbi_resim"
export BLENDEMU_SIM_RUN="/home/z/Zekang.Zhang/MultiBand_ImSim/modules/Run.py"
CONFIG=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe/doc/evidence/galsbi_resim_pilot_20261101/galsbi_lsst_i_pilot.yaml

python -c "import blendemu, sys; print('blendemu from', blendemu.__file__); assert blendemu.__file__.startswith('/home/z/Zekang.Zhang/blendemu_galsbi_resim/')"
git -C /home/z/Zekang.Zhang/blendemu_galsbi_resim log --oneline -1
python run_pipeline.py --config "$CONFIG" --steps "$STEPS" --case-offset "$CASE" --n-cases "$NCASES" --n-mpi 1

echo "FINISH GalSBI pilot"; date
