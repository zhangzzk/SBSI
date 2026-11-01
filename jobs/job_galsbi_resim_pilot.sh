#!/bin/bash
#SBATCH --job-name=galsbi-pilot
#SBATCH --time=20:00:00
#SBATCH --mem=150G
#SBATCH --ntasks-per-node=10
#SBATCH --nodes=1
#SBATCH --chdir=/home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/logs/pilot_%j.out
#SBATCH --partition=cluster
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
# GalSBI re-simulation pilot: 10 SBSI scenes (cases 9000-9009), LSST-i, +-0.02, centred 65 px stamps.
# BlendEMU code: branch galsbi-resim. Renderer: MultiBand_ImSim as used for the FS2 sims.
set -eo pipefail
echo "START GalSBI pilot"; date

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
python run_pipeline.py --config "$CONFIG" --steps "${PILOT_STEPS:-all}" --n-mpi 10 --n-cases 10

echo "FINISH GalSBI pilot"; date
