#!/usr/bin/env bash
# Example Slurm wrapper for BlendEMU image simulation and measurement.
#
# Submit from the SBSI repository with explicit paths, for example:
#   sbatch --export=ALL,BLENDEMU_CONFIG=/path/to/config.yaml,BLENDEMU_SIM_RUN=/path/to/Run.py \
#       examples/job_blendemu.sh
#
# The YAML owns the input population catalogue and every output path.  See
# blendemu/configs/fs2_lsst_r.example.yaml for the complete schema.

#SBATCH --job-name=sbsi-catalogues
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=50
#SBATCH --mem=250G
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=sbsi_catalogues_%j.out
#SBATCH --error=sbsi_catalogues_%j.err

set -euo pipefail

: "${BLENDEMU_CONFIG:?set BLENDEMU_CONFIG to a BlendEMU pipeline YAML}"
: "${BLENDEMU_SIM_RUN:?set BLENDEMU_SIM_RUN to MultiBand_ImSim modules/Run.py}"

BLENDEMU_ROOT="${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBSI_ROOT="${SBSI_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
N_MPI="${N_MPI:-50}"
N_JOBS="${N_JOBS:-16}"

eval "$(conda shell.bash hook)"
conda activate sims1
module load sextractor

export PYTHONPATH="${SBSI_ROOT}:${BLENDEMU_ROOT}:${PYTHONPATH:-}"
export BLENDEMU_SIM_RUN

# BlendEMU owns all four operations:
#   1  generate realization catalogues and simulator configs
#   2  render images
#   3/3b  measure primary and secondary shapes
#   4/4b  assemble response, detection, and self-response catalogues
# run_pipeline.py dispatches its own srun calls for the MPI steps, so do not
# wrap this command in another srun.
python -u "${BLENDEMU_ROOT}/scripts/run_pipeline.py" \
    --config "${BLENDEMU_CONFIG}" \
    --steps 1,2,3,3b,4,4b \
    --n-mpi "${N_MPI}" \
    --n-jobs "${N_JOBS}"
