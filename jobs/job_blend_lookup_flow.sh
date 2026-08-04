#!/bin/bash
#SBATCH --job-name=bllookup
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bllookup_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bllookup_%j.err

# Per-(case, input_index) R_blend from FLOW #2 over constgold cases 40-139 -- the same case range as
# the fiducial `results/blend_lookup_indomtuned_c40-139.feather` it is compared against.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then unset PYTORCH_CUDA_ALLOC_CONF; else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CKPT=${CKPT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow/blendflow_rw1k_nll01_s501.pt}
TAG=${TAG:-rw1k_nll01_s501}
OUT=${OUT:-results/blend_lookup_flow_${TAG}_c40-139.feather}

echo "### BLEND LOOKUP FROM FLOW #2  ckpt=$CKPT job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/build_blend_lookup_flow.py \
  --cases $(seq 40 139) --checkpoint "$CKPT" --output "$OUT" \
  --aperture ${APERTURE:-7.0} --allow-extrapolation \
  || { echo "FAILED"; exit 1; }

echo "--- end-to-end constgold m, in-domain primaries only ---"
python -u scripts/eval_m_with_blendflow.py --flow-lookup "$OUT" --in-domain-only \
  || { echo "M EVAL FAILED"; exit 1; }
echo "BLLOOKUP_DONE"; date
