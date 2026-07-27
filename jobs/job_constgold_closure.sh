#!/bin/bash
#SBATCH --job-name=cg_closure
#SBATCH --time=01:40:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_closure_%j.out

# THE make-or-break test: run the reframe joint-model ENSEMBLE (seeds 421-424) on the constgold
# true-cut population, assemble the coherent shape-response closure R_sim/(R_flow*bridge[+R_blend])-1.
# Decisive band = ISOLATED (neighbored=False): pure flow test, no R_blend, where the certified
# measured-conditioned flow failed at -8..-18%. FIREWALL: constgold read for validation only.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

TAG=${TAG:-ensemble}
CKPTGLOB=${CKPTGLOB:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_joint_c0-99_truecut_seed*_joint.pt}
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/constgold_closure_${TAG}.npz
echo "### CG_CLOSURE job=$SLURM_JOB_ID node=$SLURMD_NODENAME tag=$TAG ###"; nvidia-smi -L; date
echo "ckpt-glob=$CKPTGLOB  EXTRA=$EXTRA"
python -B -u scripts/eval_constgold_closure.py \
  --ckpt-glob "$CKPTGLOB" \
  --bridge 1.0853 --per-mag-bridge \
  --max-case 40 \
  --true-re-min 0.3 --true-mag-max 26 \
  --output "$OUT" $EXTRA \
  || { echo "CG_CLOSURE FAILED"; exit 1; }
echo "### CG_CLOSURE_DONE job=$SLURM_JOB_ID ###"; date
