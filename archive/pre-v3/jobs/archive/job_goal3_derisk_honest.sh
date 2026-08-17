#!/bin/bash
#SBATCH --job-name=g3_honest
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/g3_honest_%j.out

# UNIFIED-OBJECTIVE DE-RISK (task #36), firewall-clean. Goal-3 collider de-risk on the HONEST
# certified pipeline: certified R_flow (dump default) + EMULATOR R_blend (dump default, constgold-FREE)
# -- the empty --scene-rblend forces the fallback to the certified density lookup, so the CIRCULAR
# scene R_blend (rblend_scene_prodrflow) is NOT used. constgold r_sim = validation truth only.
# Question: with the honest pipeline, does inferring true mag/size from measured observables and
# cutting on the inferred truth (CURE) recover the true-property-cut level (TARGET), dissolving the
# measured-cut blowup (DISEASE, 40-68% in selrobust_current_meas)? Windows fixed a priori.
#   CURE~TARGET<<DISEASE => measured selection dissolvable to the true-property level -> build joint flow.
#   CURE~DISEASE>>TARGET => fundamental collider -> rethink before GPU spend.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### G3_HONEST job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
echo "R_flow=certified (dump); R_blend=EMULATOR (dump default, constgold-free); scene R_blend DISABLED"
stdbuf -oL -eL python -B -u scripts/goal3_collider_derisk.py \
  --scene-rblend "" --tag cert_honest \
  || { echo "G3_HONEST FAILED"; exit 1; }
echo "### G3_HONEST_JOB_DONE job=$SLURM_JOB_ID ###"; date
