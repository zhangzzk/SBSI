#!/bin/bash
#SBATCH --job-name=rbgapmeas
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbgapmeas_%j.out

# FLOW vs EMULATOR on the faint shell, settled with the PER-PAIR ruler (no constgold anywhere).
#
# Constgold's mag>26 measured shell shows R_sim 0.34088 vs model 0.44900 with R_flow alone at 0.33624
# -- suggestive that R_blend is the culprit, but constgold only measures the SUM. This runs the
# half-shear per-pair R_blend ruler binned by the PRIMARY'S MEASURED magnitude / S/N / size, which
# separates them. The 45-degree null is reported PER BIN, so a bin-selection artefact cannot hide.
#
# CFG=nn3  (default) : the standard 3"-aperture ruler catalogue, one (nearest) neighbour per primary
#                      -- reproduces the published ruler numbers exactly, iid errors.
# CFG=ap7            : the 7"-aperture catalogue with EVERY annotated neighbour row kept (~4.2 rows
#                      per primary). Each neighbour carries its own independent shear direction, so
#                      these are ~4x more (weakly correlated) measurements of the same quantity;
#                      errors are cluster-robust with the PRIMARY as the cluster.
# FIREWALL: half-shear legs only. Nothing is trained, nothing is tuned.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
C=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D

CFG=${CFG:-nn3}
if [ "$CFG" = "ap7" ]; then
  GS=$C/det_meas_ngmix_ap7_g0.05_val.feather
  G0=$C/det_meas_ngmix_ap7_g0.0_train.feather
  EXTRA="--all-neighbours"
else
  GS=$C/det_meas_ngmix_g0.05_val.feather
  G0=$C/det_meas_ngmix_g0.0_train.feather
  EXTRA=""
fi

echo "### RBLEND GAP by MEASURED props  job=$SLURM_JOB_ID  CFG=$CFG ###"; date
python -u scripts/eval_rblend_gap_measured.py \
  --gs-leg "$GS" --g0-leg "$G0" $EXTRA \
  --true-re-min ${REMIN:-0.3} --true-mag-max ${MAGMAX:-26.0} \
  --tags ${TAGS:-lsst_r_extnbr_indom_tuned lsst_r_extnbr_ho} \
  --output "$D/rblend_gap_measured_${CFG}.npz"
echo "RBGAPMEAS_DONE"; date
