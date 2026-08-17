#!/bin/bash
#SBATCH --job-name=fwd_proto99
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fwd_proto99_%j.out

# ROBUSTNESS check for the unified-forward-model prototype (EXPERIMENTAL, NOT certified):
# scale from cases 0-19 to 0-99 (train 0-79, validate 80-99) and repeat over 3 seeds to confirm
# the feasibility gate is not a small-sample fluke.  Keeps the SAME 2D shape target + nearest-
# neighbour scene (this is a scale/seed check, NOT the 4D/all-neighbour extension).  Reuses
# scripts/train_forward_prototype.py unchanged except the added count-weighted b_true report.
# Additive-only; never wired into m = R_sim/(R_flow+R_blend)-1.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
SEEDS=${SEEDS:-421 422 423}
echo "### FWD_PROTO99 job=$SLURM_JOB_ID node=$SLURMD_NODENAME seeds='$SEEDS' ###"; nvidia-smi -L; date

for S in $SEEDS; do
  echo "===== SEED $S ====="; date
  stdbuf -oL -eL python -B -u scripts/train_forward_prototype.py \
    --outdir "$OUTDIR" --tag "proto_c0-99_seed${S}" \
    --max-case 100 --train-case-max 80 \
    --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
    --n-flux 4 --n-size 3 --delta 0.05 \
    --lam-r 50 --lam-s 50 --lam-d 1 \
    --epochs 50 --patience 12 --no-sensitivity \
    --batch-size 16384 --lr 7e-4 --seed "$S" \
    --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
    --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
    || { echo "FWD_PROTO99 FAILED seed=$S"; exit 1; }
  echo "SEED_DONE $S"; date
done

echo "===== AGGREGATE (per-seed + mean +/- scatter) ====="
python -B - "$OUTDIR" $SEEDS <<'PY'
import sys, os, json, numpy as np
outdir = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]]
rows, dP_bins, R_bins, Rself_bins, b_true = [], [], [], [], None
for s in seeds:
    f = os.path.join(outdir, f"validation_proto_c0-99_seed{s}.npz")
    if not os.path.exists(f):
        print("MISSING", f); continue
    z = np.load(f, allow_pickle=True); v3 = json.loads(str(z["v3"]))
    rows.append(dict(
        seed=s, R=float(z["R_global"]), Rtruth=float(z["rself_val_global"]),
        relerr=100 * (float(z["R_global"]) / float(z["rself_val_global"]) - 1),
        dP=float(z["dP_global"]), b_popw=float(z["b_true_popw"]), global_b=float(z["global_b"]),
        nll_dt=float(v3["joint_nll"]) - float(v3["flowbase_nll"]),
        bce_dt=float(v3["joint_bce"]) - float(v3["detbase_bce"]),
        detfrac=float(z["det_frac"])))
    dP_bins.append(np.asarray(z["dP_bin"], float)); R_bins.append(np.asarray(z["R_bin"], float))
    Rself_bins.append(np.asarray(z["rself_bin_val"], float)); b_true = np.asarray(z["b_true"], float)

def ms(key):
    a = np.array([r[key] for r in rows], float); return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0

L = []
L.append("SBSI unified forward-model PROTOTYPE -- ROBUSTNESS (cases 0-99, train 0-79 / val 80-99), 3 seeds")
L.append("EXPERIMENTAL / not certified; 2D shape + nearest-neighbour (scale/seed check only).")
L.append("")
L.append("PER-SEED:")
L.append(f"  {'seed':>5} {'V1<R>':>8} {'truth':>7} {'relerr%':>8} | "
         f"{'V2 dP(cw)':>9} {'b_true(cw)':>10} {'glob_b':>7} | {'V3 dNLL':>8} {'V3 dBCE':>8}")
for r in rows:
    L.append(f"  {r['seed']:>5} {r['R']:>+8.4f} {r['Rtruth']:>7.4f} {r['relerr']:>+8.2f} | "
             f"{r['dP']:>+9.4f} {r['b_popw']:>+10.4f} {r['global_b']:>+7.4f} | "
             f"{r['nll_dt']:>+8.4f} {r['bce_dt']:>+8.4f}")
L.append("")
L.append("MEAN +/- SCATTER (std, ddof=1) across seeds:")
for k, lbl in [("R", "V1 <R_model>"), ("relerr", "V1 rel.err %"),
               ("dP", "V2 dP count-weighted"), ("b_popw", "V2 b_true count-weighted"),
               ("nll_dt", "V3 joint-flowbase NLL delta"), ("bce_dt", "V3 joint-detbase BCE delta")]:
    m, s = ms(k); L.append(f"  {lbl:<32} = {m:+.4f} +/- {s:.4f}")
L.append(f"  (npz global_b reference = {rows[0]['global_b']:+.4f}; det_frac ~ {rows[0]['detfrac']:.3f})")
L.append("")
L.append("V2 dP/dg per mag bin  (mean +/- std across seeds) vs b_true:")
dP = np.array(dP_bins); mdP, sdP = np.nanmean(dP, 0), np.nanstd(dP, 0, ddof=1)
L.append("  mag_bin : " + "  ".join(f"{i}" for i in range(len(b_true))))
L.append("  dP mean : " + "  ".join(f"{v:+.3f}" for v in mdP))
L.append("  dP std  : " + "  ".join(f"{v:.3f}" for v in sdP))
L.append("  b_true  : " + "  ".join(f"{v:+.3f}" for v in b_true))
L.append("")
L.append("V1 R_model per (flux x size) bin (mean across seeds) vs R_self val truth:")
Rm = np.nanmean(np.array(R_bins), 0); Rt = np.nanmean(np.array(Rself_bins), 0)
L.append("  R_model : " + "  ".join(f"{v:+.3f}" for v in Rm))
L.append("  R_self  : " + "  ".join(f"{v:+.3f}" for v in Rt))
rep = "\n".join(L); print("\n" + rep)
with open(os.path.join(outdir, "validation_proto_c0-99_summary.txt"), "w") as fh:
    fh.write(rep + "\n")
print("\nSaved:", os.path.join(outdir, "validation_proto_c0-99_summary.txt"))
PY

echo "### FWD_PROTO99_ALL_DONE job=$SLURM_JOB_ID ###"; date
