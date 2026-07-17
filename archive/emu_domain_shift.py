"""DIRECT test of the covariate-shift hypothesis for the +1.7% constgold residual.

Established: linear sum is exact (toy), flow self-response is clean -> the residual lives entirely
in the emulator's per-pair R_blend, and a marginal Delta(r_s, distance) correction (built on the
emulator's held-out validation) does NOT close it (slightly worsens: +1.69 -> +1.82%).

That non-transfer is the signature of covariate shift: at fixed (r_s, distance) the emulator's
per-pair residual depends on OTHER covariates (target mag r_p, size, ...), and the constgold
per-pair population is distributed differently along those axes than the training/validation set.
A marginal correction then applies the WRONG average.

This script proves it three ways:
  (A) emulator residual on its OWN held-out set, binned 2D by (r_s x r_p) -> does the bias vary
      with target mag at fixed neighbour mag?  (if yes, a (r_s,distance)-only correction is blind
      to it)
  (B) the per-pair population itself: TRAINING pairs vs CONSTGOLD scene pairs, distribution of
      r_p / r_s / distance -> where does constgold sit that training does not?
  (C) the clincher: mean emulator bias RE-WEIGHTED by the training population vs by the constgold
      population.  If they differ, the validation-derived correction is mis-weighted for constgold.
"""
import sys, numpy as np, pandas as pd, pyarrow.feather as pf, os
sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from blendemu.config import load_config
from blendemu import data_utils
from blendemu.inference import BlendingPredictor
from scripts.train_emulator import _catalogue_shear_scale
from sklearn.model_selection import train_test_split
import xgboost as xgb

CFG = "/home/z/Zekang.Zhang/blendemu/configs/fs2_lsst_r_extnbr_ho.yaml"
CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"

RS_EDGES = np.array([13, 24.0, 25.0, 25.7, 26.0, 26.5, 27.0, 27.5, 28.0, 29.01])
RP_EDGES = np.array([18, 23.0, 24.0, 24.5, 25.0, 25.5, 26.0, 27.5])


def lbl(edges):
    return [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]


# ---------- TRAINING-SET residual (emulator's own held-out) ----------
print("### (A) emulator residual on held-out training pairs, 2D by (r_s x r_p) ###", flush=True)
cfg = load_config(CFG); tr = cfg['training']; rc = tr['rescale']
raw = ["Re_input_p", "Re_input_s", "r_input_p", "r_input_s", "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1"]
df = pf.read_table(CAT, columns=raw).to_pandas()
df = data_utils.source_select_reg(df, cuts=tr['regression_cuts'])
df = data_utils.rescale(df, pixel_rms=rc['pixel_rms'], pixel_size=rc['pixel_size'], zero_mag=rc['zero_mag'],
                        psf_fwhm=rc['psf_fwhm'], moffat_beta=rc['moffat_beta'])
shear = _catalogue_shear_scale(cfg, 'response')
y = (df["delta_et1"] / shear).to_numpy(float)
feats = tr['features']
xtr, xte, ytr, yte = train_test_split(df[feats], y, test_size=tr['test_size'], random_state=tr['random_state'])
pred = BlendingPredictor.load("/home/z/Zekang.Zhang/blendemu/models", tag="lsst_r_extnbr_ho", conditions=COND, device="cpu")
p = pred.bst_reg.predict(xgb.DMatrix(xte)) * pred.y_std + pred.y_mean
t = np.asarray(yte)
idx = xte.index.to_numpy()
rp_tr = df.loc[idx, "r_input_p"].to_numpy(float)
rs_tr = df.loc[idx, "r_input_s"].to_numpy(float)
print(f"TEST rows={len(t):,}  global bias <pred>/<true>-1 = {p.mean()/t.mean()-1:+.2%}", flush=True)
_hdr = "r_s|r_p"
print(f"  {_hdr:>10} " + " ".join(f"{c:>12}" for c in lbl(RP_EDGES)), flush=True)
for i in range(len(RS_EDGES) - 1):
    ms = (rs_tr >= RS_EDGES[i]) & (rs_tr < RS_EDGES[i + 1])
    row = f"  {lbl(RS_EDGES)[i]:>10} "
    for j in range(len(RP_EDGES) - 1):
        mp = ms & (rp_tr >= RP_EDGES[j]) & (rp_tr < RP_EDGES[j + 1])
        if mp.sum() < 500:
            row += f"{'.':>12} "; continue
        b = p[mp].mean() / t[mp].mean() - 1 if abs(t[mp].mean()) > 1e-9 else np.nan
        row += f"{b:+.0%}({mp.sum()//1000}k)".rjust(12) + " "
    print(row, flush=True)

# per-pair TRUE and PRED, plus raw covariates, kept for (C)
train_pairs = pd.DataFrame(dict(r_p=rp_tr, r_s=rs_tr,
                                dist=df.loc[idx, "distance"].to_numpy(float), true=t, pred=p))

# ---------- CONSTGOLD scene pair population ----------
print("\n### (B) per-pair population: TRAINING vs CONSTGOLD scene ###", flush=True)
cg_parts = []
for c in range(6):
    fp = f"{CBASE}/case{c}_0.02/real0/catalogues/input/gals_info_{TILE}.feather"
    if not os.path.exists(fp):
        continue
    tt = pf.read_table(fp).to_pandas()
    tt = tt.rename(columns={col: col.replace("_input", "") for col in tt.columns})
    reg = pred.predict_response(tt, tt)
    cg_parts.append(pd.DataFrame(dict(r_p=reg["r_input_p"].to_numpy(float),
                                      r_s=reg["r_input_s"].to_numpy(float),
                                      dist=reg["distance"].to_numpy(float))))
    print(f"  case{c}: {len(reg):,} pairs", flush=True)
cg = pd.concat(cg_parts, ignore_index=True)


def dist_line(name, v, edges):
    tot = len(v)
    frac = [((v >= edges[i]) & (v < edges[i + 1])).mean() for i in range(len(edges) - 1)]
    print(f"  {name:>18}: " + " ".join(f"{f:6.1%}" for f in frac), flush=True)


for col, edges, nm in [("r_s", RS_EDGES, "neighbour mag r_s"), ("r_p", RP_EDGES, "target mag r_p")]:
    print(f"  --- {nm} fractional population ---", flush=True)
    print(f"  {'bins':>18}: " + " ".join(f"{c:>6}" for c in lbl(edges)), flush=True)
    dist_line("TRAINING", train_pairs[col].to_numpy(), edges)
    dist_line("CONSTGOLD", cg[col].to_numpy(), edges)

# ---------- (C) population-reweighted bias ----------
print("\n### (C) emulator mean bias re-weighted by each population's (r_s x r_p) cells ###", flush=True)
# per-cell emulator bias (from held-out training residual), then average with TRAIN vs CG cell weights
ri_tr = np.clip(np.digitize(train_pairs["r_s"], RS_EDGES) - 1, 0, len(RS_EDGES) - 2)
pj_tr = np.clip(np.digitize(train_pairs["r_p"], RP_EDGES) - 1, 0, len(RP_EDGES) - 2)
ri_cg = np.clip(np.digitize(cg["r_s"], RS_EDGES) - 1, 0, len(RS_EDGES) - 2)
pj_cg = np.clip(np.digitize(cg["r_p"], RP_EDGES) - 1, 0, len(RP_EDGES) - 2)
ncell = (len(RS_EDGES) - 1) * (len(RP_EDGES) - 1)
key_tr = ri_tr * (len(RP_EDGES) - 1) + pj_tr
key_cg = ri_cg * (len(RP_EDGES) - 1) + pj_cg
true = train_pairs["true"].to_numpy(); predv = train_pairs["pred"].to_numpy()
sum_true = np.bincount(key_tr, weights=true, minlength=ncell)
sum_pred = np.bincount(key_tr, weights=predv, minlength=ncell)
w_tr = np.bincount(key_tr, minlength=ncell).astype(float)
w_cg = np.bincount(key_cg, minlength=ncell).astype(float)
# cell-level ABSOLUTE emulator error contribution: (pred-true) per pair; population mean over pairs
cell_err = np.divide(sum_pred - sum_true, np.maximum(w_tr, 1))  # mean (pred-true) per cell
# mean per-pair blend magnitude per cell (true)
cell_true = np.divide(sum_true, np.maximum(w_tr, 1))
valid = w_tr >= 200
def wmean(err, w):
    ww = w * valid
    return (err * ww).sum() / max(ww.sum(), 1)
bias_tr = wmean(cell_err, w_tr) / max(wmean(cell_true, w_tr), 1e-9)
bias_cg = wmean(cell_err, w_cg) / max(wmean(cell_true, w_cg), 1e-9)
print(f"  mean per-pair (pred-true)/true, TRAIN-weighted  = {bias_tr:+.2%}", flush=True)
print(f"  mean per-pair (pred-true)/true, CONSTGOLD-weight = {bias_cg:+.2%}", flush=True)
print(f"  -> weighting shift changes the apparent per-pair bias by {bias_cg - bias_tr:+.2%}", flush=True)
print("EMU_DOMAIN_SHIFT_DONE", flush=True)
