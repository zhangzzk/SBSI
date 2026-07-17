"""Decisive nonlinearity number: mean per-primary R_blend at g=0.02 vs g=0.2 (both isotropic half-shear,
same fields 40-59). The per-bin dip curve is noise-dominated at g=0.02; the MEAN averages it down.
  <R_blend>(0.02) > (0.2)  -> emulator (trained at 0.2) UNDER-predicts constgold -> nonlinearity IS a
                              deficit driver (principled fix: retrain at 0.02).
  <R_blend>(0.02) <= (0.2) -> nonlinearity is the wrong sign; the deficit is coherent cross-term, not g.
Applies the same regression cuts as the emulator; bootstrap-by-case error.
"""
import numpy as np, pandas as pd, pyarrow.feather as pf

R002 = "/home/z/Zekang.Zhang/SBSI/results/resp002_c40-59.feather"
R02 = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
CASES = list(range(40, 60))
REG = dict(rs=(13, 29), rp=(18, 28), Res=(0.0, 10.0), Rep=(0.1, 1.5), dist=(0, 10))

def load_cut(path, from_train):
    cols = ["case", "input_index", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s", "distance", "delta_et1"]
    t = pf.read_table(path, columns=cols).to_pandas()
    if from_train:
        t = t[t.case.isin(CASES)]
    m = ((t.r_input_s > REG["rs"][0]) & (t.r_input_s < REG["rs"][1]) & (t.r_input_p > REG["rp"][0]) & (t.r_input_p < REG["rp"][1])
         & (t.Re_input_s > REG["Res"][0]) & (t.Re_input_s < REG["Res"][1]) & (t.Re_input_p > REG["Rep"][0]) & (t.Re_input_p < REG["Rep"][1])
         & (t.distance > REG["dist"][0]) & (t.distance < REG["dist"][1]) & np.isfinite(t.delta_et1))
    return t[m].reset_index(drop=True)

def per_primary(t, gamma):
    t = t.copy(); t["resp"] = t.delta_et1.to_numpy(float) / gamma
    g = t.groupby(["case", "input_index"]).agg(R=("resp", "sum")).reset_index()
    return g

def boot(g, ncases, seed=0):
    rng = np.random.default_rng(seed); cases = g.case.unique(); vals = []
    per = {c: (g.R[g.case == c].sum(), (g.case == c).sum()) for c in cases}
    for _ in range(200):
        pick = rng.choice(cases, len(cases), replace=True)
        s = sum(per[c][0] for c in pick); n = sum(per[c][1] for c in pick); vals.append(s / n)
    return np.std(vals)

print("loading g=0.02 ...", flush=True); a = per_primary(load_cut(R002, False), 0.02)
print("loading g=0.20 ...", flush=True); b = per_primary(load_cut(R02, True), 0.2)
ma, mb = a.R.mean(), b.R.mean()
ea, eb = boot(a, 20), boot(b, 20)
print(f"\n=== mean per-primary R_blend (cases 40-59, reg cuts) ===")
print(f"  g=0.02 (constgold shear) : <R_blend> = {ma:.4f} +/- {ea:.4f}   ({len(a):,} primaries)")
print(f"  g=0.20 (emulator shear)  : <R_blend> = {mb:.4f} +/- {eb:.4f}   ({len(b):,} primaries)")
print(f"  ratio 0.02/0.20 = {ma/mb:.3f}   diff = {ma-mb:+.4f} +/- {np.hypot(ea,eb):.4f}")

# FIELD-PAIRED: same fields at both shears -> field variance cancels in the per-case difference/ratio.
pa = a.groupby("case").R.mean(); pb = b.groupby("case").R.mean()
common = sorted(set(pa.index) & set(pb.index))
da = pa.loc[common].to_numpy(); db = pb.loc[common].to_numpy()
diff = da - db; ratio = da / db
print(f"\n=== FIELD-PAIRED per-case (n={len(common)} cases; field variance cancels) ===")
print(f"  per-case <R>(0.02): mean={da.mean():.4f}   per-case <R>(0.20): mean={db.mean():.4f}")
print(f"  paired diff (0.02-0.20) = {diff.mean():+.4f} +/- {diff.std(ddof=1)/np.sqrt(len(diff)):.4f}  "
      f"({diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))):+.1f} sigma)")
print(f"  paired ratio 0.02/0.20  = {ratio.mean():.3f} +/- {ratio.std(ddof=1)/np.sqrt(len(ratio)):.3f}")
print(f"  >0 diff => nonlinearity UNDER-predicts at g=0.2 => deficit driver (retrain at 0.02).")
print(f"  <=0     => wrong sign => deficit is the coherent cross-term, not shear-mag.")
print("RBLEND_NONLIN_DONE")
