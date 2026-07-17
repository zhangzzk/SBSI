"""DECISIVE branch test: emulator blend-response vs HALF-SHEAR TRUTH on the leakage-free gold pairs.

The ho emulator was trained on cases 40-199 (gold 0-39 excluded), so response_catalogue_train cases
0-39 are held-out truth for the constgold population (same fields/positions, verified). The emulator
uses NO angle feature => its 'response' is the isotropic <delta_et1/gamma>. The apples-to-apples truth
is therefore delta_et1/gamma per pair, and the coherent (isotropic) blend per primary is Sum_neighbours
delta_et1/gamma. We compare emulator vs truth:
  (A) per-pair, binned by neighbour mag r_s & distance  -> where is the emulator biased on OUR pairs?
  (B) per-primary SUM, binned by the production R_blend quantile (q3 = the +9% constgold regime)
      -> does truth_sum - R_blend(emulator) ~ +0.047 in q3 (=> emulator under-predicts => branch B),
         or ~0 (=> emulator fine; deficit is R_flow or coherent-anisotropy => branch A/C)?
Clean-bin sanity: at r_s in [18,24] the emulator was ~unbiased, so pred/truth there should be ~1
(also validates the gamma scaling).
"""
import os, sys, numpy as np, pandas as pd, pyarrow.feather as pf, yaml
sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from blendemu.inference import BlendingPredictor

CFG = "/home/z/Zekang.Zhang/blendemu/configs/fs2_lsst_r_extnbr_ho.yaml"
RESP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
BLEND_LOOKUP = "/home/z/Zekang.Zhang/SBSI/results/blend_lookup_extnbrho_c0-39.feather"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
REG = dict(rs=(13, 29), rp=(18, 28), Res=(0.0, 10.0), Rep=(0.1, 1.5), dist=(0, 10))

cfg = yaml.safe_load(open(CFG))
gamma = float(max(cfg["simulation"]["response"]["shear_values"]))
print(f"gamma(response) = {gamma}", flush=True)

cols = ["case", "input_index", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
        "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1"]
print("reading gold rows from response_catalogue_train ...", flush=True)
t = pf.read_table(RESP, columns=cols).to_pandas()
t = t[t["case"] < 40].reset_index(drop=True)
print(f"gold(<40) rows: {len(t):,}", flush=True)

m = ((t.r_input_s > REG["rs"][0]) & (t.r_input_s < REG["rs"][1])
     & (t.r_input_p > REG["rp"][0]) & (t.r_input_p < REG["rp"][1])
     & (t.Re_input_s > REG["Res"][0]) & (t.Re_input_s < REG["Res"][1])
     & (t.Re_input_p > REG["Rep"][0]) & (t.Re_input_p < REG["Rep"][1])
     & (t.distance > REG["dist"][0]) & (t.distance < REG["dist"][1])
     & np.isfinite(t.delta_et1))
t = t[m].reset_index(drop=True)
print(f"after regression cuts + finite: {len(t):,} pairs", flush=True)

pred = BlendingPredictor.load("/home/z/Zekang.Zhang/blendemu/models", tag="lsst_r_extnbr_ho",
                              conditions=COND, device="cpu")
out = pred.predict_on_pairs(t, task="response", warn_extrapolation=False)
t["pred"] = out["response"].to_numpy(float)
t["truth"] = t["delta_et1"].to_numpy(float) / gamma

# ---------------- (A) per-pair by neighbour magnitude & distance ----------------
print("\n=== (A) per-pair emulator vs truth, by neighbour mag r_s ===")
print(f"  {'r_s bin':>11} {'<pred>':>9} {'<truth>':>9} {'pred-truth':>11} {'ratio':>7} {'N':>10}")
edges = [13, 24, 25, 25.5, 26, 26.5, 27, 27.5, 29]
for i in range(len(edges) - 1):
    b = (t.r_input_s >= edges[i]) & (t.r_input_s < edges[i + 1])
    if b.sum() < 1000:
        continue
    p, tr = t.pred[b].mean(), t.truth[b].mean()
    print(f"  {edges[i]:.1f}-{edges[i+1]:.1f}".rjust(11)
          + f" {p:>9.4f} {tr:>9.4f} {p-tr:>+11.4f} {p/tr if abs(tr)>1e-6 else np.nan:>7.2f} {int(b.sum()):>10,}")

print("\n=== (A) per-pair by distance ===")
print(f"  {'dist bin':>11} {'<pred>':>9} {'<truth>':>9} {'pred-truth':>11} {'N':>10}")
dedges = [0, 1, 2, 3, 4, 5, 7, 10]
for i in range(len(dedges) - 1):
    b = (t.distance >= dedges[i]) & (t.distance < dedges[i + 1])
    if b.sum() < 1000:
        continue
    p, tr = t.pred[b].mean(), t.truth[b].mean()
    print(f"  {dedges[i]}-{dedges[i+1]}\"".rjust(11) + f" {p:>9.4f} {tr:>9.4f} {p-tr:>+11.4f} {int(b.sum()):>10,}")

# ---------------- (B) per-primary coherent (isotropic) sum ----------------
g = t.groupby(["case", "input_index"]).agg(pred_sum=("pred", "sum"), truth_sum=("truth", "sum"),
                                           npair=("pred", "size")).reset_index()
bl = pf.read_table(BLEND_LOOKUP).to_pandas()[["case", "input_index", "R_blend"]]
g = g.merge(bl, on=["case", "input_index"], how="left")
print(f"\n=== (B) per-primary sum: {len(g):,} primaries; "
      f"pred_sum vs production R_blend corr check: <pred_sum>={g.pred_sum.mean():.4f} "
      f"<R_blend>={g.R_blend.mean():.4f} (should match) ===")
rb = g["R_blend"].fillna(g["pred_sum"]).to_numpy(float)
hi = rb >= 0.02
qe = np.quantile(rb[hi], np.linspace(0, 1, 5)); qe[0] -= 1e-9; qe[-1] += 1e-9
bi = np.where(~hi, 0, 1 + np.clip(np.digitize(rb, qe) - 1, 0, 3))
print(f"  {'R_blend bin':>12} {'<R_bl_emu>':>10} {'<truth_sum>':>11} {'truth-emu':>10} {'<npair>':>8} {'N':>9}")
for c in range(5):
    b = bi == c
    if b.sum() < 2000:
        continue
    emu, tr = g.pred_sum[b].mean(), g.truth_sum[b].mean()
    rbm = g.R_blend[b].mean()
    tag = "ISO" if c == 0 else f"q{c}"
    print(f"  {tag:>12} {rbm:>10.4f} {tr:>11.4f} {tr-rbm:>+10.4f} {g.npair[b].mean():>8.1f} {int(b.sum()):>9,}")
print("\nINTERPRETATION: truth_sum - <R_bl_emu> in q3 ~ +0.047 => emulator UNDER-predicts coherent blend")
print("(branch B, recalibratable). ~0 => emulator fine; the q3 deficit is R_flow or coherent anisotropy.")
print("AUDIT_BLEND_DONE")
