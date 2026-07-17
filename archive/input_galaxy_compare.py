"""Last piece: is the ~1.5-3% lower shear response in 40-79 (at fixed magnitude) explained by a different
INPUT galaxy sample? Compare Re / sersic_n / axis_ratio at fixed magnitude between cases 0-39 and 40-79.
Smaller Re at fixed mag in 40-79 -> more PSF dilution -> lower response -> the render diff is galaxy-sample
(seed-driven) variance. Identical galaxy props -> the diff is in the render/measurement given the same inputs.
"""
import numpy as np, pandas as pd, pyarrow.feather as pf
CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"; TILE = "tile180.0_-0.5"
COLS = ["r_input", "Re_input", "sersic_n_input", "axis_ratio_input"]
EDGES = np.array([18, 23, 24, 25, 26, 27, 28.1])

def load(cases):
    fr = []
    for c in cases:
        fp = f"{CBASE}/case{c}_0.02/real0/catalogues/input/gals_info_{TILE}.feather"
        fr.append(pf.read_table(fp, columns=COLS).to_pandas())
    return pd.concat(fr, ignore_index=True)

a = load(range(0, 40)); b = load(range(40, 80))
print(f"0-39: {len(a):,} gals   40-79: {len(b):,} gals")
print(f"\n{'mag':>10} {'N_0-39':>9} {'N_40-79':>9} | {'Re 0-39':>8} {'Re 40-79':>9} {'dRe%':>6} | "
      f"{'n 0-39':>7} {'n 40-79':>7} | {'q 0-39':>7} {'q 40-79':>7}")
for i in range(len(EDGES) - 1):
    lo, hi = EDGES[i], EDGES[i + 1]
    da = a[(a.r_input >= lo) & (a.r_input < hi)]; db = b[(b.r_input >= lo) & (b.r_input < hi)]
    if len(da) < 1000 or len(db) < 1000:
        continue
    print(f"  {lo:.0f}-{hi:.0f}".rjust(10) + f" {len(da):>9,} {len(db):>9,} | "
          f"{da.Re_input.median():>8.3f} {db.Re_input.median():>9.3f} {(db.Re_input.median()/da.Re_input.median()-1)*100:>+6.1f} | "
          f"{da.sersic_n_input.median():>7.2f} {db.sersic_n_input.median():>7.2f} | "
          f"{da.axis_ratio_input.median():>7.2f} {db.axis_ratio_input.median():>7.2f}")
print(f"\noverall: <Re> 0-39={a.Re_input.median():.3f} 40-79={b.Re_input.median():.3f}  "
      f"<mag> {a.r_input.mean():.3f} vs {b.r_input.mean():.3f}  Ngal/case {len(a)/40:.0f} vs {len(b)/40:.0f}")
print("INPUT_GAL_DONE")
