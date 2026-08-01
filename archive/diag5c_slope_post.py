"""Post-process `diag5c_slope.py` .npz output: the drift-vs-bank separation.

The (5.9b) prediction as written divides by `I_keep = <I> - I_sel`, which is the very
cancellation that round 1 showed is consistent with zero -- so the prediction inherits an
unusable error bar.  Two well-conditioned things can still be said:

 (a) substitute `Var(s - <s>_sel)` for `I_keep`.  At gamma = 0 the information equality
     says they are the same number, and Var is measured to ~4%, so this is the (5.9b)
     prediction with its ill-conditioned factor replaced by its own theoretical equal.
 (b) the drift is LINEAR IN GAMMA and vanishes at gamma = 0, so fitting
     m_(5.9)(gamma) = m0 + c*gamma over a 20x range in gamma separates the two:
     `m0` is the gamma-independent (bank / g=0) failure, `c*gamma` is all the room a drift
     of any origin has to live in.

Runs on the saved arrays only -- no GPU, no model, negligible cost.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b")
from sbs_shear.lagrangian_score import shear_estimate_bartlett  # noqa: E402


def analyse(path, B=2000):
    z = np.load(path)
    G, S, I, KEEP = z["gammas"], z["S"], z["I"], z["KEEP"]
    s_sel, i_sel = float(z["s_sel"]), float(z["i_sel"])
    n_gal = S.shape[1]
    rng = np.random.default_rng(7)
    nz = G != 0

    def stats(idx):
        m59, mu3, cov, var_, Ic, meanc = [], [], [], [], [], []
        for gi in range(len(G)):
            sel = idx[KEEP[gi, idx]]
            sk = S[gi, sel] - s_sel
            ik = I[gi, sel] - i_sel
            m59.append(shear_estimate_bartlett(S[gi, sel], s_sel))
            mu3.append(np.mean((sk - sk.mean()) ** 3))
            cov.append(np.mean((ik - ik.mean()) * (sk - sk.mean())))
            var_.append(np.mean(sk ** 2))
            Ic.append(ik.mean())
            meanc.append(sk.mean())
        m59 = np.array(m59)
        # PAIRED, offset-free: the g=0 row shares the same bank and the same galaxies by
        # common random numbers, so subtracting it removes any galaxy/bank offset that is
        # common to all g.  This is the cleanest denominator-free per-gamma m.
        meanc = np.array(meanc)
        mpair = np.where(nz, (meanc - meanc[G == 0][0]) / np.where(nz, G, 1)
                         / var_[int(np.flatnonzero(G == 0)[0])] - 1, np.nan)
        mfrac = np.where(nz, m59 / np.where(nz, G, 1) - 1, np.nan)
        A = np.vstack([np.ones(nz.sum()), G[nz]]).T
        c0, c1 = np.linalg.lstsq(A, mfrac[nz], rcond=None)[0]
        drift_var = -G * (np.array(mu3) - np.array(cov)) / np.array(var_)
        drift_I = -G * (np.array(mu3) - np.array(cov)) / np.array(Ic)
        p0, p1 = np.linalg.lstsq(A, mpair[nz], rcond=None)[0]
        return mfrac, c0, c1, drift_var, drift_I, mpair, p0, p1

    idx0 = np.arange(n_gal)
    mfrac, c0, c1, dv, dI, mpair, p0f, p1f = stats(idx0)
    bs = [stats(rng.integers(0, n_gal, n_gal)) for _ in range(B)]
    se = lambda f: np.std(np.array([f(b) for b in bs], dtype=float), axis=0, ddof=1)

    print(f"\n=== {path.split('/')[-1]} ===")
    print("m_(5.9) vs gamma  -- a DRIFT of any origin is linear in gamma and 0 at gamma=0")
    print(f"{'gamma':>8} | {'m_(5.9)':>18} | {'m PAIRED (g=0 sub.)':>20} | "
          f"{'(5.9b) with Var(s)':>20} | {'(5.9b) with <I>':>18}")
    se_m = se(lambda b: b[0])
    se_dv = se(lambda b: b[3])
    se_dI = se(lambda b: b[4])
    se_mp = se(lambda b: b[5])
    for gi, g in enumerate(G):
        if g == 0:
            continue
        print(f"{g:>8.3f} | {mfrac[gi]:>10.2%} +- {se_m[gi]:<6.2%} | "
              f"{mpair[gi]:>10.2%} +- {se_mp[gi]:<6.2%} | "
              f"{dv[gi]:>10.2%} +- {se_dv[gi]:<6.2%} | {dI[gi]:>10.2%} +- {se_dI[gi]:<6.2%}")
    se_c0, se_c1 = se(lambda b: b[1]), se(lambda b: b[2])
    print(f"\n  FIT m_(5.9)(gamma) = m0 + c*gamma  over gamma in [{G[nz].min()},{G[nz].max()}]:")
    print(f"    m0 (gamma-INDEPENDENT failure) = {c0:+.2%} +- {se_c0:.2%}")
    print(f"    c  (all the room a drift has)  = {c1:+.3f} +- {se_c1:.3f} per unit gamma")
    print(f"    => drift-attributable part of m at gamma=0.05: {c1*0.05:+.2%} +- {se_c1*0.05:.2%}")
    print(f"    => fraction of the gamma=0.05 discrepancy explained by ANY linear drift: "
          f"{abs(c1*0.05)/abs(mfrac[np.argmin(np.abs(G-0.05))]):.2%}")
    se_p0, se_p1 = se(lambda b: b[6]), se(lambda b: b[7])
    print(f"\n  SAME FIT on the PAIRED m (g=0 offset removed -- the clean version):")
    print(f"    m0 (gamma-INDEPENDENT failure) = {p0f:+.2%} +- {se_p0:.2%}")
    print(f"    c  (all the room a drift has)  = {p1f:+.3f} +- {se_p1:.3f} per unit gamma")
    print(f"    => drift-attributable part at gamma=0.05: {p1f*0.05:+.2%} +- {se_p1*0.05:.2%}"
          f"   ({abs(p1f*0.05)/abs(p0f):.1%} of the gamma-independent failure)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyse(p)
