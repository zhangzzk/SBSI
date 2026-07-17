"""Decisive test: does the coherent response become SUPER-ADDITIVE at CLOSE (sub-arcsec)
separations?  The constgold residual (+1.8% global) concentrates in the closest-distance
tercile (blend d1: +8.1%) and the moderate-high emulator R_blend bin (q3: +9.6%), while
truly-isolated (m_bare~0 -> flow is fine) and extreme-blend (q4: +0.0%) are clean.  Per-pair
(r_s,distance) emulator debiasing did NOT close it.

Two hypotheses make opposite toy predictions:
  (P) PHYSICS super-additivity: R_full > R_self + sum R_blend once isophotes overlap.
      -> excess grows as separation d -> 0.  Emulator's linear sum structurally can't capture it.
  (E) EMULATOR population bias: linear sum is exact (excess~0 in toy even at small d / many
      faint nbrs); the constgold residual is then a multi-dim OOD of the per-pair emulator.

Prior sweeps only tested d>=1.2" (excess~0).  Here we push to d in [0.4,2.0]" and add a
faint-crowding multiplicity ladder (the q3 / faint-OOD analog).  Seeded + SNC + averaged.
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from scripts.toy_blend_decompose import resp  # noqa  (render+measure+SNC response)


def decompose(target, neighbours, g=0.05, sky=10.0, nreal=300):
    def flag(gal, s):
        d = dict(gal); d["shear"] = s; return d
    only_t = [flag(target, True)] + [flag(n, False) for n in neighbours]
    Rs, es = resp(only_t, g, sky, nreal)
    blends, ebl = [], []
    for j in range(len(neighbours)):
        g_j = [flag(target, False)] + [flag(n, i == j) for i, n in enumerate(neighbours)]
        rb, eb = resp(g_j, g, sky, nreal); blends.append(rb); ebl.append(eb)
    alls = [flag(target, True)] + [flag(n, True) for n in neighbours]
    Rf, ef = resp(alls, g, sky, nreal)
    Rlin = Rs + sum(blends); ex = Rf - Rlin
    el = np.sqrt(es ** 2 + sum(e ** 2 for e in ebl)); eex = np.hypot(ef, el)
    return dict(Rs=Rs, sumb=float(sum(blends)), Rf=Rf, Rlin=Rlin, ex=ex, eex=eex)


def line(tag, r):
    frac = r["ex"] / r["Rf"] if abs(r["Rf"]) > 1e-9 else 0.0
    sig = r["ex"] / r["eex"] if r["eex"] > 0 else 0.0
    print(f"  {tag:<26} R_self={r['Rs']:+.3f}  sum_bl={r['sumb']:+.3f}  R_full={r['Rf']:+.3f}  "
          f"R_lin={r['Rlin']:+.3f}  EXCESS={r['ex']:+.3f}+/-{r['eex']:.3f} ({frac:+.0%}, {sig:+.1f}s)",
          flush=True)


def tgt(flux, hlr=0.4):
    return dict(hlr=hlr, flux=flux, e1=0.0, e2=0.0, x=0.0, y=0.0)


def nbr(d, ang, flux, hlr=0.4):
    a = np.deg2rad(ang)
    return dict(hlr=hlr, flux=flux, e1=0.0, e2=0.0, x=d * np.cos(a), y=d * np.sin(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sky", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--nreal", type=int, default=400)
    a = ap.parse_args()
    kw = dict(g=a.g, sky=a.sky, nreal=a.nreal)
    DS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]

    print(f"[regime] sky={a.sky} g={a.g} nreal={a.nreal}", flush=True)

    print("\n===== A. EQUAL-FLUX pair (flux=1000, S/N~11), separation scan =====")
    print("      (P) predicts EXCESS climbing as d->0.4; (E) predicts EXCESS~0 throughout.")
    for d in DS:
        line(f'pair @ {d:.1f}"', decompose(tgt(1000), [nbr(d, 40, 1000)], **kw))

    print("\n===== B. FAINT neighbour (flux=250, ~1.5mag fainter), separation scan =====")
    for d in DS:
        line(f'faint-nbr @ {d:.1f}"', decompose(tgt(1000), [nbr(d, 40, 250)], **kw))

    print("\n===== C. FAINT TARGET regime (flux=500 -> low R_self, the q3 analog) =====")
    for d in [0.5, 0.7, 1.0, 1.5]:
        line(f'faint-tgt pair @ {d:.1f}"', decompose(tgt(500), [nbr(d, 40, 500)], **kw))

    print("\n===== D. FAINT-CROWDING ladder: N faint nbrs (flux=250) ringed at 1.0\" =====")
    print("      (E)-OOD analog: does a SUM over many faint pairs accumulate excess?")
    for N in [1, 2, 4, 8]:
        angs = list(np.linspace(0, 360, N, endpoint=False))
        line(f'{N} faint nbrs @ 1.0"', decompose(tgt(1000), [nbr(1.0, ang, 250) for ang in angs], **kw))

    print("\n===== E. CLOSE faint-crowding: N very-faint nbrs (flux=120) ringed at 0.8\" =====")
    for N in [1, 2, 4, 8]:
        angs = list(np.linspace(0, 360, N, endpoint=False))
        line(f'{N} vfaint nbrs @ 0.8"', decompose(tgt(1000), [nbr(0.8, ang, 120) for ang in angs], **kw))

    print("\nTOY_CLOSE_PAIR_SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
