"""Does the coherent response = R_self + sum(per-pair blend)?  (the emulator's exact assumption)

Clean decomposition matching the gold's R_sim = R_flow + sum R_blend, by choosing WHICH source
is sheared (per-galaxy flag), with all galaxies always PRESENT (consistent blend baseline):

  R_self    : shear only the TARGET (neighbours present, unsheared)      ~ R_flow
  R_blend_j : shear only NEIGHBOUR j (target + others unsheared)         ~ one emulator per-pair
  R_full    : shear EVERYTHING coherently                                ~ R_sim
  R_linear  = R_self + sum_j R_blend_j                                   ~ R_flow + R_blend (emulator)
  excess    = R_full - R_linear   (>0 emulator UNDER-counts like the gold +6.5%; <0 it OVER-counts)

Faint regime (S/N~15-22) reproduces the gold: R_self~0.3, positive blend. Seeded+SNC+averaged.
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from scripts.toy_blend_linearity import measure, _PSF, PIX  # noqa
import galsim


def render(gals, g1, g2, stamp=96, shear_pos=True):
    img = galsim.ImageF(stamp, stamp, scale=PIX)
    for gg in gals:
        obj = galsim.Gaussian(half_light_radius=gg["hlr"], flux=gg["flux"])
        if gg.get("e1", 0.0) or gg.get("e2", 0.0):
            obj = obj.shear(g1=gg.get("e1", 0.0), g2=gg.get("e2", 0.0))
        x, y = gg["x"], gg["y"]
        if gg.get("shear", True):                      # only sheared sources feel (g1,g2)
            obj = obj.shear(g1=g1, g2=g2)
            if shear_pos:
                x, y = x * (1 + g1) + y * g2, y * (1 - g1) + x * g2
        galsim.Convolve([obj, _PSF]).drawImage(image=img, add_to_image=True, offset=(x / PIX, y / PIX))
    return img.array


def resp(gals, g, sky, nreal, stamp=96):
    cp = render(gals, +g, 0.0, stamp)
    cm = render(gals, -g, 0.0, stamp)
    v = []
    for rr in range(nreal):
        nz = np.random.RandomState(rr).normal(0, sky, cp.shape)
        ep, em = measure(cp + nz, rr), measure(cm + nz, rr)
        v.append((ep[0] - em[0]) / (2 * g))
    v = np.array(v)
    return v.mean(), v.std() / np.sqrt(len(v))


def decompose(name, target, neighbours, g=0.05, sky=10.0, nreal=200):
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
    print(f"\n### {name}  (g={g}, sky={sky}, nreal={nreal}, N_nbr={len(neighbours)}) ###", flush=True)
    print(f"  R_self  (R_flow)         = {Rs:+.3f} +/- {es:.3f}")
    print(f"  blends  (per-pair)       = [{', '.join(f'{b:+.3f}' for b in blends)}]  sum={sum(blends):+.3f}")
    print(f"  R_linear (R_flow+Rblend) = {Rlin:+.3f} +/- {el:.3f}")
    print(f"  R_full   (R_sim)         = {Rf:+.3f} +/- {ef:.3f}")
    print(f"  EXCESS = R_full-R_linear = {ex:+.3f} +/- {eex:.3f}  ({ex/Rf:+.0%} of R_full, {ex/eex:+.1f}σ)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux", type=float, default=1000.0)
    ap.add_argument("--sky", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--nreal", type=int, default=200)
    args = ap.parse_args()

    def tgt():
        return dict(hlr=0.4, flux=args.flux, e1=0.0, e2=0.0, x=0.0, y=0.0)

    def nbr(d, ang, flux=None):
        a = np.deg2rad(ang)
        return dict(hlr=0.4, flux=flux or args.flux, e1=0.0, e2=0.0, x=d * np.cos(a), y=d * np.sin(a))

    print(f"[regime] flux={args.flux} sky={args.sky} -> S/N~{args.flux/args.sky/np.sqrt(20):.0f}")
    decompose("1 nbr @1.2\"", tgt(), [nbr(1.2, 40)], g=args.g, sky=args.sky, nreal=args.nreal)
    decompose("2 nbrs @1.2\"", tgt(), [nbr(1.2, 40), nbr(1.2, 200)], g=args.g, sky=args.sky, nreal=args.nreal)
    decompose("4 nbrs @1.2\"", tgt(), [nbr(1.2, a) for a in (20, 110, 200, 290)],
              g=args.g, sky=args.sky, nreal=args.nreal)
    decompose("4 nbrs mixed dist 1-2.5\"", tgt(),
              [nbr(1.0, 20), nbr(1.5, 110), nbr(2.0, 200), nbr(2.5, 290)],
              g=args.g, sky=args.sky, nreal=args.nreal)


if __name__ == "__main__":
    main()
