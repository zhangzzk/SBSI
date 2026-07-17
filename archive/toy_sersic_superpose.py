"""Does superposition still hold for SERSIC galaxies fit by single-Gaussian ngmix (the REAL sim
setup)?  My Gaussian-galaxy toy showed exact superposition, but the sim uses Sersic profiles and
ngmix fits ONE Gaussian -> a profile-mismatch nonlinearity that could break superposition and
produce the gold +6.5%. Test the clean decomposition (shear each source separately) with Sersic.
"""
import argparse
import sys
import numpy as np
import galsim

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from scripts.toy_blend_linearity import measure, _PSF, PIX  # noqa


def render(gals, g1, g2, stamp=96):
    img = galsim.ImageF(stamp, stamp, scale=PIX)
    for gg in gals:
        obj = galsim.Sersic(n=gg.get("n", 1.0), half_light_radius=gg["hlr"], flux=gg["flux"],
                            trunc=gg["hlr"] * 10)
        if gg.get("e1", 0.0) or gg.get("e2", 0.0):
            obj = obj.shear(g1=gg.get("e1", 0.0), g2=gg.get("e2", 0.0))
        x, y = gg["x"], gg["y"]
        if gg.get("shear", True):
            obj = obj.shear(g1=g1, g2=g2)
            x, y = x * (1 + g1) + y * g2, y * (1 - g1) + x * g2
        galsim.Convolve([obj, _PSF]).drawImage(image=img, add_to_image=True, offset=(x / PIX, y / PIX))
    return img.array


def resp(gals, g, sky, nreal, stamp=96):
    cp, cm = render(gals, +g, 0.0, stamp), render(gals, -g, 0.0, stamp)
    v = []
    for rr in range(nreal):
        nz = np.random.RandomState(rr).normal(0, sky, cp.shape)
        ep, em = measure(cp + nz, rr), measure(cm + nz, rr)
        v.append((ep[0] - em[0]) / (2 * g))
    v = np.array(v)
    return v.mean(), v.std() / np.sqrt(len(v))


def decompose(name, target, neighbours, g, sky, nreal):
    def flag(gal, s):
        d = dict(gal); d["shear"] = s; return d
    Rs, es = resp([flag(target, True)] + [flag(n, False) for n in neighbours], g, sky, nreal)
    blends, ebl = [], []
    for j in range(len(neighbours)):
        rb, eb = resp([flag(target, False)] + [flag(n, i == j) for i, n in enumerate(neighbours)],
                      g, sky, nreal)
        blends.append(rb); ebl.append(eb)
    Rf, ef = resp([flag(target, True)] + [flag(n, True) for n in neighbours], g, sky, nreal)
    Rlin = Rs + sum(blends); ex = Rf - Rlin
    el = np.sqrt(es ** 2 + sum(e ** 2 for e in ebl)); eex = np.hypot(ef, el)
    print(f"\n### {name} ###", flush=True)
    print(f"  R_self={Rs:+.3f}  Σblend={sum(blends):+.3f}  R_linear={Rlin:+.3f}±{el:.3f}  "
          f"R_full={Rf:+.3f}±{ef:.3f}")
    print(f"  EXCESS={ex:+.3f}±{eex:.3f}  ({ex/Rf:+.0%} of R_full, {ex/eex:+.1f}σ)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, default=1.0, help="Sersic index of all galaxies")
    ap.add_argument("--flux", type=float, default=1200.0)
    ap.add_argument("--sky", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--nreal", type=int, default=250)
    args = ap.parse_args()

    def tgt():
        return dict(n=args.n, hlr=0.4, flux=args.flux, e1=0.0, e2=0.0, x=0.0, y=0.0)

    def nbr(d, ang):
        a = np.deg2rad(ang)
        return dict(n=args.n, hlr=0.4, flux=args.flux, e1=0.0, e2=0.0,
                    x=d * np.cos(a), y=d * np.sin(a))

    print(f"[Sersic n={args.n}] flux={args.flux} sky={args.sky} S/N~{args.flux/args.sky/np.sqrt(20):.0f}")
    decompose("1 nbr @1.2\"", tgt(), [nbr(1.2, 40)], args.g, args.sky, args.nreal)
    decompose("2 nbrs @1.2\"", tgt(), [nbr(1.2, 40), nbr(1.2, 200)], args.g, args.sky, args.nreal)
    decompose("4 nbrs @1.2\"", tgt(), [nbr(1.2, a) for a in (20, 110, 200, 290)], args.g, args.sky, args.nreal)


if __name__ == "__main__":
    main()
