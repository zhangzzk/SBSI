"""Postage-stamp toy: is the coherent multi-neighbour blend response LINEAR
(= sum of per-pair marginals) or SUPER-ADDITIVE?

Inject target + neighbours into a GalSim stamp, coherently shear (shapes + optionally
positions), measure the TARGET's ngmix shear response via antithetic +/-g.

  R_self, R_pair_j, marginal_j = R_pair_j - R_self, R_full, R_linear = R_self + sum marginals,
  excess = R_full - R_linear   (>0 => super-additive collective blend; ~0 => emulator's
  linear-superposition assumption holds and the gold +6.5% must come from elsewhere).

Uses Gaussian galaxies (match the ngmix 'gauss' model -> no profile-mismatch railing) and a
SEEDED rng (deterministic). Noiseless by default (clean response); optional noise+SNC+average.
"""
import argparse
import sys
import numpy as np
import galsim
import ngmix

sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from blendemu import shape as bshape  # noqa

PIX = 0.2
PSF_FWHM = 0.73
BETA = 2.224
_PSF = galsim.Moffat(beta=BETA, fwhm=PSF_FWHM)
_PSF_IM = _PSF.drawImage(nx=48, ny=48, scale=PIX).array


def measure(im, seed=42):
    """ngmix PSF-corrected shape, seeded for determinism (replicates blendemu.shape)."""
    im = np.ascontiguousarray(im)
    rng = np.random.RandomState(seed)
    hsm = galsim.hsm.FindAdaptiveMom(galsim.Image(im, scale=PIX), strict=False)
    Tg = 2 * (hsm.moments_sigma * PIX) ** 2 if hsm.error_message == "" else 1.0
    prior = bshape._get_prior(rng, PIX)
    fitter = ngmix.fitting.Fitter(model="gauss", prior=prior)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tg, prior=prior)
    pr = ngmix.runners.PSFRunner(fitter=ngmix.em.EMFitter(),
                                 guesser=ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1), ntry=2)
    r = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=2)
    return ngmix.bootstrap.Bootstrapper(runner=r, psf_runner=pr).go(
        bshape.make_obs(im, _PSF_IM, PIX))["g"]


def render(gals, g1, g2, stamp=96, shear_pos=True):
    img = galsim.ImageF(stamp, stamp, scale=PIX)
    for gg in gals:
        obj = galsim.Gaussian(half_light_radius=gg["hlr"], flux=gg["flux"])
        if gg.get("e1", 0.0) or gg.get("e2", 0.0):
            obj = obj.shear(g1=gg.get("e1", 0.0), g2=gg.get("e2", 0.0))
        obj = obj.shear(g1=g1, g2=g2)                     # coherent shear on the shape
        x, y = gg["x"], gg["y"]
        if shear_pos:                                      # coherent shear distorts positions
            x, y = x * (1 + g1) + y * g2, y * (1 - g1) + x * g2
        galsim.Convolve([obj, _PSF]).drawImage(image=img, add_to_image=True, offset=(x / PIX, y / PIX))
    return img.array


def response(gals, g=0.02, stamp=96, shear_pos=True, sky=0.0, nreal=1):
    """Tangential response R=(e1(+g)-e1(-g))/(2g). sky>0 adds SNC noise averaged over nreal."""
    cp = render(gals, +g, 0.0, stamp, shear_pos)
    cm = render(gals, -g, 0.0, stamp, shear_pos)
    if sky <= 0:
        ep, em = measure(cp), measure(cm)
        return (ep[0] - em[0]) / (2 * g), (ep[1] - em[1]) / (2 * g), 0.0
    vals = []
    for rr in range(nreal):
        nz = np.random.RandomState(rr).normal(0, sky, cp.shape)
        ep, em = measure(cp + nz, rr), measure(cm + nz, rr)
        vals.append((ep[0] - em[0]) / (2 * g))
    v = np.array(vals)
    return v.mean(), 0.0, v.std() / np.sqrt(len(v))


def run_scenario(name, target, neighbours, g=0.02, shear_pos=True, sky=0.0, nreal=1):
    R_self, _, e0 = response([target], g=g, shear_pos=shear_pos, sky=sky, nreal=nreal)
    margs, emarg = [], []
    for nb in neighbours:
        Rp, _, ep = response([target, nb], g=g, shear_pos=shear_pos, sky=sky, nreal=nreal)
        margs.append(Rp - R_self); emarg.append(np.hypot(ep, e0))
    R_full, _, ef = response([target] + neighbours, g=g, shear_pos=shear_pos, sky=sky, nreal=nreal)
    R_lin = R_self + sum(margs)
    excess = R_full - R_lin
    err_lin = np.sqrt(e0 ** 2 + sum(e ** 2 for e in emarg))
    err_ex = np.hypot(ef, err_lin)
    print(f"\n### {name}  (g={g}, shear_pos={shear_pos}, sky={sky}, N_nbr={len(neighbours)}) ###")
    print(f"  R_self            = {R_self:+.4f}")
    print(f"  marginals         = [{', '.join(f'{m:+.4f}' for m in margs)}]")
    print(f"  sum(marginals)    = {sum(margs):+.4f}")
    print(f"  R_linear          = {R_lin:+.4f}")
    print(f"  R_full (coherent) = {R_full:+.4f}" + (f" +/- {ef:.4f}" if sky > 0 else ""))
    rel = excess / R_full if R_full else float("nan")
    sig = f"  ({excess/err_ex:+.1f} sigma)" if sky > 0 else ""
    print(f"  EXCESS            = {excess:+.4f}   ({rel:+.1%} of R_full){sig}   [>0 => super-additive]")
    return dict(R_self=R_self, R_full=R_full, R_lin=R_lin, excess=excess, rel=rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=0.02)
    ap.add_argument("--sky", type=float, default=0.0)
    ap.add_argument("--nreal", type=int, default=1)
    ap.add_argument("--no-shear-pos", action="store_true")
    args = ap.parse_args()
    sp = not args.no_shear_pos
    kw = dict(g=args.g, shear_pos=sp, sky=args.sky, nreal=args.nreal)

    def tgt():
        return dict(hlr=0.4, flux=1.0e4, e1=0.0, e2=0.0, x=0.0, y=0.0)

    def nbr(dist, ang, flux=1.0e4, hlr=0.4):
        a = np.deg2rad(ang)
        return dict(hlr=hlr, flux=flux, e1=0.0, e2=0.0, x=dist * np.cos(a), y=dist * np.sin(a))

    Rs, xs, _ = response([tgt()], **kw)
    print(f"[sanity] isolated target R_self={Rs:+.4f} cross={xs:+.4f}")

    run_scenario("4 equal nbrs @1.5\" ring", tgt(), [nbr(1.5, a) for a in (0, 90, 180, 270)], **kw)
    run_scenario("4 equal nbrs @1.0\" ring", tgt(), [nbr(1.0, a) for a in (0, 90, 180, 270)], **kw)
    run_scenario("2 bright nbrs @1.0\" on g1 axis", tgt(),
                 [nbr(1.0, 0, flux=3e4), nbr(1.0, 180, flux=3e4)], **kw)
    run_scenario("6 nbrs crowded 0.8-1.5\"", tgt(),
                 [nbr(0.8, 30), nbr(1.0, 100), nbr(1.2, 150), nbr(0.9, 210),
                  nbr(1.3, 280), nbr(1.5, 340)], **kw)


if __name__ == "__main__":
    main()
