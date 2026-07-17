"""Model-on-toy CALIBRATION: does the flow's predicted R_flow match the toy's measured self-
response R_self for an ISOLATED galaxy of matched truth + sim photometry?

If yes -> toy and flow training-domain agree, and the model-on-toy localization of the gold +6.5%
is trustworthy. If no -> a toy-vs-sim domain gap to resolve first.

Photometry matched to the sim: flux = 10^(-0.4(mag - zero_mag=30)), per-pixel noise = pixel_rms.
"""
import sys
import numpy as np
import pandas as pd
import galsim

SBSI = "/home/z/Zekang.Zhang/SBSI"
sys.path.insert(0, SBSI)
sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from sbs_shear.measurement_model import load_measurement_model  # noqa
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa
from scripts.response_ratio_diagnostic import model_mean_proj  # noqa

PIX, PSF_FWHM, BETA, ZP, RMS = 0.2, 0.73, 2.224, 30.0, 0.312
RK = dict(pixel_rms=RMS, pixel_size=PIX, zero_mag=ZP, psf_fwhm=PSF_FWHM, moffat_beta=BETA)
_PSF = galsim.Moffat(beta=BETA, fwhm=PSF_FWHM)
_PSF_IM = _PSF.drawImage(nx=48, ny=48, scale=PIX).array
MODEL = "/home/z/Zekang.Zhang/SBSI/models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt"


def flow_R(bundle, Re, mag, n, q, pa, g=0.05):
    df = pd.DataFrame([{
        "Re_input_p": Re, "r_input_p": mag, "sersic_n_input_p": n, "redshift_input_p": 0.7,
        "axis_ratio_input_p": q, "position_angle_input_p": pa,
        "nbr_flux_near": 0.0, "nbr_flux_far": 0.0,
        "gamma1_input_p": 0.0, "gamma2_input_p": 0.0, "distance": 1.0,
        "Re_input_s": Re, "r_input_s": mag,  # dummy secondary cols (rescale needs them; crowd_flux ignores)
    }])
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i; df["e2_input_rot0_p"] = e2i
    intr = (e1i.copy(), e2i.copy())
    gh1, gh2 = np.ones(1), np.zeros(1)
    mp, _ = model_mean_proj(bundle, df, +g, gh1, gh2, intr, RK, 256, 65536)
    mm, _ = model_mean_proj(bundle, df, -g, gh1, gh2, intr, RK, 256, 65536)
    return (mp - mm) / (2 * g)


def measure(im, seed):
    import ngmix
    from blendemu import shape as bshape
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


def toy_R(Re, mag, n, q, pa, g=0.05, nreal=400, stamp=64):
    flux = 10 ** (-0.4 * (mag - ZP))
    e1i, e2i = ellipticity_from_axis_ratio_angle(np.array([q]), np.array([pa]))
    e1i, e2i = float(e1i[0]), float(e2i[0])

    def render(gg):
        obj = galsim.Sersic(n=n, half_light_radius=Re, flux=flux, trunc=Re * 10)
        if e1i or e2i:
            obj = obj.shear(g1=e1i, g2=e2i)
        obj = obj.shear(g1=gg, g2=0.0)
        img = galsim.ImageF(stamp, stamp, scale=PIX)
        galsim.Convolve([obj, _PSF]).drawImage(image=img, add_to_image=True)
        return img.array
    cp, cm = render(+g), render(-g)
    v = []
    for rr in range(nreal):
        nz = np.random.RandomState(rr).normal(0, RMS, cp.shape)
        ep, em = measure(cp + nz, rr), measure(cm + nz, rr)
        v.append((ep[0] - em[0]) / (2 * g))
    v = np.array(v)
    return v.mean(), v.std() / np.sqrt(len(v))


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_measurement_model(MODEL, device=dev)
    print(f"[calib] isolated galaxy, matched sim photometry (ZP={ZP}, rms={RMS}); Re=0.4 n=1 q=0.7 pa=30")
    print(f"{'mag':>5} {'R_flow(model)':>14} {'R_self(toy)':>14} {'+/-':>7} {'flow/toy':>9}")
    for mag in [23.0, 24.0, 25.0, 26.0]:
        Rf = flow_R(bundle, 0.4, mag, 1.0, 0.7, 30.0)
        Rt, et = toy_R(0.4, mag, 1.0, 0.7, 30.0)
        print(f"{mag:>5.1f} {Rf:>14.4f} {Rt:>14.4f} {et:>7.4f} {Rf/Rt if Rt else float('nan'):>9.3f}", flush=True)


if __name__ == "__main__":
    main()
