"""Does the per-pair blend response dilute as R_blend_j ∝ 1/T (T = total scene trace)?

If yes, the emulator's crowd-blindness is fixable by a CLOSED-FORM crowding correction
computable from g=0 (fluxes+positions+sizes) -- no coherent renders needed.

Fix a 'probe' neighbour at 1.2"; add M other equal neighbours; measure R_blend_probe
(shear ONLY the probe) and the scene's adaptive-moment size sigma^2 (~ T/2, from the noiseless
blend). Check R_blend_probe * sigma^2 ~ const (i.e. R_blend ∝ 1/T).
"""
import sys
import numpy as np
import galsim

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from scripts.toy_blend_decompose import render, resp  # noqa
from scripts.toy_blend_linearity import PIX  # noqa


def scene_sigma2(gals):
    """Adaptive-moment sigma^2 (arcsec^2) of the NOISELESS unsheared blend (~ trace/2)."""
    img = render([dict(g, shear=False) for g in gals], 0.0, 0.0)
    hsm = galsim.hsm.FindAdaptiveMom(galsim.Image(np.ascontiguousarray(img), scale=PIX), strict=False)
    return (hsm.moments_sigma * PIX) ** 2 if hsm.error_message == "" else np.nan


def main():
    flux, sky, g, nreal = 1000.0, 10.0, 0.05, 250
    tgt = dict(hlr=0.4, flux=flux, e1=0.0, e2=0.0, x=0.0, y=0.0)
    probe = dict(hlr=0.4, flux=flux, e1=0.0, e2=0.0, x=1.2, y=0.0)   # fixed probe on +x

    def others(M):
        # M equal neighbours spread on a 1.5" ring, avoiding the +x probe direction
        angs = np.linspace(40, 320, M) if M else []
        return [dict(hlr=0.4, flux=flux, e1=0.0, e2=0.0,
                     x=1.5 * np.cos(np.deg2rad(a)), y=1.5 * np.sin(np.deg2rad(a))) for a in angs]

    print(f"[regime] flux={flux} sky={sky} S/N~{flux/sky/np.sqrt(20):.0f}, probe@1.2\" on +x axis")
    print(f"{'M others':>9} {'sigma^2':>9} {'R_blend_probe':>14} {'+/-':>7} {'R_bl*sig^2':>11}")
    for M in (0, 1, 2, 4, 6):
        scene = [tgt, probe] + others(M)
        s2 = scene_sigma2(scene)
        # shear ONLY the probe (index 1)
        flags = [dict(tgt, shear=False), dict(probe, shear=True)] + [dict(o, shear=False) for o in others(M)]
        Rb, eb = resp(flags, g, sky, nreal)
        print(f"{M:>9} {s2:>9.4f} {Rb:>14.4f} {eb:>7.4f} {Rb*s2:>11.4f}")


if __name__ == "__main__":
    main()
