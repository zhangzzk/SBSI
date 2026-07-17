"""Is the SINGLE-PAIR blend response linear in shear? (user's selection-vs-nonlinearity discriminator)

Toy: one target + one neighbour, ALWAYS present, NO detection/cross-match gate. Shear ONLY the neighbour
(blend config), measure the target's response R_blend = delta_et/g via antithetic +/-g, at several g and
separations (incl. the ~1" dip region). Gaussian galaxies + ngmix 'gauss' (no profile railing), noiseless.
  R_blend(g) FLAT across g  => per-object response is LINEAR => the full-sim g-dependence is a SELECTION
     effect (detection/cross-match sample changes with g), not intrinsic nonlinearity.
  R_blend(g) VARIES with g  => genuine single-response shear nonlinearity.
"""
import sys, numpy as np
sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/scripts")
from toy_blend_decompose import resp   # render (per-gal 'shear' flag) + ngmix measure, returns (R, err)

FLUX = 1000.0; HLR = 0.4; ANG = 40.0   # working-toy params (resolved galaxy vs 0.73" PSF)

def pair(sep):
    a = np.deg2rad(ANG)
    return [dict(hlr=HLR, flux=FLUX, x=0.0, y=0.0, shear=False),                          # target: unsheared
            dict(hlr=HLR, flux=FLUX, x=sep * np.cos(a), y=sep * np.sin(a), shear=True)]    # neighbour: sheared

GS = [0.02, 0.03, 0.05, 0.1, 0.2]
SEPS = [1.2, 2.0, 3.0]
print(f"single-pair BLEND response R=delta_et/g (flux={FLUX}, hlr={HLR}, nbr@{ANG}deg, noiseless)")
print(f"{'sep':>6} " + " ".join(f"g={g:<5}" for g in GS) + "   spread(max/min)")
for sep in SEPS:
    row = []
    for g in GS:
        R, _ = resp(pair(sep), g=g, sky=0.0, nreal=1)
        row.append(R)
    row = np.array(row)
    rng = row.max() / row.min() if row.min() != 0 and np.sign(row.max()) == np.sign(row.min()) else np.nan
    print(f"{sep:>6.1f} " + " ".join(f"{r:>7.4f}" for r in row) + f"   {rng if np.isfinite(rng) else float('nan'):.2f}")

# SANITY: isolated target self-response must be ~+1 across g (else measurement is broken).
print("\nSANITY self-response R (isolated sheared target) vs g -- must be ~+1:")
tgt = [dict(hlr=HLR, flux=FLUX, x=0.0, y=0.0, shear=True)]
print("  " + "  ".join(f"g={g}:{resp(tgt, g=g, sky=0.0, nreal=1)[0]:.4f}" for g in GS))
print("TOY_SHEAR_SCAN_DONE")
