# GalSBI-dev on the GalSBI constant-shear family at a matched-depth cut (MAG_AUTO < 25.0)

Fixed 2026-11-03, before any output of this study exists.

Owner: "let's cut at 25 then" (after learning the GalSBI sims are LSST-i, pixel rms 0.616 vs FS2 LSST-r 0.312).
Measured SExtractor S/N at MAG_AUTO 25.6-25.8 is 6.8 (GalSBI case 1000) vs 9.8 (FS2 case 40); FS2's S/N ~10 at its
25.8 cut sits at MAG_AUTO ~24.9-25.0 in GalSBI. The cut was chosen after seeing the 25.8 result, so this is a
diagnostic of the depth explanation, not a new headline or acceptance test. Test only: nothing is trained or tuned.

Identical to `galsbi_dev_constant_20261103/rule.md` except the MAG_AUTO limit of both cut rows (25.8 -> 25.0):
- cases 1000-1139, g1 = +-0.02, 64 antithetic draws, seed 7301 + 10000019 case + 1000003 block, preset GalSBI-dev
  (hashes checked), per-leg measured cuts, own-leg normalization, case bootstrap 10,000 seed 20260925.
- `parent/` and `c8/` are symlinks to the first study (they do not depend on the cut).
- Code: `scripts/evaluate_galsbi_constant.py` with EVAL_OUT=this root, EVAL_MAG_LIMIT=25.0; the frozen
  `output_cuts` reads `FLUX_MINIMUM`, which the script sets to 10^(-0.4 (25.0 - 30)) for both sides.

Reported: m (C8 and true flags) for uncut, radius > 0.6" & MAG_AUTO < 25.0 (primary), radius >= 0.75" & < 25.0,
flow only, and the true-magnitude bins; plus each bin's contribution to the total.

Reading, primary row (C8, > 0.6" & < 25.0), m +- case SE:
- |m| < 2 SE: the 25.8 failure is consistent with the cut sitting too close to the LSST-i detection limit.
- m > 2 SE but below +1.37% (half of the 25.8 value +2.74%): depth explains part of it.
- otherwise: depth is not the main explanation.
Limitations carried over: g1 only, one seed per model, errors exclude training and finite-draw noise.
