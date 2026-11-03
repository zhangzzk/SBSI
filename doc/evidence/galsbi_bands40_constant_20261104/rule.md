# Does the band-guard flow change carry through to the end-to-end constant-shear result?

Fixed 2026-11-04, before any output of this study exists.

Owner: "we need build a better guard so the model recovers 0.3% at cuts around mag < 25 with radius > 3".
The band-guard flow (`galsbi_guard_strength_20261103/bands40`) failed its flow-alone gate (target grid RMS 0.78 >
0.5, max |m_eq| 1.12 > 1.0) but cut the held-out flow-alone RMS from 5.12 to 1.52 (radius >= 3.75 px & < 25.8:
-5.36 -> -2.14). The flow-alone check and the end-to-end evaluation disagree for GalSBI-dev (radius >= 3.75 px &
< 25.0: flow alone -4.83 +- 0.41, end to end +0.97 +- 0.18), so whether flow-alone changes transfer is unknown.
This is a diagnostic, not an acceptance test: nothing is trained, tuned or selected on these constant-shear sims.

Runs (identical to `galsbi_dev_constant_20261103/rule.md` except as stated):
- Cases: chunks 0, 4, 8, 12 = cases 1000-1009, 1040-1049, 1080-1089, 1120-1129 (40 of 140; fixed by chunk number).
- A: this root, preset `preset.json` = GalSBI-dev with the flow replaced by bands40 (sha256 1e7ada5b...).
- B: `galsbi_dev_constant_grid_20261104`, preset = GalSBI-dev unchanged, same 40 cases.
- Same seeds (7301 + 10000019 case + 1000003 block), 64 antithetic draws, per-leg measured cuts, own-leg
  normalization; `parent/` and `c8/` are symlinks to the first study (C8 and inputs do not depend on the flow).
- Cuts: the 8 existing rows plus MAG_AUTO < 25 with radius > 2.8, 2.9, 3.1, 3.2 px and >= 3.75 px, and radius
  > 3 px with MAG_AUTO < 24.6 and < 25.4 (15 rows; `scripts/evaluate_galsbi_constant.py` CUTS).
- Check: B rows 0-2 must equal `galsbi_dev_constant_20261103/sums` and B row 3 must equal
  `galsbi_dev_constant_mag25_20261103/sums` row 1 (relative difference < 1e-10) on all 40 cases.

Reported (C8 weights; true flags as a side column): m for A and B on the 40 cases, and the paired change A - B,
each with a case bootstrap over the 40 cases (10,000 resamples, seed 20260925).

Reading:
- Transfer: compare the paired end-to-end change A - B with the flow-alone change bands40 - steps4 from
  `check_bands.json` on each cut. If they agree within 2 SE on radius >= 3.75 px & < 25.8 (flow alone +3.20), flow-alone
  changes carry through and the remaining end-to-end gap comes from parts the flow check does not see
  (emulator, C8, population). If the end-to-end change is < 1/3 of the flow-alone change there, the flow-alone
  check is not a usable proxy for the end-to-end target and guard tuning on it alone cannot reach 0.3%.
- Target: report A at MAG_AUTO < 25 & radius > 3 px and its neighbours; |m| < 0.3% is not expected.
Limitations: 40 cases, g1 only, one flow seed, errors exclude training and finite-draw noise.
