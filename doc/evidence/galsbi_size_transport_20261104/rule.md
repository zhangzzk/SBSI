# Why GalSBI-dev's flow gets the shear response wrong near the 3 px size cut (fixed 2026-11-04, before output)

Owner request 2026-11-04: "we need to understand why current flow fails on shape response in the dimension of
size, particularly around 3px". Diagnostic only: no training, no constant-shear data, no model change.

## What is already known (not repeated)

- Flow-alone response map (WORKLOG 2026-11-03 "Flow response map by measured size"): a size "wave" -- too much
  response at 2.6-3.0 px, too little at 3.1-3.5 px; the simulation's per-galaxy response at 2.6-2.7 px is -0.50
  (galaxies crossing the cut), the flow's -0.22 / -0.02.
- Saved map counts (checked 2026-11-04, no new run): the flow's zero-leg count per 0.1 px bin at MAG_AUTO < 25 is
  within 1-3% of the simulation at 2.3-3.9 px, so the number of galaxies AT the cut is right.
- September FS2 flows (archive/research-2026-09-14/doc/FLOW_PAIRED_TRANSPORT.md, FLOW_SELECTION_LOCALIZATION.md):
  near the old 0.75" cut the radius transport coefficient (slope of dln r/|g| vs intrinsic e along the shear) was
  sim +0.50, plain flow -0.18; freezing the flow's residual distribution between legs (mean-only) moved
  radius-only m from +5.1% to -0.1%. Never measured on GalSBI-dev (F8 lineage, circularized size input).

## What is measured (scripts/run_galsbi_guard_strength.py transport)

All matched usable pairs of the 40 validation cases of the GalSBI training family (legs g = 0 and 0.05), flows
nll_only (density only), steps4 (GalSBI-dev, the preset flow), bands40; 16 antithetic CRN draws per pair with
the check/map seeds (614000000001 + block). Shear enters the flow only through the sheared input ellipticity.

1. Cut split: for hard cuts FLUX_RADIUS >= T (T = 2.5..4.0 px step 0.1) AND MAG_AUTO < 25.0 / 25.8 / none, each
   leg by its own measurement: response sum = kept-in-both shape change + entering - exiting, with
   s = e_measured . design (the map/check projection). m-like ratios use the simulation's total.
2. Radius transport: slope of y = dln(FLUX_RADIUS)/|g| against x = intrinsic e . shear direction, per own
   zero-leg (radius 0.1 px x MAG_AUTO band) bin and per true (circularized Re x true mag) bin; also sum of
   (zero-leg measured s) x y, i.e. how the measured ellipticity of a galaxy relates to its size change.
3. Static: slope of zero-leg ln FLUX_RADIUS against |e_in|^2 per true bin (what the flow can learn from the
   population; with shear entering only via e_in, the flow's mean transport equals this gradient).
4. Mean-only ablation of each flow: per pair keep each leg's draw mean in (e1, e2, ln r, ln flux) and give both
   legs the mid-point residual.

Errors: case bootstrap (10,000, seed 20260925) on per-case sums; flows share draws, so flow differences are paired.

## Validation (must pass before reading results)

Per cut, kept + entering - exiting must equal the earlier map's bin sums over the same region (map.npz for
sim / nll_only / steps4, map_bands.npz for bands40), up to threshold ties (relative difference < 1e-6 of the
cut total); otherwise the result is not reported.

## Reading (decided now)

- H1 mean size shift too small: flow transport slope near 2.6-3.4 px and the static ln r vs |e|^2 slope clearly
  below the simulation's, and the mean-only flow keeps the entering/exiting deficit.
- H2 spread change wrong (September pattern): mean-only flow's entering/exiting sums move to the simulation's
  (within errors) while the full flow's do not.
- H3 right size shift, wrong ellipticity of the movers: transport slopes agree but the sum of s0 x y (measured
  ellipticity of galaxies whose size changes) differs.
- None of the above: the error is in the kept-in-both shape change itself; report the shape part by bin.
More than one can hold; the entering/exiting split decides how much of the 3 px error each part carries.

Job: one GPU (cip A40, `jobs/job_galsbi_guard_strength.sh transport`), analysis on CPU. Output transport.npz here.

## Amendment (2026-11-04, after the first run's validation, before reading any result)

Run 17713793 failed the validation for the simulation only (flows reproduced the map to 3e-15; simulation up to
2.8e-3): simulated FLUX_RADIUS is stored rounded to 0.001 px, so ~0.06% of galaxies sit exactly on each
threshold, and the map counts them as passing (bucketize, >=) while the first run used >. The cut is now >= T
(the map convention); the run is repeated with nothing else changed. No result table of the first run was read.
