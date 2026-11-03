# GalSBI-dev on the GalSBI constant-shear family: size and magnitude limits separated, shape/selection split

Fixed 2026-11-03, before any output of this study exists.

Owner: "go ahead and split the size and magnitude cuts", after the matched-depth test (MAG_AUTO < 25.0:
m +2.47% ± 0.14, `galsbi_dev_constant_mag25_20261103`) showed depth is not the main cause and the ≥ 0.75″ row
was much smaller (+0.97 ± 0.18). Diagnostic only: test data, nothing trained or tuned, no headline.

Same as `galsbi_dev_constant_20261103/rule.md` (cases 1000–1139, g1 ±0.02, 64 antithetic draws, seed
7301 + 10000019 case + 1000003 block, preset GalSBI-dev with hash checks, per-leg measured cuts on both sides,
own-leg normalization, case bootstrap 10,000 seed 20260925). `parent/` and `c8/` are symlinks to that study.

Cut rows (measured FLUX_RADIUS in 0.2″ pixels, MAG_AUTO; identical on the simulation and on the model draws):
0 uncut; 1 r > 0.6″ & < 25.8; 2 r ≥ 0.75″ & < 25.8; 3 r > 0.6″ & < 25.0; 4 r > 0.6″ only; 5 r ≥ 0.75″ only;
6 MAG_AUTO < 25.8 only; 7 MAG_AUTO < 25.0 only. Rows 0–2 must reproduce the first study and row 3 the
matched-depth study (same draws); a mismatch beyond rounding (1e-9 relative on the sums) stops the study.

Shape/selection split, true usable flags only (C8 gives no joint probability for both legs): the shape part is
the response of galaxies (simulation, usable and kept in both legs) or draws (model, kept in both legs, weighted
by true usability in both legs) that pass a cut in both legs; the selection part is the rest. Their
contributions 100 (R_sim,part − R_model,part)/R_model add up to m.

Reported: m for every row (C8 and true flags) with true-magnitude bins; the shape and selection parts for
every row. Reading:
- size or magnitude: the row (4 or 6) whose m is closer to the combined row 1 (both within its SE) identifies
  the limit that carries the bias; if neither alone reaches half of row 1, it comes from the combination.
- shape or selection: the part whose contribution exceeds 2 SE and makes up most of m at row 1.
Limitations carried over: g1 only, one seed per model, errors exclude training and finite-draw noise.
