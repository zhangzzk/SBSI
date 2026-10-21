# Observation 0 phase-corrected score localization

## Fixed question and boundary

The completed phase-control diagnostic found observation 0 corrected g1 score
means 0.404688 ± 0.095263 (v6v100) and 1.069976 ± 0.111169 (v7), a descriptive
4.55 combined-MCSE discrepancy. This read-only follow-up asks whether saved
morphology/scene variation, a few score tails, or a few populations/initial
ancestors account for the difference. It does not fit any coefficient, alter
a target, launch sampling/GPU work, read heldout populations, or change gates.
Only dev_v5_0..7, dev_v6v100_0..7, dev_v7_0..7 and their frozen phase-v2 arrays
are inputs. The new higher-effort fitted-q0 arm is outside this analysis.

## Predeclared accounting

Before inspecting numerical particle arrays, fix observation 0 and retain
all final particles. Within each complete run use uniform weights. Across
runs use original normalizer weights Zhat_r/sum(Zhat), including the proper
Zhat_r/N_r weight for each pooled particle. The complete score mean and MCSE
use independent whole populations. Leaving one population out renormalizes
remaining original Zhat weights; it is descriptive sensitivity, not a set
of independent estimates or another uncertainty calculation.

Read native m_r, log(circularized radius in arcsec), Sersic n, distortion a,
phase and native R_kpc directly from saved coordinates/scenes, validating
agreement with saved summaries. Include redshift, intrinsic e1/e2, the three
log1p crowding summaries, response b, neighbour count, neighbour counts within
0.5/1/2 arcsec, and log1p of total/maximum neighbour-to-target flux ratio.
All neighbours in the saved environment aperture contribute. Phase uses
circular mean/resultant and sin/cos correlations, not linear angle Pearson
correlation. Report exact empirical weighted covariance/correlation with
corrected g1 per population and for the pooled weighted distribution; these
are descriptive products, not effective independent-sample estimates.

Distribution quantiles are fixed at [1%, 5%, 25%, 50%, 75%, 95%, 99%]. Fixed
absolute-score upper-tail masses are 1% and 5%. Sort decreasing |corrected g1|
with stable ties, allow a fractional boundary-particle weight to attain the
exact requested probability mass, and account for signed, positive, negative,
absolute and squared-score contributions. The complement remains part of
the estimate. No particles are dropped. Also report tail mass by population,
and each population's five largest initial-ancestor groups by absolute-score
contribution. Initial ancestry describes genealogy, not post-mutation ESS.

Fixed bin boundaries (first/last infinities included where needed):

- m_r: 18, 20, 22, 24, 26, 28, 30.
- Circularized radius: 0.01, 0.03, 0.1, 0.3, 1, 3, 10 arcsec, represented in log radius.
- n: 1, 2, 3, 4, 6, 8; a: 0, .2, .4, .6, .8, 1.
- Phase: 12 equal bins on [0, 2pi).
- Each log1p crowding quantity: 0, .1, .3, 1, 3, 10, 30, 100, infinity.
- Response b: -4, -2, -1, -.5, 0, .5, 1, 2, 4.
- Neighbour counts: 0, 1–4, 5–9, 10–19, 20–49, 50–99, 100+.

Each bin reports probability mass, conditional mean score, and its signed
contribution to the full mean. Between-group differences use exact midpoint
accounting on mutually occupied bins: sum((p-q)*(mu_p+mu_q)/2) for composition
and sum((p+q)*(mu_p-mu_q)/2) within bins. Unmatched-bin score contributions
remain explicit; no missing conditional mean is imputed. This is additive
descriptive accounting, not a fitted or causal explanation. The fixed bins
may conceal variation within bins.

## Validation and provenance

`scripts/localize_particle_phase_score.py` validates original receipts,
particle/scene hashes, successful full v2 diagnostic receipts, unique seeds,
identical diagnostic targets and the frozen coefficient hash. It exactly
replays corrected scores from the fixed coefficient and saved raw score/T.
It refuses incomplete/extinct or altered inputs and existing output paths.
Focused tests cover unequal population weights/counts, fractional tail mass,
signed cancellation, ancestry accounting, fixed-bin decomposition with
unmatched support, circular wrap, and native/CSR row alignment. Full CPU
execution will use a frozen source manifest, input hashes and a separate
pre-execution receipt. No new model evaluation is involved.

CPU Slurm 17420920: 9 focused tests passed in 0.97 seconds.
