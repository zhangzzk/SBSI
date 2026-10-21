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

## Completed CPU evidence

CPU Slurm **17420950 completed in 5 seconds**, exit 0, with two CPUs and
no GPU. Source commit `1fe3d630b7764b9b62c855891c8426989825bb84`; source
manifest SHA256 `8e85ab6ab993b1cec2718e544ec3b68a011cc5aac928cb75338f584863bde2d2`.
The pre-execution receipt hashes all 145 inputs. Original input hashes,
native/summary alignment, v2 replay status, unique seeds and frozen correction
all passed. Output `phase_localization_20261022.json` has SHA256
`3cdf9e11f2abbbe573bce06ed579e9a91ce7fe1e6b7e8893501e66249e10b3fa`.
The full per-population feature covariance/correlation table is also exported
as `phase_localization_20261022_correlations.csv`, SHA256
`95854b96542197db96d2d4513509631884f0ece5f7df19fc9449b4d8ff87911b`.
All artifacts are under `sampler_search_20261021` in the project data root.

### Whole-population weighting and sensitivity

| Group | Corrected g1 mean ± MCSE | Run-weight ESS / 8 | Leave-one-population mean range | Largest single-particle absolute mean contribution |
|---|---:|---:|---:|---:|
| v5 | 0.478684 ± 0.187010 | 7.5555 | 0.404861–0.612849 | 0.044647 |
| v6v100 | 0.404688 ± 0.095263 | 7.9176 | 0.363128–0.486680 | 0.019921 |
| v7 | 1.069976 ± 0.111169 | 7.9391 | 1.006962–1.116391 | 0.073577 |

The v7−v6 difference is **0.665288 ± 0.146402**, or **4.544 combined reported
MCSEs**. No single omitted run closes it: even the smallest v7 leave-one mean
is above the largest v6 leave-one mean. This remains descriptive finite-eight-
population evidence, not a confidence interval from overlapping leave-one fits.

| Group | Run | log Zhat | Z weight | Corrected g1 | Leave-one shift | Distinct initial ancestors | Genealogy mass ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 0 | 16.790113 | 0.13118 | 0.824990 | -0.052289 | 22 | 11.074 |
| v5 | 1 | 16.677144 | 0.11717 | 1.034916 | -0.073823 | 20 | 8.049 |
| v5 | 2 | 16.674526 | 0.11686 | -0.535202 | +0.134165 | 22 | 7.624 |
| v5 | 3 | 16.625238 | 0.11124 | 0.670349 | -0.023990 | 18 | 6.815 |
| v5 | 4 | 16.691531 | 0.11887 | 0.763770 | -0.038459 | 26 | 11.112 |
| v5 | 5 | 16.786690 | 0.13073 | -0.147099 | +0.094115 | 24 | 6.755 |
| v5 | 6 | 16.287264 | 0.07934 | 1.024713 | -0.047055 | 24 | 11.526 |
| v5 | 7 | 17.184474 | 0.19460 | 0.433278 | +0.010971 | 28 | 11.123 |
| v6v100 | 0 | 16.977191 | 0.13932 | -0.101855 | +0.081992 | 117 | 58.975 |
| v6v100 | 1 | 16.867965 | 0.12490 | 0.417284 | -0.001798 | 118 | 62.274 |
| v6v100 | 2 | 16.856050 | 0.12342 | 0.502197 | -0.013729 | 107 | 48.025 |
| v6v100 | 3 | 16.830751 | 0.12034 | 0.660193 | -0.034953 | 113 | 58.371 |
| v6v100 | 4 | 16.868228 | 0.12493 | 0.409320 | -0.000661 | 108 | 51.396 |
| v6v100 | 5 | 17.037279 | 0.14794 | 0.268366 | +0.023670 | 111 | 51.461 |
| v6v100 | 6 | 16.789708 | 0.11550 | 0.478729 | -0.009668 | 114 | 58.770 |
| v6v100 | 7 | 16.681469 | 0.10365 | 0.764091 | -0.041560 | 112 | 56.527 |
| v7 | 0 | 16.786495 | 0.10938 | 0.694083 | +0.046167 | 39 | 23.849 |
| v7 | 1 | 16.893774 | 0.12177 | 1.322276 | -0.034983 | 38 | 18.930 |
| v7 | 2 | 16.937681 | 0.12724 | 1.393695 | -0.047194 | 35 | 20.140 |
| v7 | 3 | 16.979160 | 0.13263 | 0.856764 | +0.032601 | 39 | 20.302 |
| v7 | 4 | 16.975585 | 0.13215 | 1.483787 | -0.063014 | 37 | 25.620 |
| v7 | 5 | 16.914515 | 0.12432 | 1.172708 | -0.014585 | 36 | 26.903 |
| v7 | 6 | 16.784061 | 0.10912 | 0.691027 | +0.046415 | 41 | 28.568 |
| v7 | 7 | 17.057176 | 0.14339 | 0.870334 | +0.033418 | 37 | 25.187 |

### Tails and ancestry concentration

| Group | Upper absolute-score mass | Signed mean contribution | Positive / negative contribution | Absolute-score share | Squared-score share |
|---|---:|---:|---:|---:|---:|
| v5 | 1% | 0.009811 | 0.223751 / -0.213939 | 7.90% | 28.69% |
| v5 | 5% | 0.143363 | 0.705556 / -0.562193 | 22.88% | 53.27% |
| v6v100 | 1% | 0.118226 | 0.244556 / -0.126331 | 6.87% | 25.53% |
| v6v100 | 5% | 0.137990 | 0.664123 / -0.526134 | 22.04% | 51.08% |
| v7 | 1% | 0.206371 | 0.303334 / -0.096963 | 7.18% | 29.69% |
| v7 | 5% | 0.497150 | 0.873384 / -0.376233 | 22.40% | 54.65% |

The fixed top 1% accounts for **0.088146 (13.25%)** of the v7−v6 difference.
The top 5% accounts for **0.359161 (53.99%)**, leaving **0.306127 (46.01%)**
in the complementary 95%. These are additive retained contributions, not
trimmed estimates. Positive/negative tail cancellation differs appreciably;
the top 5% contains roughly half the squared-score mass in every group.
This supports score-tail sensitivity, but the discrepancy is not confined
to a few extreme particles. The largest single v7 particle contributes only
0.073577 in absolute units to the full weighted mean.

Genealogy is more concentrated in the 256-particle groups than in the
1024-particle v6 group. V7 has 35–41 initial ancestors per run, genealogy
mass ESS 18.93–28.57 and largest ancestry mass 6.6–12.1%; v6 has 107–118,
ESS 48.02–62.27 and largest mass 3.7–5.4%. The largest individual ancestor
accounts for at most 18.3% of absolute score in v7, versus 8.4% in v6.
The five largest ancestor groups carry 31.2–50.4% of absolute score in v7
and 15.7–23.1% in v6 (v5: 57.3–81.2%). These groups have mutated descendants;
they are neither identical particles nor independent replicates. This
identifies concentration but does not isolate one ancestor as the cause.

### Physical and scene distributions

Numbers below are Z-weighted means ± empirical standard deviations; they
are distributions, not uncertainty on the mean. Full fixed quantiles and
per-run distributions are retained in JSON.

| Feature | v5 | v6v100 | v7 |
|---|---:|---:|---:|
| m_r | 22.575917 ± 0.288260 | 22.557000 ± 0.253511 | 22.549829 ± 0.201433 |
| log_radius_arcsec | -0.894616 ± 0.242938 | -0.880139 ± 0.239620 | -0.881672 ± 0.261704 |
| sersic_n | 2.213962 ± 1.243808 | 2.246225 ± 1.280709 | 2.279504 ± 1.342106 |
| distortion_a | 0.584593 ± 0.145458 | 0.579663 ± 0.140297 | 0.580869 ± 0.144780 |
| phase_sin | -0.798865 ± 0.487012 | -0.824176 ± 0.462552 | -0.805156 ± 0.476148 |
| phase_cos | 0.056269 ± 0.348523 | 0.072562 ± 0.318613 | 0.090825 ± 0.341698 |
| crowding_log1p_050 | 0.026802 ± 0.203014 | 0.019953 ± 0.174450 | 0.016299 ± 0.108174 |
| crowding_log1p_100 | 0.066740 ± 0.236830 | 0.053069 ± 0.201938 | 0.049721 ± 0.148623 |
| crowding_log1p_200 | 0.195681 ± 0.322527 | 0.174313 ± 0.281740 | 0.168533 ± 0.237320 |
| response_b | 0.062898 ± 0.208236 | 0.052983 ± 0.171857 | 0.058124 ± 0.184434 |
| neighbour_count | 21.985933 ± 4.745161 | 21.804930 ± 4.623675 | 21.489804 ± 4.504785 |
| log1p_neighbour_target_flux_sum | 1.163958 ± 0.814517 | 1.112117 ± 0.725094 | 1.101757 ± 0.716137 |

Circular phase means/resultants are v5 4.782709 / 0.800844, v6 4.800205 /
0.827364 and v7 4.824719 / 0.810263. Mean phase alone cannot resolve narrow
or multimodal shape structure. Magnitude, size, n and a marginals are close;
this does not establish equality of their joint or conditional distributions.

### Covariance and fixed-bin localization

The CSV retains covariance and correlation for every feature in every run.
For key features, observed within-population correlation ranges are:

| Feature | v5 correlation range | v6v100 correlation range | v7 correlation range |
|---|---:|---:|---:|
| m_r | -0.0453 to 0.0745 | -0.0224 to 0.0363 | -0.1135 to 0.0157 |
| log_radius_arcsec | -0.1140 to 0.1355 | -0.0565 to 0.0513 | -0.0436 to 0.0729 |
| sersic_n | -0.1284 to -0.0011 | -0.0537 to 0.0666 | -0.0867 to 0.1364 |
| distortion_a | -0.1141 to 0.0518 | -0.1227 to 0.0194 | -0.1705 to 0.1869 |
| phase_sin | -0.1800 to 0.1665 | -0.0455 to 0.0396 | -0.0865 to 0.0946 |
| phase_cos | -0.2215 to 0.0332 | -0.1618 to -0.0902 | -0.2586 to 0.0197 |
| crowding_log1p_050 | -0.0405 to 0.0333 | -0.0078 to 0.0156 | -0.0566 to 0.0202 |
| crowding_log1p_100 | -0.0493 to 0.0428 | -0.0127 to 0.0192 | -0.0840 to 0.0139 |
| crowding_log1p_200 | -0.0397 to 0.0135 | -0.0160 to 0.0138 | -0.1045 to -0.0070 |
| response_b | -0.0482 to 0.0221 | -0.0086 to 0.0268 | -0.0702 to 0.0169 |
| neighbour_count | -0.0502 to 0.0896 | -0.0245 to 0.0398 | -0.1034 to 0.0545 |
| log1p_neighbour_target_flux_sum | -0.0524 to 0.0517 | -0.0259 to 0.0247 | -0.1179 to 0.1031 |

Pooled linear correlations are weak: largest absolute values among all
recorded features are about 0.080 (v5), 0.116 (v6) and 0.104 (v7), involving
n or cos(phase). Weak marginal correlations do not exclude nonlinear or
joint mechanisms. Fixed-bin v7−v6 score accounting is:

| Feature bins | Occupancy-composition component | Within-bin component | Unmatched-bin component |
|---|---:|---:|---:|
| m_r | +0.001303 | +0.664115 | -0.000130 |
| log_radius_arcsec | -0.008094 | +0.675553 | -0.002171 |
| sersic_n | +0.021065 | +0.644223 | +0.000000 |
| distortion_a | +0.015683 | +0.649605 | +0.000000 |
| phase | -0.057737 | +0.723025 | +0.000000 |
| crowding_log1p_050 | +0.001518 | +0.663869 | -0.000099 |
| crowding_log1p_100 | +0.001592 | +0.663696 | +0.000000 |
| crowding_log1p_200 | +0.010103 | +0.655185 | +0.000000 |
| response_b | -0.001451 | +0.666688 | +0.000051 |
| neighbour_count | +0.008735 | +0.655850 | +0.000702 |

These rows are separate decompositions of the same difference and must not
be added together. The reported broad occupancy changes account for little
of the 0.665288 gap: most arises within the same bins. Almost the entire
magnitude-bin difference is in 22–24; radius 0.3–1 arcsec contributes 0.631117.
The common phase bin [3pi/2, 5pi/3), holding about 69% of each population,
contributes 0.479548 (72.08%) of the gap, despite its occupancy changing by
only −0.006223. Distortion bins 0.4–0.8 contribute 0.651401 (97.91%).
Most crowding-bin difference is in the low-crowding region. These are
marginal bin descriptions, not a joint region or independent causal effects.

### Bounded interpretation

The discrepancy is distributed across independent runs, partly carried by
score tails, and mainly occurs inside common morphology/phase bins. It is
not explained by one anomalous run, a single initial ancestor, or a gross
shift in these marginal physical summaries. Tail imbalance accounts for
about half, but not all, of the gap. Fine conditional phase/morphology
structure, finite-population effects or a shared coverage failure remain
possible. This analysis does not establish which mechanism is responsible;
the independently declared higher-effort arm is a separate test. No
coefficients, production estimates, inputs or gates were changed.
