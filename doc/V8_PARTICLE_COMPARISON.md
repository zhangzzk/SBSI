# N16 completed informed-environment comparison

## Question and fixed accounting

The original n16 hypothesis was that explicit informed-neighbour mass could
expose alternative light assignments missed by target-only proposals. Both
v8 arms use N256/T64/M32 and eight independently seeded populations under
source_v8. Baseline seeds are 2026102600–2026102607; informed seeds are
2026102700–2026102707. The informed arm adds a single-point augmentation with
probability 0.5, uniform native type and phase, and a 90% central-position
component at scale 0.1 arcsec. The fitted target proposal, physical geometry,
observation panel and shear are frozen. No coefficients are refit.

The standalone summary reuses the tested v9 reporting arithmetic. It verifies
all referenced hashes, declared populations, all states/stride 1, frozen phase
coefficients and raw score/Z replay. Signed informed−baseline differences use
independent-population MCSE sqrt(MCSE_informed²+MCSE_baseline²). All existing
whole-population Z weighting and original full-N F64 precision-reweighting
accounting remain unchanged. Raw regions come directly from the audited raw
reports; no F64 region reweighting is calculated.

Baseline's `comparison_gate=false` means **no comparator was supplied**, not a
failed comparison. The informed report contains the actual comparison against
baseline. Existing raw score, region and normalizer gates remain unchanged.

## Required limitations

The old baseline correction can admit marks with **zero continuous numerical
environment density despite finite analytic target intensity**. The informed
arm rejects that region. Whether these trajectories encountered this support
defect is unknown. Thus this experiment changes both informed proposal mass
and the support guard, and differences cannot be attributed solely to informed
mass. Matching declared endpoint factors do not certify either old transition
kernel or a literal finite-RNG target. The separately corrected v10 experiment
is outside this summary and is not read.

Historical F32 phase aggregates incorrectly asserted a known-zero control
expectation. This new summary preserves their historical field but sets the
applicable F32 expectation to null/unknown. Explicit F64 reweighting changes the
numerical measure; the zero-mean identity applies to the intended continuous
posterior, numerically evaluated, not an exact rounded program.

Optional observation-0 baseline contrasts against v6/v7/v9 are descriptive
under the separate frozen endpoint receipt. They do not invoke or bypass the
strict report source-identity gate. Eight populations cannot establish rare-mode
coverage or exactness. Cost comparisons use recorded inference runtime and
exclude pilots, phase diagnostics, queueing and startup outside driver timing.

## Preserved reporting failure and numerical repair

CPU 17422693 passed all six initial tests, then stopped before writing a result:
baseline observation 8 raw-report and phase weight arrays differ by
1.3877787807814457e-17 (one float64 ULP), while their log normalizers agree
exactly. All other arrays agree bitwise. Requiring exact equality across
independently computed aggregates was a reporting implementation error.
The v2 summary permits at most eight ULPs per finite weight and records the
maximum observed difference, without changing either weight array or any
scientific gate. A focused regression accepts one ULP but rejects a 1e-8
perturbation, missing entries and NaNs. The failed source, receipt and log are
preserved. Scientific inputs and the n16 hypothesis remain unchanged.

## Reproduction and result

Pending focused CPU tests and the frozen completed-data summary. No GPU,
new model forward, fitting, held-out input or sampler changes are part of this
reporting task. Original reports and source snapshots are preserved.
