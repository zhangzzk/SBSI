# N18 higher-effort fitted-proposal comparison

## Question and accounting

The preserved n18 declaration increases the frozen fitted-proposal arm from
v7 N=256/T=64/M=32 to v9 N=1024/T=64/M=64, with eight fresh seeds
2026102800–2026102807. It retains the v7 source, proposal and geometry, and
compares against eight independently seeded v6v100 broad-proposal populations
at N=1024/T=64/M=64 and the original eight v7 populations. Its motivation was
the observation-0 corrected-g1 disagreement of about 4.54 combined MCSEs.
No phase coefficients are refit: only the four v1 pilots fitted the frozen
constant control. No held-out outcomes enter this comparison.

`scripts/summarize_v9_particle_comparison.py` reads completed audited raw
reports and completed phase aggregates. It verifies referenced hashes,
independent seeds, all eight populations, stride 1, all original particles,
matching target identity and the frozen coefficient artifact. It checks raw
score, MCSE, weights and normalizer replay between raw and phase reports.
It preserves existing whole-population Z weighting and the original full-N
F64 reweighting denominators. It adds no particle thinning or aggregation fit.

Each signed contrast is A−B with independent-group MCSE
sqrt(MCSE_A²+MCSE_B²). Normalizer contrasts use a common exponential scale
and the reported relative MCSE. Both sampled-F32 and explicitly reweighted-F64
measures are reported separately. Every observation/component and sign is
retained. Variance times total recorded inference cost is descriptive and
excludes bias, pilot fitting, phase diagnostics, scheduler delay and startup
outside the driver's timer. Different N/M arms are kept explicit.

## Shared correctness limits

All these old trajectories share a native numerical proposal-support defect:
the q/p implementation omitted the native uniform lower clamp, permitting
environment marks with zero real-prior density. Whether an old trajectory
actually encountered the defect is unknown. Root's separate fix dca103e is
not present in these frozen trajectories. These comparisons cannot certify
exactness or establish that every discrepancy arises from equilibration.

Historical phase aggregates incorrectly label F32 control expectation as
known zero. The new summary retains that original field as historical metadata
and sets the applicable F32 expectation to null/unknown. Zero expectation in
the F64 diagnostic is qualified as the identity for the intended continuous
posterior, numerically evaluated; it is not an exact identity of a rounded
floating-point program. F64 precision reweighting explicitly changes the
numerical measure. Existing raw score/Z gates remain unchanged; no new
corrected-score acceptance threshold is introduced. Eight populations cannot
rule out shared rare-mode bias.

## Reproduction and results

Pending frozen CPU summary and focused correctness tests. Source and input
receipts will be recorded here after execution. Original reports, phase
outputs and source snapshots remain untouched.
