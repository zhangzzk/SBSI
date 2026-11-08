# Optional informed initial environment — fixed n16 design, 2026-10-21

This design was recorded before implementation. No production run or efficiency
claim is authorized by these CPU implementation tests. Existing defaults remain
disabled, with identical arithmetic, random draws and saved payload fields.

## Initial law and reference measure

For each fixed observation, independently draw the existing numerical Poisson
environment qEnv. With fixed probability a=0.5, augment it with one independent
h point, inserting uniformly among the resulting N+1 ordered slots. The optional
mixture is Q_E=(1-a)qEnv+a(qEnv augmented by h). No point sorting, retry,
truncation, filtering, or population refill is permitted.

h uses the frozen independent-pilot target proposal's nine-dimensional Student-t
mixture, with its phase law explicitly replaced by independent uniform phase.
Native type is independent uniform over the declared types. Position follows
the existing normalized truncated central Gaussian/uniform-disk mixture, with
scale0.1 arcsec and central weight0.9 in the native aperture. New metadata records
the source artifact hash and phase override; old component phase metadata must
not survive through generic dictionary copying.

For a representable native point, h=qX(x) p(type) qposition/|d native/dx|.
The base point intensity in native-mark/type and arcsec Cartesian position
measure is lambda=q_native/ARCSEC2_PER_SR. q_native is the standalone numerical
continuous intensity, including inverse-CDF slopes and generator cuts, not
analytic intensity or the analytic-support-limited exchange correction.

On valid scenes R(E)=(1-a)+a sum_j h(point_j)/lambda(point_j), summing every
neighbour, not only the inserted point. h depends on the fixed observation but
not the current target. The bridge is
qEnv qtarget^(1-beta) (pK)^beta R^(1-beta), and its incremental potential is
logp+logK-logqtarget-logR. Every mutation changing E includes
(1-beta) delta logR; target-only moves may reuse R. Independence-target Hastings
uses only qtarget. Complete take/scatter/resampling/extinction retain R.

## Support and numerical failures

The full-support raw coordinate draw and type are drawn once. Proven native or
numerical-environment support violations remain explicit zero-target initial
rows. They keep their original population denominator and are removed only by
the existing strictly-positive-beta weighting and ordinary resampling. No
likelihood or mutation is allowed on those rows beforehand. Raw coordinates,
type and classification are retained; safe native placeholders are bookkeeping
only. The off-support mixture density need not be evaluated on this execution
path. Photometry must never see out-of-domain redshifts.

Unknown numerical exceptions, inverse-table generator failures, unexplained
nonfinite density/decoder results, and representation overflow are errors, not
known zero scientific mass. They must raise and never be retried or relabelled.
The numerical density covers the continuous interpolation component; the tiny
uniform-clamp atom and discrete RNG law are not implemented. No mathematical
finite-RNG exactness claim is made. Parent bridge audit establishes the allowed
extended-space zero-weight execution order, subject to these qualifications.

## CPU acceptance gates

Check an analytic toy ordered-Poisson mixture against its direct density and
normalization; native Jacobian/unit/type factors; all-neighbour tilt; support
classification and no retry; initial zero-weight denominator; complete state
resampling/scatter/extinction; actual bridge MH ratios for physical, nuisance,
SED, independence, environment, positions and native exchange; and bitwise
default replay. Only after module tests pass may the optional driver flag and
protocol/source hashes be wired. Keep all source changes isolated from current
running snapshots and parent files. The saved source receipt identifies copied
dependencies; only changed/new implementation files are to be integrated.

## Implementation and evidence — 2026-10-22

`sbsi/informed_environment.py` implements the fixed observation-specific point
law, the all-neighbour tilt, explicit support classification, and the complete
`EnvironmentPayload`. `galsbi_particle_target.py` carries the tilt through
weighting and every affected Metropolis ratio; `particle_populations.py`
preserves it through scatter and extinction. The optional exchange path also
rejects the inverse-table strip with log U below log(float.tiny): the older
numerical/analytic ratio helper can be finite there although the chosen
continuous numerical density is zero. Clamp atoms and unknown numerical failures
still raise, rather than becoming rejected scientific samples.

The driver accepts `--informed-environment FROZEN_PROPOSAL.json`, optionally with
`--environment-mixture-probability 0.5`. It validates the source identity and
pilot seeds before model loading. `initial_environment.npz` records every raw
draw and classification before reweighting or ancestry selection, with its hash
and original per-observation denominator in `result.json`. Final retained
payloads are saved in `particles.npz`. Source hashes and the actual bridge enter
the protocol only when enabled. With neither flag, the original random stream,
arithmetic, and saved payload schema remain unchanged.

CPU job 17420283 passed all 174 checks, including 40 new module/driver cases and
bitwise replay against the frozen original target module. The preceding job
17420200 passed 173 checks before the support-edge regression was added. The
first module job 17420009 failed one finite-ratio assertion because its hot
nuisance fixture had negligible supported mass; moving only the toy fixture's
redshift into its ordinary support gave 100/100 checks in job17420067. All
receipts, including the failure, are preserved under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/informed_environment_validation`.
The `final_run_v2.sh` launcher pins the tested source hashes, uses one CPU,
checks the imported worktree, and supplies the frozen default-replay module.

Independent final audit17420432 reconstructed the ordered native mixture
through insertion histories and independently assembled actual mutation ratios
at beta0.001,0.4,1. Maximum errors were7.11e-15 and2.14e-14 respectively;
invalid-weight, skip/guard, phase, and all-field scatter checks also passed.
Its frozen-source receipt is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/smc_informed_review_final_20261022/result.json`.

These checks establish implementation behavior on the tested continuous law.
They do not show faster posterior exploration, lower normalizer variance, or
accepted shear calibration. Independent report aggregation requires its own
informed-artifact identity and seed gates, and a compatible fresh baseline;
existing strict target/bridge report gates have not been relaxed. No GPU
inference was run for this implementation.
