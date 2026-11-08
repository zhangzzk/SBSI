# Independent informed-environment review — 2026-10-22

## Mathematical and payload review

The optional n16 implementation passes the independent native-measure and
bridge-ratio checks below. This is an implementation result, not evidence of
posterior convergence, good normalizer precision or improved throughput. No GPU
job was run by the reviewer, and the reviewer did not edit the implementation.

For each fixed observation, the implemented h is the frozen nine-dimensional
Student-t mixture with independent uniform phase, independent uniform native
type, and the normalized central-Gaussian/uniform-aperture position mixture.
`informed_environment.py:85–108,171–203` uses the full native Jacobian, converts
the numerical mark intensity from per steradian to per arcsec squared, and sums
h/lambda over **every** neighbour. Reconstructing h removes the source artifact's
nonuniform phase parameters. It remains independent of the current target.

The ordered insertion is uniform among the original N+1 slots. With the ordered
Poisson 1/N! convention, explicitly summing those histories gives
Q/qEnv=(1−a)+a sum h/lambda, without another N or expected-count factor.
`informed_environment.py:236–272` preserves old neighbour order and native fields
while inserting only valid sampled points. Invalid raw points remain recorded
in the payload and contribute zero target mass before ordinary resampling.

The target potential subtracts log R. The reduced bridge adds
(1−beta)log R, and every environment-changing mutation recomputes R. Target-only
physical/nuisance/SED/independence moves reuse R; the independence Hastings term
uses qtarget alone. `galsbi_particle_target.py:19–68,132–177,250–310` and
`particle_populations.py:58–66` preserve the tilt and every origin-evidence field
through acceptance, repeated ancestors and partial-observation scatter.

Two necessary edge guards are present in the reviewed version:

- A native exchange whose reverse numerical mark density is zero resets the
  proposed native payload to the old scene before evaluating R, then rejects
  through its minus-infinity Hastings correction. It does not send a known
  unsupported environment to likelihood evaluation.
- Existing numerical clamp atoms raise explicitly, as do unexplained numerical
  failures. Only analytically classified raw support violations become zero
  initial weights. The law is qualified as the continuous interpolation
  component; full finite-RNG/mixed-measure exactness is not claimed.

## Independent CPU evidence

Slurm **17420035**, cluster, **COMPLETED 0:0**, **17 seconds**, evaluated a native
fixture with0,1 and2 neighbours. The reviewer used a separate Student-t formula,
explicit native Jacobian, position-mixture formula and existing shared-support
q/p evaluator. The complete scene mixture density was assembled by summing all
insertion histories, rather than reusing the new tilt helper.

| Check | Result |
|---|---:|
| Full ordered-mixture log density versus implemented tilt | maximum error7.11e−15 |
| Actual position/environment/native-exchange MH ratios at beta=.001,.4,1 | maximum error1.60e−14 |
| Explicit removal of source von Mises phase dependence | passed |
| Invalid initial rows skip K and cannot mutate before resampling | passed |
| Initial zero weights retain the original N denominator | passed |
| Complete payload scatter, including every raw origin field | passed |

The source hashes were unchanged during the CPU check. Reproducer:
`doc/evidence/review_informed_environment_20261021.py`; receipt:
`doc/evidence/informed_environment_review_20261021.json`. The original receipt is
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/smc_informed_review_20261021/result.json`.
Those hashes identify the tested files independently of subsequent edits.

After the support-boundary fix below, independent rerun **17420432** completed
**0:0 in 29 seconds** against the frozen final implementation. The explicit
mixture error remained **7.11e−15** and the maximum mutation-ratio error was
**2.14e−14**; all other checks passed. Source hashes were unchanged during this
rerun. Its separate pinned receipt is
`doc/evidence/informed_environment_review_final_20261022.json`.

## Final driver and support-boundary review

The driver saves every original raw environment draw and classification in
`initial_environment.npz` before resampling, with its hash and per-observation
counts in the result. Final particles retain all environment payload fields.
`scripts/check_galsbi_particles.py:126–143,182–196,245–254` also records the
source artifact and continuous-measure qualification. The source pilot's seed,
observation order, native target identity and original bridge are validated
before the optional R bridge suffix is recorded. No optional artifact or new
payload keys are introduced on the disabled path.

This review identified a narrow mismatch between the older native-exchange
q/p helper and the new continuous native intensity: the old helper can assign
a finite density to the inverse-table strip with log U < log(float.tiny),
where the continuous component has zero density. Sending such an exchanged
target into the environment would previously raise while computing R.
The author fixed this with the optional-law-specific
`continuous_exchange_support` guard (`informed_environment.py:204–234`),
called after the existing native support gate
(`galsbi_particle_target.py:252–266`). The guard restores the old safe payload
and gives the proposal a minus-infinity Hastings correction. Conditional-cap
failures, clamp atoms and unexplained density errors still raise; they are not
classified as support zeros.

The regression exposes that actual inverse-table strip by widening only the
toy intrinsic magnitude window. It verifies finite analytic target density,
finite old q/p correction, zero new continuous environment density and unchanged
retained state after rejection. This is a support-boundary correctness fix,
not an observed production failure or explanation of the score residual.

Author CPU **17420200** passed **173 tests** before this fix. Final CPU
**17420283**, cluster, **COMPLETED 0:0**, passed **174 tests in 31.29 seconds**,
including the new support regression, driver raw receipts and partial extinction,
payload ancestry/scatter, mutation ratios and disabled-path bitwise state/RNG
replay. The frozen final source hashes and JUnit receipt are in
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/informed_environment_validation/final_input_v2.json`
and `final_junit_v2.xml` in the same directory. This broader author test receipt
is distinct from the independent mathematical checks above.

## Remaining report aggregation gate

The existing reporter fingerprints physical geometry and target initial proposal
artifacts only. It must also fingerprint the informed-environment artifact and
its pilot seeds before scientific reporting. It additionally requires equal
literal bridge descriptions and target-module hashes across comparison reports.
Consequently old v5/v6 reports are not automatically compatible with the new
implementation even though the beta=1 target is unchanged. Preserve that gate:
use explicitly reviewed compatibility evidence or a fresh baseline under the
same source. This review does not authorize weakening identity checks.

The generic Poisson helper's former global h<<lambda/full-off-support-density
wording now includes the valid-scene/zero-initial-weight qualification documented in
[the bridge audit](PARTICLE_BRIDGE_AUDIT.md#addendum-off-support-informed-poisson-draws-need-not-be-truncated).
No modification to its finite-tilt formula was required. Reporter integration
and explicit comparison compatibility remain pending in this review receipt;
the coordinator assigned those changes as a separate bounded follow-up.

## 2026-10-22 addendum: common endpoint does not clear the old default kernel

The [frozen endpoint audit](PARTICLE_ENDPOINT_COMPATIBILITY_AUDIT.md#why-the-old-native-exchange-violates-continuous-support)
establishes that the missing lower uniform-clamp restriction is a real
target-preservation error in the default exchange, not just an optional-law
edge check. Actual red and blue prior marks pass i14–28 while having finite
old corrections and zero continuous environment density. The informed guard
correctly rejected them; a new unconditional helper restriction fixes future
default exchanges too. Existing snapshots are preserved. Their declared
endpoints match, but practical impact requires trajectory evidence; no prior
tail-mass argument or final-state-only check establishes absence of impact.
