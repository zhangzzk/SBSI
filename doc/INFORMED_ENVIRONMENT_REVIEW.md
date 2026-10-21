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
`informed_environment.py:205–241` preserves old neighbour order and native fields
while inserting only valid sampled points. Invalid raw points remain recorded
in the payload and contribute zero target mass before ordinary resampling.

The target potential subtracts log R. The reduced bridge adds
(1−beta)log R, and every environment-changing mutation recomputes R. Target-only
physical/nuisance/SED/independence moves reuse R; the independence Hastings term
uses qtarget alone. `galsbi_particle_target.py:19–68,132–177,250–298` and
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

## Integration gates still outside this CPU check

The optional driver must save the original invalid raw draws and classifications
before resampling, because surviving ancestry cannot reconstruct their initial
frequency. Its artifact/protocol hashes and all-extinct receipts need the same
explicitness as the mathematical payload. The implementation author's broader
test suite covers additional blocks, extinction and default bitwise replay;
their final receipt is separate from this independent check.

The existing reporter fingerprints physical geometry and target initial proposal
artifacts only. It must also fingerprint the informed-environment artifact and
its pilot seeds before scientific reporting. It additionally requires equal
literal bridge descriptions and target-module hashes across comparison reports.
Consequently old v5/v6 reports are not automatically compatible with the new
implementation even though the beta=1 target is unchanged. Preserve that gate:
use explicitly reviewed compatibility evidence or a fresh baseline under the
same source. This review does not authorize weakening identity checks.

The generic Poisson helper's former global h<<lambda/full-off-support-density
wording needs the valid-scene/zero-initial-weight qualification documented in
[the bridge audit](PARTICLE_BRIDGE_AUDIT.md#addendum-off-support-informed-poisson-draws-need-not-be-truncated).
No modification to its finite-tilt formula is required.
