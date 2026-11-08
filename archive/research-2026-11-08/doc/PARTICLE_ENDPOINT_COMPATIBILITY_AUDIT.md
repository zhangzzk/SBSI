# Frozen v7/v8/v9 endpoint and exchange-support audit

## Scope and conclusion

This 2026-10-22 audit reads frozen sources, declarations, startup protocols,
reference/configuration hashes and synthetic CPU fixtures. It reads no v8/v9
inference or phase outcomes and no heldout outcomes. Existing snapshots and
evidence are unchanged.

The implementations declare the same endpoint
`q_environment(E) * p_analytic(x) * K(y|x,E)`. v9 uses the identical v7 source;
v8 default and informed share the same endpoint factors. This is **not** an
unqualified exact-kernel compatibility certificate: the old default native
exchange can enter a zero-density part of the continuous numerical environment.
The informed v8 branch rejects that region. Production intrinsic cuts do not
make the discrepancy unreachable. The future source change below fixes both
default and informed exchanges; it does not rewrite old runs.

## Frozen identity evidence

The reproducible whitelist-only script is
`doc/evidence/audit_endpoint_identity_20261022.py`; its full receipt is
`doc/evidence/particle_endpoint_identity_20261022.json`. It verifies all 286
`source_v7` and 298 `source_v8` manifest entries, every recorded implementation
hash in the available startup protocols, and actual model/reference artifacts.
The source roots are under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021`.

| Comparison | Evidence | Qualification |
| --- | --- | --- |
| v7 → v9 | `run_v9.sh:10` uses `source_v7`; all eight v9 startup protocols point there; identical source hashes, proposal, geometry, observations, shear, beta schedule, block sequence and likelihood batch size | N changes 256→1024 and moves/stage 32→64; both use the old support-defective kernel |
| v8 default ↔ informed | Same `source_v8` and exact endpoint string; four available startup protocols per arm checked at receipt time; same N256/T64/M32 and all common settings | Informed bridge adds `R_frozen_environment(E)**(1-beta)`; its optional support guard also changes which exchanges can be accepted |
| v8 ↔ v7 | 21 explicitly listed endpoint-factor sources are byte-identical; all other shared `sbsi/*.py` changes are enumerated in the receipt | Target implementation source changes for optional payload/bridge handling, so old strict source-identity report gates remain incompatible |

The shared factor list includes analytic prior, numerical scene sampler,
`build_prior`, coordinates, native summary, score gradients, catalogue
likelihood, disk transport/precision, flow, classifier preprocessing, model
loading and response prediction. Actual prior and environment SHA256 values are
`f6612ec614f966993946e3beb94221e6e6ad3adcf1782393f18cb8facc782652`
and
`575e4057c94db07f0b14b792628f02e6a21f28443121198e16c669e72f7515ae`;
the complete list, rather than these shortened selections, is authoritative.

All four groups use observations `[58165,0,1,8]`, inference shear `[0,0]`,
density batch 4096, the same reference protocol/observation bytes and V3.9
F8/C8/physical-moment model assets. The response contract is radius 10 arcsec,
nearest 20, unchanged physical conditions. The pilot proposal hash is
`7f11db32019d4e0014a10acc61ea890c17d6fc61b67d0fd9ff36afe3763c46e4`
and geometry hash is
`a2eb847b956a986a94257bf2fce808185813e447dafcc45981f499a7c9d9b2c0`.
These are inference-endpoint comparisons, not a new claim that the analytic
target prior equals the numerical prior that generated the mocks.

## Why the old native exchange violates continuous support

`sbsi/galsbi_prior.py:246` draws `U=max(random(),float.tiny)` and then inverts
`log(max_g)+log(U)`. Consequently the continuous luminosity law has no density
where the inverse-map value gives `log(U)<log(float.tiny)`.
`source_v7/sbsi/scene_exchange.py:46` (also unchanged in `source_v8`) checks the
table domain and upper bound only. Its returned `log(q/p)` is finite in part of
that excluded lower strip. `sbsi/numerical_native_intensity.py:67` contains the
correct lower bound.

For an old analytic target mark u and numerical neighbour v, the native swap
correction contains `log(q(u)/p(u))-log(q(v)/p(v))`. If `q(u)=0`, the proposed
environment has zero target density at **every** bridge beta because qEnv is
not tempered. Any finite correction that can accept this proposal violates
target preservation. Finite analytic p(u), supported v and finite likelihoods
do not repair it. This is a kernel error relative to the declared continuous
endpoint, not a legitimate difference between initial proposal laws.

The optional v8 guard at
`source_v8/sbsi/galsbi_particle_target.py:252–264` is entered only with an
environment payload. It correctly rejects the strip before computing R.
Thus v8 default/informed is not a clean isolation of informed proposal mass
unless a trajectory audit shows that the extra rejection never mattered.
v7/v9 share the defect, but different effort can still produce different
exposure. Whole-population MCSE cannot certify that this target-preservation
error had no effect, and a disagreement cannot yet be attributed exclusively
to insufficient equilibration.

## Actual prior counterexamples and practical limits

CPU job 17421563 runs `audit_native_exchange_support_20261022.py` against the
untouched `source_v8` implementation. The retained receipt
`doc/evidence/native_exchange_production_support_20261022.json` contains full
native marks, photometry, exact sources and all scan counts. No prior setting
or magnitude cut was changed. On a 96-point redshift grid per type, using the
mean interior SED, 82 red and 88 blue points pass the real i14–28 cut and have
finite old corrections in the continuous-zero strip. These counts are an
existence demonstration, not occurrence-rate estimates.

| Native type | z | M_B | intrinsic i | log analytic intensity | old log(q/p) | inverse log(U) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| red | 0.7006944346 | −28.3120124503 | 14.2427648456 | −675.0037801359 | −0.0002244804 | −708.6385187909 |
| blue | 0.5123232251 | −27.7206077002 | 14.0272037010 | −682.0992118881 | −0.0002530466 | −709.3551234047 |

The lower continuous bound is −708.3964185323. No production trajectory was
replayed here. Small prior tail mass alone does not bound impact: broad q0
draws and low-beta target proposals can visit regions that are extremely rare
under the prior. Last retained particles also cannot exclude earlier accepted
excursions. This finding does not identify the cause or size of any score gap.

This fix addresses a continuous-density support error. The ideal continuous
uniform with a lower clamp has a singular atom, and a literal finite RNG has a
discrete law. Neither is fully represented by these Lebesgue densities. The
informed law explicitly raises on detected clamp atoms; the default path does
not gain a complete mixed-measure implementation from this narrow fix.

## Future fix, diagnostics and validation

`sbsi/scene_exchange.py:numerical_over_analytic` now sets the correction to
−infinity below the lower bound for every caller. Its existing supported-path
floating arithmetic and proposal RNG sequence remain intact. Both native
exchange blocks inherit the restriction without changing proposal draws or
other mutation laws. The optional informed guard remains a defensive check.

The opt-in `return_support_audit=True` returns the correction plus aligned
`legacy_finite` and `below_uniform_clamp` masks. It consumes no random numbers
and supplies no production option to disable the restriction. For a later
historical-trajectory check, instrument an isolated frozen helper: evaluate
these masks, return the **original** correction, and pair the old/new mark
masks within each exchange. Count rows for which both legacy corrections were
finite and either mark is below the bound; also count old accepted proposals
in that set. Continue the original RNG and retain original denominators.
A final-particle-only check is insufficient. No such inference replay was
performed in this audit.

Focused CPU job 17421564 passed 118 tests, with one optional frozen-baseline
test skipped and one existing NumPy deprecation warning. Regressions include
default exchanges at beta=0,.001,.4,1; both exchange variants; informed
payload retention; and finite density immediately above the clamp boundary.
Final CPU job 17421627 passed 795 tests (3 optional skips) in 60.21 s. Its
independent frozen-v7 comparison also reproduced full ordinary state, decisions
and RNG across eight blocks at beta=.001,.4,1, plus bitwise ratios for 268
generated production marks. Both real-prior counterexamples now reject with
identical proposal payload and RNG. Before/after source hashes passed. Details
are in `doc/evidence/endpoint_support_fix_validation_20261022.json` and
`doc/evidence/native_exchange_support_fix_20261022.json`.

The first two support-harness jobs (17421485,17421519) failed before building
the prior because importing PyCosmo before torch disturbed importlib metadata
initialization. The harness now imports torch first, matching the production
driver. The failed logs are retained; no scientific inputs were changed.

## Missing historical provenance and a bounded future check

v7/v9 do not record `endpoint_target`. Default startup protocols omit prior,
environment, prior factory and some transitive model sources. Frozen snapshots
provide explicit current source evidence, not retroactive protocol fields.
The GalSBI/PyCosmo installation, ABC-row data, photometry/cosmology resources
and constructed prior tables were not all pinned in those protocols. Matching
current files and shared source cannot prove that every unrecorded external
dependency was immutable at each historical startup. No existing report gate
is bypassed by this receipt.

The phase checker still lists only coordinates, summary, gradients and model
manifest in `scripts/check_particle_phase_control.py:27–28`; its reporter
compares that reduced `target_identity` at lines 38–44. It could silently
combine changed non-phase priors. A small future fail-closed identity should
be shared by inference, phase replay, frozen control fitting and reporting:

1. Version and canonical endpoint measure, including continuous numerical
   environment/clamp qualification, observation hash, conditions and shear.
2. Required prior/environment/factory/coordinate/summary/likelihood/model
   source identities and actual model/config artifacts, not optional hashes.
3. Serialized effective prior parameters plus dtype/shape/byte hashes of the
   constructed redshift/luminosity tables and pinned external prior data.
4. A separate bridge/proposal identity. Different bridges may compare only
   when complete endpoint identities match; missing fields or changed factors
   reject by default. Historical equivalence needs a separate reviewed receipt,
   not inserted metadata or an ignore/allow flag.

This is a recommendation, not an implemented identity-gate expansion.
