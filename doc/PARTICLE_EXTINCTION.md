# Particle population extinction

## Numerical meaning and absorbing state

`sbsi/particle_populations.py` records extinction only when an observation's
incremental annealing weights are all zero. Its normalizing-constant estimate
then remains zero for that seed. No redraw, retry, uniform reweighting or
subsequent mutation revives that population. NaNs, overflow and arbitrary
execution failures remain errors.

Other observations continue with their original observation IDs, proposals
and frozen geometry. Only surviving particle rows enter mutation; complete
target arrays and ragged neighbour marks/positions are scattered back into
their original slots. The all-surviving path retains the original calls and
random-number order. If all populations are extinct, subsequent stages make
no model or mutation calls, while the receipt still reaches the declared
final stage and beta.

## Receipt and artifact contract

The protocol, result and final stage declare
`extinction_schema="per-observation-zero-mass-v1"`. Each result includes:

- `extinct`: one Boolean per observation.
- `extinction_stage`: the first all-zero-weight stage, or `null` for survivors.
- `log_normalizer`: a finite value for survivors and `null` for extinct
  observations. Only the explicit mask interprets that `null` as log zero.
- `score_mean` and `score_variance`: finite two-component vectors for
  survivors and two `null` entries for extinct observations.
- `status`: `completed_with_extinction` if any observation is extinct;
  otherwise `completed`.

Final scene-region fractions and magnitude quantiles are also null for extinct
observations. Their ancestral diagnostics and attempted/accepted mutation
counts after extinction are zero. Initial retained physical payloads can stay
in the binary scene artifact, but they are not posterior samples.

`particles.npz` contains the Boolean extinction mask and numerical log
normalizers (`-inf` for zero mass). Score and summary arrays use finite zero
placeholders for extinct observations, which are undefined under this mask.
Surviving likelihood replay and score evaluation occur on surviving rows only.
The all-extinct result has `final_state_replay_max=null` because no posterior
state was evaluated.

## Independent-population reporting

`scripts/report_galsbi_particles.py` verifies the declared schema, status,
mask/stage consistency, null placement, binary mask/log normalizers, artifact
hashes and recorded implementation identities. Missing artifacts and unknown
failed runs are rejected; a missing run is never assigned zero mass.

The estimator retains every declared seed. Zero-mass populations contribute
zero to the unnormalized numerator and denominator but remain in the number
of independent populations used for the average normalizer and its errors.
Undefined extinct score moments are masked before forming empirical second
moments, avoiding `0 * NaN` arithmetic.

If every population for an observation is extinct, its posterior mean,
variance and relative normalizer error are undefined. The report emits null
values and fails its precision and comparison gates. A zero empirical score
MCSE also does not earn an infinite effective-count claim: its count is null
and the precision gate fails.

Legacy completed, surviving receipts remain readable. Old failed or missing
receipts are not retrospectively classified as extinctions. Report version
`independent-populations-v3-explicit-extinction` must match across comparisons.

## Validation and scope

The CPU tests in `tests/test_particle_extinction.py` cover exact legacy-loop
state/RNG replay, partial/all extinction, ragged replacement, original
observation ownership, survivor-only diagnostics, three mocked end-to-end
driver runs, zero-mass report aggregation, and malformed/missing artifacts.
The targeted suite passed 33 tests in Slurm job 17419103 (19.43 seconds).
The full CPU suite passed 607 tests with one skip and one existing warning in
Slurm job 17419141 (108.26 seconds). Its command was
`py31 -m pytest tests -q` through `jobs/job_tests.sh` on `cluster`.

These are implementation checks. No real-data extinction was observed in this
task, no GPU experiment was run, and no sampling-efficiency improvement is
claimed. This change preserves failed normalizing-mass trials correctly; it
does not by itself make a poorly mixing sampler reliable.
