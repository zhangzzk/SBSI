# ConstGold: fixed detection, separate selection

## Definition and purpose

Requested on 2026-09-06 to diagnose the strong paired-transport model's
complete-population **m=+1.075825% +/- 0.219835 percentage points**.
Keep the identities with usable measurements and detections in **both**
antithetic legs, then apply each leg's measured cuts independently.
The two selected sets can differ. This is different from requiring an
object to pass selection in both legs.

With this fixed parent D, leg-specific cuts C and measured ellipticity e,

    mean_e_plus  = sum_{i in D} C_i_plus e_i_plus / sum_{i in D} C_i_plus
    mean_e_minus = sum_{i in D} C_i_minus e_i_minus / sum_{i in D} C_i_minus
    R11 = (mean_e_plus[0] - mean_e_minus[0]) / (2h)

The model uses the same parent identities and applies its own cuts to each
of 64 flow draws per galaxy per leg. Pool passing shape sums and passing
draw counts before taking each leg's mean. It uses no classifier weights
in this branch. The unchanged R_blend shift is included before selection.
Use common random numbers between legs. See [response conventions](CONVENTIONS.md#6-response-estimators)
and [the definition of m](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).

A fixed detected parent removes changes in detection membership from this
finite-difference comparison. It does not make the flow explicitly
conditional on the joint event of being detected in both legs. Therefore
this is a diagnostic of measurement plus selection on that parent, not a
replacement for complete-population closure or a retrained model.

## Population, cuts, and reproducibility

- Cases 40–139; truth primary domain 18<mag<25.8 and 0.37<Re<1.5.
- Measured |e|<0.6, MAG_AUTO<25.8, flux radius >=0.75 arcsec.
- Antithetic h=0.02; 64 QMC draws, sampling seed 7301.
- Strong paired-transport checkpoint SHA-256:
  `91dbfac91a16ffd975ec5d3f8298896f66b9524f12ef404245069e706319b506`.
- Bootstrap whole cases jointly across model and simulation, 10,000
  replicates, seed 20260902. Pool per-leg numerators and denominators in
  every replicate. Never average case or shard m values.
- Preserve per-case missing-crossmatch, invalid-measurement and support
  exclusion reports. Both-detected here means **both usable and detected**.
- One training seed; case uncertainty does not measure training-seed or
  finite-draw uncertainty.

## Implementation and execution

`scripts/evaluate_constgold_complete_forward_response.py` adds a separate
`matched_detected_selected` branch and serializes its per-case leg totals.
The original complete, matched-detected and matched-selected branches keep
their existing definitions. Tests check draw-level cuts, independent
leg selection, independence from detector weights, correct pooled ratios
with unequal denominators, and the paired-case bootstrap.
Initial full suite: **321 passed, 2 skipped**, 37.24 s, Slurm **16327285**.
The first array 16327311 exposed missing recursion through lists in JSON
serialization of the new raw per-model totals. No output was written.
Stopped the other affected shards, added list/tuple serialization and a
round-trip regression test, and queued the retry after full tests **16327357: 322 passed, 2 skipped,
37.81 s**.

External artifacts are under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/paired_transport_train_v1/both_detected_separate_selection/`.
`evaluate.sh` runs ten disjoint ten-case shards, Slurm array **16327359**;
`combine.py` and `combine.sh` pool the raw leg totals and recompute the
100-case bootstrap, dependent job **16327400**. Combination rejects
inconsistent models, cuts, sampling, input provenance or case coverage,
and verifies the old complete branch against its saved 100-case result.
One retry worker (16327359_0) still loaded the old serializer and failed;
tasks 1–3 saved successfully and reproduce original per-leg complete means
exactly. Replacement task **16327399_0** includes a startup serialization
check. The combiner waits for both the original array to finish and this
replacement to succeed; it still requires all ten output files.
No output correction, training, or model promotion is part of this test.

## Completed 100-case result

Final combiner **16327400** completed in 18 s, exit 0, empty stderr.
All ten shard outputs cover exactly cases 40–139; source identities,
models, cuts, sampling and inputs match. Every compared original complete
mean, response, count and population fraction is **exactly** reproduced
(maximum difference 0), as are the original full m and bootstrap SE.

| Population and cuts | Measured R11 | Model R11 | m (%) | Case SE (pp) |
|---|---:|---:|---:|---:|
| Independent detection and selection per leg | 0.606383322 | 0.599929133 | +1.075825 | 0.219835 |
| Both usable/detected; separate selection per leg | 0.612282077 | 0.596558679 | **+2.635683** | **0.217120** |
| Same both-usable/detected parent; no measured cuts | 0.960887415 | 0.985982290 | -2.545165 | 0.144319 |

The requested diagnostic's 95% paired-case interval is
**[2.214406%, 3.060383%]**. Parent size is **8,676,652**; measured selected
counts are **4,836,710 / 4,835,747** in plus/minus legs. Model passing-draw
weights (draw counts divided by 64) are **5,513,559.828 / 5,507,166.266**.

On that fixed parent, selection changes measured R11 by **-0.348605337**,
versus model **-0.389423611**. Its excess change is **-0.040818273 +/-
0.001841185**. Before selection the model overpredicts R11 by
**+0.025094875**; these contributions partially cancel, leaving the selected
response residual **-0.015723398 +/- 0.001295452**. Thus changing detection
membership is not necessary for the observed failure: measurement and
selection on the fixed detected parent still disagree. Conditioning on
both detections remains a diagnostic limitation, so this does not identify
an isolated causal detector error or certify unconditional flow behavior.

The requested comparison is complete. The full-population 0.3% goal remains
unmet. Results and per-case exclusion reports: external `comparison.json`;
code snapshot and job provenance: `evaluator_snapshot.py`, `run_manifest.json`.

