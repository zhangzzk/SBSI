# V3.9-like retained model

## Decision and identity

On 2026-09-25 the owner closed the simplification investigation and named the
working **F8 flow + C8 classifier + original disk emulator** `V3.9-like`.
The [manifest](../configs/models_v3_9_like.json) pins all four artifact hashes,
feature conventions, measured cuts, primary seed, replicate and evidence.
This is the retained development model. Naming it does not convert the failed
broader non-degradation checks into passes or establish independent validation.

| Component | Inputs | Output | Retained artifact |
| --- | --- | --- | --- |
| F8 joint flow | Eight target/crowding features below | Joint measured e1, e2, flux radius and flux | `bd51dbbbeeb7…` |
| C8 usable-event classifier | Same eight raw features | Usable-measurement probability | seed20260913, epoch99; `231bbbdf1f88…` |
| Physical-moment disk emulator | Seven target–neighbour features plus generated target log radius and magnitude | Scalar pair disk-transport coefficient | 218 trees, max depth 11; `194c1d11b67d…` |

The primary classifier is the first predeclared seed, not the seed with the
most favorable bias. Seed20260914 remains a diagnostic replicate; no averaging.
Emulator simplification was discussed but never run. Historical V3.6-like and
V3.5-like identities remain distinct.

## Feature and prediction contract

The flow and classifier share this ordered raw input:

| Feature | Meaning |
| --- | --- |
| `e1_input_p`, `e2_input_p` | Target input ellipticity |
| `sersic_n_input_p` | Target Sérsic index |
| `r_input_p` | Target true r-band magnitude |
| `circularized_Re_input_p` | Pre-shear circularized effective radius |
| `nbr_close_050`, `nbr_close_100`, `nbr_close_200` | Physical Gaussian crowding at 0.5, 1 and 2 arcsec |

Each crowding value is `log1p(sum_j (F_j/F_i) exp(-d_ij²/(2s²)))`, using all
rendered nonself neighbours within 10 arcsec. It is not the old PSF-normalized
17-input classifier representation or the emulator's truncated pair list.
At tested shear, positions, fluxes and circularized radii remain fixed;
only target ellipticity changes. The flow appends missing-value indicators;
the classifier standardizes the eight inputs without indicators. Always use
the checkpoint's own preprocessor.

Emulator inputs are `Re_input_p_scaled`, `Re_input_s_scaled`,
`r_input_p_scaled`, `r_input_s_scaled`, `sersic_n_input_p`, `sersic_n_input_s`,
`distance_scaled`, `baseline_log_flux_radius`, `baseline_mag_auto`.
At inference the last two are `ln(radius_pixels)` and `30−2.5 log10(flux)`
from the **same joint F8 draw used for measured selection**. Training used
baseline measured outputs. Sum pair coefficients and apply the physical
Möbius disk composition to shape; it does not change the draw's radius/flux.

## Development evidence

Evaluation uses cases40–119, g1 antithetic h=0.02,64 common-random-number draws,
latent seed7301. Each leg independently applies measured
`FLUX_RADIUS >0.60 arcsec` (3pixels) and `MAG_AUTO <25.8`, with its own selected
normalization. There are no truth-property or zero-shear-usability cuts on the
generated target-role parent; this parent is not every rendered identity.
Bias follows [response conventions](CONVENTIONS.md#6-response-estimators) and
[the m definition](CONVENTIONS.md#7-multiplicative-bias):
`100*(R_sim/R_model−1)` percent.

| Combination | Bias (%) | Case-bootstrap SE (percentage points) |
| --- | ---: | ---: |
| F11 + C17 + disk control | +0.285858 | 0.265609 |
| F8 + retained C17 + disk | +0.138352 | 0.266521 |
| **V3.9-like: F8 + primary C8 + disk** | **+0.020770** | **0.265152** |
| F8 + replicate C8 + disk | +0.062423 | 0.265272 |

V3.9-like's bootstrap 95% interval is [−0.502131,+0.533644]%. The C8 comparison
uses 10,000 paired case resamples, seed20260925; the earlier flow comparison
uses bootstrap seed20260924. These errors exclude training and finite-draw
uncertainty. Point estimates are within 0.3%; robust absolute 0.3% calibration
is not established.

C8 tuning BCE is 0.193621 versus 0.183038 for its matched C17 control, and
constant-shear BCE/Brier also worsen. C12 was rejected. The owner retained C8
for its simpler shared input and observed shear calibration despite broader
failed checks. The absolute sum of magnitude-region response contributions
remains 3.812848 percentage points for primary C8, indicating substantial
cancellation. The emulator still has a true-magnitude-limited training domain.
These reused development scenes and g1-only results do not establish robustness
to population changes or untouched two-axis performance.

Original machine-readable receipts are preserved under
[`doc/evidence/v3_9_like/`](evidence/v3_9_like/); their hashes and external source
paths are in the model manifest. Detailed [experiment history](../archive/research-2026-09-25/doc/V36_FEATURE_SIMPLIFICATION.md#owner-retains-c8-provisionally-emulator-inputs)
and [model comparison](../archive/research-2026-09-25/doc/V36_MODEL_COMPARISON.md)
remain available. No scientific result or old decision was overwritten.

## Measured |e| < 0.7 cut on ConstGold (2026-10-05)

The owner asked whether V3.9-like still gives an unbiased m on ConstGold when the common survey cut on measured
|e| < 0.7 is added, both with C8 and with the simulation's true usable flags in place of C8. It does not.
Nothing was trained or tuned; ConstGold stays test-only.

Setup (study `$DATA_DIR/../sbsi_caches/v39_constgold_ecut07_20261005/`):
- Same cases (40–119), legs, 64 draws, seeds, batching and main cut as
  [the development evidence](#development-evidence). The new cut is added per leg on the measured shape:
  - simulation side: ngmix |e| < 0.7;
  - model side: |e| < 0.7 of the F8 draw after the Möbius disk shift (the model's measured shape).
- `f8_e07_rows.py` rebuilds every F8 draw on A40 GPUs, as in the saved run. The saved rows are reproduced
  exactly (max difference 0).
- `c8_probabilities.py` recomputes the C8 probabilities and checks them against the saved C8 sums.
- `summary.py` reproduces the published +0.020770% ± 0.265152 exactly before reporting anything new. It uses the
  same 10,000-resample case bootstrap, seed 20260925.
- The cut removes 1,167,947 of 7,264,215 selected simulation galaxies (16.1%) in the plus leg.

| weights on the model side | without the cut | with \|e\| < 0.7 | change from the cut |
| --- | ---: | ---: | ---: |
| C8 (V3.9-like) | +0.021% ± 0.265 | **−6.384% ± 0.352** | −6.405 ± 0.261 pp |
| true usable flags | +0.228% ± 0.266 | **−6.248% ± 0.355** | −6.476 ± 0.261 pp |

The C8 95% interval with the cut is [−7.08, −5.70]%.

- With the cut, the simulated response is 0.46442. The model predicts 0.49609 (C8) or 0.49537 (true flags), even
  though the selected counts agree to 0.1%.
- The classifier is not the cause: the same bias appears with the true flags.
- The cause is the F8 flow + disk response. They get the overall response right, but not how shear moves
  measured shapes across |e| = 0.7, i.e. the shape distribution and its shear response near that boundary.
- The same cut on the model-generated paired mock is unbiased
  ([−1.3% ± 4.4%](HYBRID_SAMPLING.md#paired-mock-with-a-measured-e--07-cut-m--13--44-2026-10-05)),
  because that mock comes from the model itself.
- Limitations of this check: g1 only, development cases only, 80 cases.

## Loading and inference boundary

Set `SBSI_CACHE_DIR` and `BLENDEMU_RUNS_DIR` to the external artifact roots:

```bash
python -m sbsi show-model V3.9-like
python -m sbsi validate-model V3.9-like
```

```python
from sbsi.models import get_model, load_detection_classifier, load_disk_response
from sbsi.measurement_model import load_measurement_model

paths = get_model("V3.9-like")
paths.validate()
flow = load_measurement_model(paths.flow_checkpoints[0])
classifier = load_detection_classifier(paths)
disk = load_disk_response(paths)
# Supply a frame with the eight exact named features above.
# Classifier probability and joint flow samples use their own preprocessors.
```

Model loading/prediction supports these artifacts. The default additive
likelihood is unchanged. The retained `run_disk_inference.py` driver and its
prepared feature stores remain V3.6-specific: they expect legacy flow crowding
and a 17-input classifier. They must not be relabelled or reused for V3.9.
The owner subsequently authorized a separate model-generated sampling study.
Its explicit physical-crowding cache is `sbsi/v39_inference.py`; the study
driver is `scripts/run_v39_sampling_study.py`. This does not convert the older
V3.6 driver or its caches into V3.9 caches. See the
[20k model-closure study](#20k-model-closure-sampling-study).

## 20k model-closure sampling study

Authorized September25 after wrap-up. Generate20,000 retained observations
from the pinned F8/C8/disk composition at g=(0.02,0), drawing atoms from the
28,012,032-row expanded weighted prior. Apply Bernoulli usability and measured
radius>0.6arcsec/magnitude<25.8 selection; no truth-property cuts. The disk
transport uses the same joint radius/flux draw as selection. Since transport
does not change those two outputs, applying it after the radius/flux cut is
equivalent to transporting before selection.

Both inference arms use the same model-generated observations, prior masses,
QMC selection normalizer, full nine-node stencil (h=.001), initial centre
equal to the full mock's mean observed shape, and seed8701 draw stream.
Gaussian re-ranking uses K1024 from a131072 location prefilter. The alternative
uses the nearest1024 directly with the same fixed metric. Both draw16384
times from the unchanged defensive Gaussian mixture. Retain512–16384 prefix
moments, but compare the fixed16384 rung as primary, not adaptive stopping.
Combine all20 disjoint1000-row partitions; report paired uncertainty and
draw-count convergence. A non-positive information matrix is a failed solve,
not a reason to add empirical corrections.

Features are rebuilt from complete scenes, including neighbours absent from
the target subset. New bright rows inherit the original donor environments;
their crowding flux ratios are recalculated for the new target flux. Runtime,
feature, model and data hashes are pinned separately under external root
`inference_v39_mock20k_distance_20260925_v1`. The old V3.6 caches stay intact.
Current status and Slurm receipts are in the
[work log](WORKLOG.md#2026-09-25--start-v39-model-generated-20k-sampling-comparison).

This is numerical model closure, not image-simulation validation. A single
20k mock, one proposal seed, one Newton step and64 QMC selection draws cannot
establish robust absolute0.3% shear calibration. Production defaults remain
unchanged; the completed comparison below does not support adoption.

### Completed 20k comparison

All20 partitions and strict combination completed on September25. Retry
16699600 recovered only the two timed-out cip partitions17/19; combine16697082
verified every partition's execution identity, source hashes and row coverage.
Primary results use all20000 observations and the fixed16384-draw rung:

| Exact-set ranking | Estimated g1 | Robust SE of g1 | Relative error against injected0.02 |
| --- | ---: | ---: | ---: |
| Gaussian | 0.02215861 | 0.00097257 | +10.79% ±4.86 percentage points |
| Distance-only | 0.01864592 | 0.00080721 | −6.77% ±4.04 percentage points |

Uncertainties are one-standard-error influence-function estimates across
observations, not repeated-seed numerical-error estimates. The paired
(distance minus Gaussian) change is −0.00351269 ±0.00088132 in g1,
or −17.56 ±4.41 percentage points relative to injected shear. This isolates
the change of exact-set selection within this fixed realization; it does
not establish either algorithm's repeated-realization bias.

| Random draws | Gaussian relative error (%) | Distance-only relative error (%) |
| ---: | ---: | ---: |
| 512 | +4.83 | −43.63 |
| 1024 | +6.29 | −38.95 |
| 2048 | +13.26 | −32.19 |
| 4096 | +14.08 | −24.21 |
| 8192 | +10.64 | −16.24 |
| 16384 | +10.79 | −6.77 |

The distance-only estimate changes by9.47 percentage points from8192 to16384;
it is not numerically converged. Gaussian's last two g1 estimates are closer,
but that alone does not establish convergence or correct model closure.
Neither arm demonstrates0.3% accuracy, and distance-only is not adopted.
The earlier three-observation centre-numerator gains did not establish
accurate shear derivatives across this broader V3.9 mock population.

Next diagnostic should compare exact finite-prior scores/curvatures for a
fixed subset against both samplers, then separate integration error from
one-step/stencil and selection-normalizer error. Those causes have not yet
been individually identified; no new large run is implied by this result.
The full machine-readable result is external `result.json` under the study
root above; all prefixes and failed-run artifacts remain preserved.

### Fresh100k distance-only follow-up

Owner authorized a100k run with distance-only ranking after inspecting the
20k comparison. External root `inference_v39_mock100k_distance_20260925_v1`
contains the new immutable protocol and execution pins. Generate a fresh
realization using scene/detection/flow seeds202609261/202609262/202609263,
with the same V3.9 models, expanded weighted prior, injected shear(.02,0),
measurement cuts and no truth cuts. This is not a nested extension of20k.
The new mean observed shape sets the centre. Reuse verified physical input
features and zero-shear proposal moments; rebuild all selection/classifier
nodes around that centre. Reject the run on a retained numerically invalid
draw without changing seeds, clipping, dropping or replacing the object.

Use only the distance arm, K1024/M16384, all six prefix rungs and full
nine-node stencil.100 fixed1000-row partitions run with a combined six-GPU
cap across cip/inter, then strict complete-population combination. This tests
a larger observation sample at unchanged integration budget; it does not
resolve the observed finite-draw error merely by increasing observation count.
The previous20k result and artifacts remain intact. Current status and job
receipts: [launch record](WORKLOG.md#2026-09-25--launch-fresh100k-distance-only-v39-model-mock).

### Completed100k distance-only result

All100 partitions and combination16699859 completed successfully; no recovery
was needed. For the fresh100000-observation mock at injected g=(.02,0),
K1024/M16384 distance-only inference gives
**g1=0.0195024115 ±0.0007965902**, or **−2.48794% ±3.98295 percentage points**
relative to injected g1 (robust one-standard-error across observations).
g2=0.0003397816 ±0.0003021170. All100 partition identities, input/output hashes
and disjoint row coverage passed the strict combine stage.

| Random draws | Relative g1 error (%) | Robust SE (percentage points) |
| ---: | ---: | ---: |
| 512 | −36.79 | 1.44 |
| 1024 | −33.02 | 1.53 |
| 2048 | −27.53 | 1.89 |
| 4096 | −22.23 | 2.18 |
| 8192 | −14.03 | 2.77 |
| 16384 | −2.49 | 3.98 |

The last doubling moves the estimate by11.54 percentage points, so the
16384-draw estimate remains unconverged despite proximity to injected shear.
Reported uncertainty does not decrease appreciably versus20k (4.04pp), and
increases with draws within100k; these observations call for influence and
integration diagnostics, not an assumed cause or a robust calibration claim.
This fresh mock has a different starting centre and normalizer stencil from
20k; their point estimates are not a paired sample-size comparison.
No0.3% acceptance and no production default change. Next useful work is exact
finite-prior score/curvature comparisons on a fixed subset, with explicit
checks of high-influence observations, before scaling observation count again.
Results are preserved in the100k study root's `result.json`.

### Authorized500k distance-only follow-up

Owner requested500k on September26. Separate root
`inference_v39_mock500k_distance_20260926_v1` uses the unchanged100k driver and
runtime, fresh fixed scene/detection/flow seeds202609271/202609272/202609273,
and500000 measured-selected observations. The F8/C8/disk models, expanded
prior, measured0.6arcsec/25.8 selection without truth cuts, distance-only
K1024/M16384, full draw ladder and one-step estimator are unchanged. Generation
must pass strict retained-shape validity without dropping, clipping, replacing
or changing the predeclared seeds. No result yet.

The measured100k inference wall time was4h41m; unchanged throughput predicts
about24h including preparation for500k. Timing depends strongly on available
GPUs: median1000-row tasks took9.5min on inter versus39.6min on cip. The
combined ceiling remains six GPUs. Increasing observation count is a larger
sample test, not a remedy for the established finite-draw convergence issue.
[Launch record](WORKLOG.md#2026-09-26--start-authorized500k-distance-only-follow-up).

500k generation failed before inference: two retained draws (selected ordinals
86174 and112155, zero-based) round to squared shape norm1.0000000000000002,
outside the numerically invertible open disk. The validity guard stopped
job16702042; no observations were dropped/replaced and no downstream jobs
were launched. The failed draws and fixed seeds are preserved. This requires
a consistent boundary-precision representation in generation and likelihood
before continuing the uncut500k test. It is not a completed inference result.
[Failure audit](WORKLOG.md#2026-09-26--500k-mock-stopped-by-retained-disk-boundary-rounding).

### Disk-boundary precision amendment for500k

Owner authorized fixing the precision failure and resuming with the same
500k seeds. Root `inference_v39_mock500k_distance_20260926_v2` preserves the
failed v1 artifacts and pins the amended implementation separately. Record
`measured_shape_log_disk_gap = log(1−|e|²)` before Cartesian rounding, then
propagate it with the exact Möbius identity
`gap(T(e)) = gap(e)*(1−|t|²)/|1+conj(t)*e|²`. The inverse density recovers
unbounded radius via `atanh(|e|)=log1p(|e|)−log_gap/2`, retaining the physical
Jacobian. This field carries measurement precision, not an intrinsic galaxy
property or an extra stochastic dimension. It cannot be reconstructed from
an already rounded boundary catalogue. No truth-property cut is introduced.
Ordinary catalogues without the metadata retain their original density path.

Regenerated500000 selected observations with exactly unchanged seeds, no
replacements and no clipping. The entire previously failed second batch
replays byte-identical atom IDs and physical draws. Both retained draws that
rounded outside the disk before transport are now valid encoded shapes.
53808 selected targets have true r>=26. Forward/inverse Cartesian difference
is at most4.44e-16, and radial metadata round-trips within the declared tolerance.

Validation:305 tests passed,1 skipped; independent120-digit Decimal radial and
transport checks, forward-Jacobian density and RNG equivalence tests. Frozen
V3.9 GPU preflight verifies interior old/new and eager/compiled densities.
The full500k inference is gated on a sampler audit covering all observations
rounded outside before or after transport, plus ordinary rows. Preparation
used six GPUs across cip/inter. Final inference and convergence results remain
pending; this precision fix does not resolve the earlier finite-draw error.

### Persistent500k continuation and numerical batching

On September26 the owner requested four persistent jobs, one GPU each, with
larger memory-aware batches. The first238000 completed observations are preserved;
array16706357 uses four32GB V100s and a shared queue for the remaining262000.
Models and proposal tables load once per worker; original1000-row checkpoints
remain independently verifiable and are published atomically. Combiner16706358
checks the additional execution/validation receipts before the original strict
coverage and scientific-identity checks.

Accepted execution uses128-observation atom searches,24-observation tilted
proposal blocks, and the original4-observation likelihood groups. Replay on1032
observations, including both known boundary groups, matched every stored moment
and diagnostic exactly, with identical candidates, draws and probabilities.
Peak allocated GPU memory24.23GiB leaves headroom on the31.73GiB allocations.
This is finite-sample execution validation, not a new shear-calibration result.

Larger likelihood batches were rejected: despite identical atom draws, float32
finite differences changed. A held-out1024-row comparison, augmented with the
two boundary groups and mapped through the238k reference information matrix,
estimated a g1-relative shift of+0.097959 percentage points with paired
SE0.068034pp. The predeclared numerical limit including2SE was0.003pp; batches
32,128,160,192 failed. Their raw receipts are retained. Keeping the original
likelihood grouping avoids mixing those numerical changes into the500k result.
See [continuation record](WORKLOG.md#2026-09-26--resume262k-unfinished-rows-on-four-persistent-v100-workers).

### Completed500k model-mock result

All500000 observations completed on September26; strict combiner16706358
passed after the four persistent workers finished the262k continuation in
6h39m49s. At16384 draws per observation, g1=0.020891396494±0.000437371819
and g2=0.001622446128±0.000220339621 (robust1SE), versus injected(0.02,0).
The relative g1 error is+4.456982%±2.186859 percentage points, defined as
100*(g1/0.02−1); see [bias convention](CONVENTIONS.md#7-multiplicative-bias).
Measured radius>0.6arcsec and magnitude<25.8; no truth-property cuts.

| Draws per observation | Relative g1 error | Robust1SE (pp) |
|---:|---:|---:|
|512|−35.380935%|0.796048|
|1024|−31.042642%|0.833733|
|2048|−25.626496%|0.922614|
|4096|−19.323900%|1.089082|
|8192|−9.035359%|1.476396|
|16384|+4.456982%|2.186859|

The draw ladder remains unconverged, and g2 is nonzero. This is not a robust
0.3% inference result or an independent image-simulation calibration. Reported
sampling errors do not bound the remaining draw-budget, one-Newton-step, or
64-draw QMC normalization errors. Scientific settings and previous completed
observations were preserved. See [completion audit](WORKLOG.md#2026-09-26--complete500k-model-mock-inference-draw-ladder-remains-unconverged).
