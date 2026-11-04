# GalSBI-dev emulator vs simulation: neighbour-shear response by size and magnitude (2026-11-04)

Written before any output. Owner request 2026-11-04: "go ahead and check the emulator response against the sims".

## Question

The paired end-to-end split (`doc/evidence/galsbi_bands40_constant_20261104/split_by_mag.json`) shows the
GalSBI-dev shape part under-responds more where the emulator carries more of the model response
(cut > 3 px & MAG_AUTO < 25, true r < 24 / 24–25.5 / 25.5–27.5: m in bin +1.0 / +3.9 / +21.1 %, emulator share
5 / 36 / 95 %). If the emulator alone caused this, its neighbour response would be low by roughly
+21% / (0.95) ≈ +22% in the faint bin and +3.9% / 0.36 ≈ +11% in the 24–25.5 bin. This check measures that directly.

## Data (no new simulation, no training)

- Training family `sbsi/galsbi_resim/training`, half-shear: primaries (first half of scene rows) are never sheared;
  secondaries carry |g| = 0.2 at a random angle each. A primary's shape change is caused only by its neighbours.
- Development cases 0–39 only (never used to fit either emulator; the constant-shear family is not touched).
- Pair rows: `blendemu_runs/disk_response_moment_galsbi_cut_20261103/pairs/parts` (+ `prepared/gains`), cases 0–39.
  These cover true r < 30 inside g0 FLUX_RADIUS > 3 px and MAG_AUTO < 25.8, so they include the faint bin the
  GalSBI-dev parts (true r < 26) miss. Same 9 features, same label definition, same gain definition.
- True r of each primary from `case{c}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather` (`r_input`,
  looked up by `index_input` = `primary`; a primary without a truth row stops the run). Amended before any
  output: case 18 skips one input index, so lookup is by index value, not row position.

## Measurement

Per pair: label y = Δe_t/0.2 (primary shape change projected on that neighbour's shear direction);
prediction s·b with b = booster(x)·y_std + y_mean of the GalSBI-dev emulator (`disk_response_moment_galsbi_20261103`,
sha cd32781d…) and s = gain. Since E[y | pair] = s·b, the bin ratio Σy / Σ(s·b) is unbiased for a correct emulator.
Reported per bin: d = 100 (Σy / Σ s·b − 1) [%] (positive = emulator too low), plus the null component
Σnull vs Σ cross_gain·b. The cut-trained variant (`…_cut_20261103`, not in the preset) is evaluated on the same rows
for information only.

Bins: g0 FLUX_RADIUS edges 3, 3.2, 3.5, 3.75, 4.5, 6 px; MAG_AUTO edges 22, 23, 24, 25, 25.8; true r edges 24,
25.5, 27.5. Cut totals: > 3 px & < 25 (target), > 3 px & < 25.8 (main), ≥ 3.75 px & < 25.
Errors: case bootstrap, 10,000 resamples, seed 20260925.

## Reading the result (fixed now)

- Emulator deficit confirmed if, at > 3 px & < 25, true r 25.5–27.5 has d ≥ +10% at > 3σ.
- Emulator cleared as the main cause if that bin's d is within ±5% (2σ). The deficit then has to come from what
  this check does not cover: galaxies near a > 5× brighter neighbour within 3″ (dropped from these pair rows but kept
  in the constant evaluation), the k = 20 neighbour cap, flow-draw instead of measured inputs, the h = 0.2 vs 0.02
  amplitude, or the Möbius transport. A per-galaxy check (measured Δe from catalogues, leg 0.05) would be next.

## Limits

Shape only (no detection or cut migration). Measured g0 size/magnitude, not flow draws. Labels at h = 0.2.
No offsets, scaling, or fitting; nothing here changes any model.
