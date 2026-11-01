# GalSBI re-simulation plan (2026-11-01)

Owner request (2026-11-01): rerun the image simulations with the GalSBI population from the start, covering the
training data for F8/C8 and for the R_self/R_blend disk emulator as well as the constant-shear sims. Study how many
sims are needed, how much space there is, and what can be cleaned up. This is a plan only: nothing has been rendered,
deleted or launched.

Why: test (a) (WORKLOG.md, entry "2026-11-01 — Test (a) result") showed that inferring FS2 galaxies with the GalSBI prior costs about −3.3% in m. Simulating with the same population as
the prior removes that mismatch from the whole chain.

Sources: three read-only surveys made on 2026-11-01. Disk numbers are in
`doc/evidence/disk_audit_20261101/`, from `du` on `cluster` (arrays 17640619, 17640620, 17641126).

## What exists now (FS2)

Every case is one ~1 deg² LSST-r image: 18181² px at 0.2″, Moffat PSF with 0.73″ FWHM. It holds 699,568 galaxies
resampled from FS2 at r < 29 and placed at uniform random positions, so FS2 clustering and IA are already dropped.
SExtractor finds about 317k objects per case-leg. Raw FITS images are kept.

| Family | Used for | Cases | Renders per case | CPU-h | Disk |
|---|---|---|---|---|---|
| Half-shear training (`lsst_sims_fs2_25876`, g = 0 / 0.05 secondaries) | F8 (160/40 split), C8 (40–119) | 200 | 2 | ≈ 1,300 | — |
| Neighbour shear (same scenes, g = 0.2 secondaries) | disk emulator (R_self, R_blend), fit 40–199 | 200 | +1 | ≈ 700 | — |
| **Training total** | | 200 | 3 | **≈ 1,900–2,100** | **≈ 1.1 TB** |
| ConstGold (`lsst_sims_fs2_25876_constant`, ±0.02) | tests only | 140 | 2 | **≈ 1,200–1,350** | **≈ 0.5 TB** |

CPU-h come from log wall times × cores booked (±30%), because `sacct` no longer holds the May–July jobs. Rendering
needs about 5 GB per process and no GPU.

## How many sims are needed

- **Training: keep 200 cases.** The current models used 31.4M usable rows per shear leg (F8), 60+20 cases (C8) and
  161M response pairs (emulator). Retraining noise is far below the evaluation error: emulator seed +0.034 pp, flow
  refinement ~0.002 pp, against ±0.265 pp. The only learning curve (July lineage, archived) improved from 40 to 100
  cases and was flat from 100 to 160. Fewer than 200 cases would save little and break the existing splits.
- **Constant shear: 140 cases at ±0.02, as now.**
  - **Forward test:** it reaches ±0.265 pp with 80 cases and scales as 1/√cases, so 140 cases give about ±0.20 pp.
    ±0.1 pp would need about 560 cases, or about 140 cases at g = 0.05.
  - **Paired inference:** it is limited by GPU time, not images. ±0.52% took 1M pairs. ±0.3% needs about 3M pairs,
    roughly 550–650 GPU-h from test (a)'s cost. 140 cases hold about 25M selected galaxies per leg, far more than any
    inference run uses.
- **Optional: an untouched acceptance set.** The conventions require a fresh set before any 0.3% claim, and none
  exists. Rendering 60 extra constant-shear cases and never looking at them during development costs about +550
  CPU-h and +0.2 TB.

| Plan | Cases | CPU-h | Wall time at 500 cores | Disk with images |
|---|---|---|---|---|
| Training + ConstGold | 200 + 140 | 3,100–3,500 | ~7 h plus queueing | 1.6–1.7 TB |
| + acceptance set | + 60 | +550 | +1 h | +0.2 TB |
| Downstream (new caches, emulator pairs, normalization, inference runs) | | GPU, see below | | ~0.5–0.8 TB |
| **Total** | | | | **≈ 2.3–2.7 TB** |

The scheduler allows 3,500 CPUs per user on `cluster`; earlier studies used about 500. After rendering comes
retraining (F8 on GPU, C8, the xgboost emulator on CPU), then the 10.01M-scene selection normalization (GPU), and only
then any inference. That model and inference work, not the rendering, sets the timeline.

## What must change before rendering

1. **A GalSBI catalogue that matches the inference prior exactly.** BlendEMU already has `catalog.type: galsbi`
   (`configs/galsbi_f24*.yaml`, catalogue `cats/galsbi/f24_0_r_ucat.gal.cat`), but it is **not** the SBSI prior:
   - It reads `int_r50` as the circularized radius. SBSI uses r50·(1−a²)^¼, so BlendEMU galaxies come out about 9%
     larger at the median.
   - It cuts at n ∈ [0.5, 6] and Re < 5″. The prior allows n from 0.2 to 10.
   - Its filter is not recorded. The prior is F24 row 0 with HSC r2/i2 photometry and native support at i 14–28.

   Proposed fix: SBSI writes the parent table from `GalSBIContinuousPrior` (`sbsi/galsbi_prior.py`) using these
   mappings:
   - semi-major axis = R·√(1+a)
   - q = √((1−a)/(1+a))
   - position angle = phase/2

   It renders to r ≈ 29 at the prior's galaxy density, and BlendEMU reads the table without cuts. This keeps the
   boundary: SBSI owns the prior, BlendEMU owns rendering. **The owner decides the band:** the models are trained on
   LSST-r while the prior's photometry is HSC r2/i2.
2. **Fold in the known image-side suspects**, because everything is rendered again anyway:
   - Use a noise seed per case. Today all sims appear to share one pixel-noise field ([WORKLOG 2026-10-29](WORKLOG.md)).
   - Use a centred measurement cutout, the suspect for the faint e2 excess. Test it first on a few existing FS2 cases
     (pending task, cheap) so the fix goes into the first measurement.
   - Measure once with subpixel centring and no magnitude cut, for primaries and secondaries alike. Today these needed
     a later remeasurement and the "c2fix" pass.
3. **Constant-shear mode** exists only on BlendEMU `dev` (46107b8). Merge it or pin that commit.
4. **A 10-case pilot before the full run.** Check the rendered input table against direct prior draws (size, n,
   |e|, magnitude, density), then compare detections and measured shapes with GalSBI model mocks.

**Keep the FS2 ConstGold sims.** They are now the out-of-population stress test: the real sky is not GalSBI, and test
(a) showed a −3.3% sensitivity to exactly this.

## Disk: space now and what can go

The project filesystem is shared by the group: 108 of 120 TB used, 13 TB free (90%). This account holds **9.42 TB in
4.39M files**, up from 6.5 TB on 2026-09-18. Almost all the growth is sampler output in `sbsi/catalogue_prior`
(3.98 TB, 2.99M files). Home holds 20 GB and needs nothing. Sizes are space on disk after Lustre compression.

### Tier 1: no conclusion lost (≈ 2.8 TB, ~1.4M files)

| Item | Frees | Why |
|---|---|---|
| `catalogue_prior/galsbi_pair1m_20261012`: `chain.npz`, `local_profiles.npz`, `final_scenes.npz` | 1.89 TB | V3.9 1M closure mock, written up Oct 12–25. Keep result.json and reports (71 G) |
| Same sampler files in the Oct 2–25 GalSBI studies (blind100k 351 G, mock500k 157 G, faintg05, blind10k, twin100k, …) | 0.74 TB | Conclusions in WORKLOG and memory |
| `__pycache__` in `galsbi_concurrency_20261026_v1/_v2` | 67 G, 1.34M files | Generated bytecode |
| Failed, smoke, pilot and REPRO sim dirs; `.oldcats_bak` | ~0.09 TB | Clearly stale |

Tier 1 alone covers the whole GalSBI rerun, so the shared disk does not grow. The 238 G of `.out` logs can also be
compressed.

### Tier 2: owner to confirm (≈ 2.1 TB)

| Item | Size | Note |
|---|---|---|
| `sbsi_caches` dirs referenced nowhere | 0.43 TB | e.g. minimal_rblend_fresh_acceptance 95 G, detection_circularized_repair 61 G, `_QUARANTINE_2026-07-30` 41 G; `derisk` (107 G) is cited, so keep |
| Closed September `inference_*` studies | 0.37 TB | Keep their JSON; `pair_features.npy` alone is 174 G |
| `blendemu_runs/joint_independent_validation_20260919_v1` | 0.32 TB | Cancelled validation; docs say partial products kept |
| `lsst_sims_fs2_25876_constant_g005_c140-239` | 0.32 TB | Referenced nowhere |
| Older catalogue versions in `sbsi_catalogues` | 0.33 TB | ap7, plain ngmix, crowd |
| Remaining anchorblend and v22 sims | 0.34 TB | Archived experiments |

### Tier 3: after the GalSBI sims are validated

- FS2 training-sim images, 1.18 TB; keep their catalogues.
- The unused FS2 g = ±0.02 extra training legs, about 0.16 TB.

Keep FS2 ConstGold.

### Other projects (owner's call)

`proj2_sims` 273 G, `skyvar_bias_variation` 245 G, `cats` 66 G, `benjamin` 41 G (an unpacked copy beside its own
.tgz, 777k files), `kids_dr4` 24 G.

## Decisions for the owner

1. **Band and photometry:** LSST-r, matching the current models, or the prior's HSC r2/i2? This sets the catalogue
   writer.
2. **Run the centred-cutout test first?** Recommended. It decides whether the measurement code changes before the
   rerun.
3. **Add a 60-case untouched acceptance set?**
4. **Approve cleanup Tier 1?** Tier 2 item by item?
