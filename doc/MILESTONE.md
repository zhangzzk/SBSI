# V3 milestones

V3/V3b frozen on 2026-08-16; V3.1 named on 2026-08-24; V3.2 named on
2026-08-26.

| Release | Flow | Emulator | Detector | True primary domain | `R_sim` | `R_flow` | `R_blend` | ConstGold result |
|---|---|---|---|---|---:|---:|---:|---|
| **V3** | V2.2, 16 SWA seeds | response-weighted Optuna trial 15, narrow-domain fit | legacy BlendEMU | `18 < r < 25.8`, `0.5 < Re < 1.5 arcsec` | 0.962559 | 0.826627 | 0.132608 | `m = +0.3488 +/- 0.1773%`, N=5,642,349 |
| **V3.1** | original E, 4 SWA seeds | exactly the V3 emulator | legacy BlendEMU | `18 < r < 25.8`, `0.5 < Re < 1.5 arcsec` | 0.962559 | 0.827700 | 0.132608 | `m = +0.2343 +/- 0.1504%`, N=5,642,349 |
| **V3.2** | exactly the V3.1 flow | exactly the V3.1 emulator | transition-aware SBSI, `lambda=1` | `18 < r < 25.8`, `0.5 < Re < 1.5 arcsec` | 0.962559 | 0.827700 | 0.132608 | full composition pending |
| **V3b** | V2 `dom6x6`, 16 SWA seeds | same fixed recipe refit on broad V2 domain | legacy BlendEMU | `18 < r < 26.0`, `0.3 < Re < 1.5 arcsec` | 0.860502 | 0.725809 | 0.143064 | `m = -0.9599 +/- 0.2149%`, N=11,674,408 |

The error bars combine flow-seed and case-blocked simulation SEM in quadrature. The
sign convention is always `m = R_sim / (R_flow + R_blend) - 1`.

## Frozen artifact identity

`sbsi.models` provides optional path-only presets and validates their flow
checkpoints plus the emulator hashes. It does not configure the workflow or
choose any catalogue. In compact form:

- V3 flow: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s{seed}_swaavg.pt`
- V3 emulator: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json`, SHA-256 `01decd...c21f`
- V3.1 flow: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s{seed}_swaavg.pt`, seeds 501--504
- V3.1 emulator: exactly the V3 emulator above, including the same SHA-256 pin
- V3.2 flow/emulator: byte-identical to V3.1
- V3.2 detector: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_transition_lambda1_v1/models/transition_aware.pt`, SHA-256 `9966cf...703455`
- V3b flow: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s{seed}_swaavg.pt`
- V3b emulator: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/weighted_model.json`, SHA-256 `3cf70b...553`

V3.1's “original E” means the 80-epoch recipe with matrix-response weight 450,
four million balanced `g=0`/`|g|=0.05` rows, grouped 160/40 case split, and SWA
over epochs 73--80.  It excludes the 100-epoch seed-501 experiment and the
non-SWA epoch-78 diagnostic.  The four flow checkpoint SHA-256 values, in seed
order 501--504, begin `38a76b`, `8ccc22`, `bc212b`, and `11bb8d`.

V3.2's detector was trained on paired `g=0`/forward `|g|=0.05` cases 0--24,
tuned on 25--29, and frozen-tested on 30--39.  It uses sheared `(e1,e2)` for
both the primary and the impact-ranked neighbour, including separation-frame
projections, with no shear, leg ID, measured shape, or shape-response target.
ConstGold cases 40--49 give balanced accuracy 0.906117 and detection response
`-0.024023`, versus truth `-0.019253 +/- 0.000652` (case SEM).  This standalone
transfer validates the sign and approximate scale; it is not an end-to-end
V3.2 shear-calibration result, and its response must not be added directly to
the separate `R_flow + R_blend` table above.

## Interpretation

V3 is the original narrow-domain headline. V3.1 names the original-E four-seed
ensemble with V3's unchanged emulator; its four-seed uncertainty remains less
mature than V3's 16-seed estimate, and its residual cross response is nonzero at
2.77 standard errors. V3.2 changes only the detector used by the catalogue
likelihood and is the preferred composition for subsequent inference tests;
its full-composition closure remains to be measured. V3b is a named broad-domain transfer milestone, not an
improvement over its older emulator baseline: the frozen broader emulator
overshoots ConstGold even though its retrospective half-shear vector slope is
`1.0077 +/- 0.0040`. The names identify compositions and provenance; they do
not imply equal accuracy.

## Inference status

The four-output flows can be evaluated as measurement likelihoods. The full
scene-prior, detection, selection-normalization, and catalogue-level
simulation-based shear estimator remain pending; `INFERENCE.md` states this
boundary.

## Deferred BlendEMU work

The latest emulator models are frozen here as external artifacts. Their training and
tuning implementation is intentionally not promoted into SBSI. The next emulator update
should be made in BlendEMU, after which only the optional model-path preset and
external derived products need updating here.
