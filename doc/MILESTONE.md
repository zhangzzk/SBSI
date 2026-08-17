# V3 milestone

Frozen on 2026-08-16.

| Release | Flow | Emulator | True primary domain | `R_sim` | `R_flow` | `R_blend` | ConstGold result |
|---|---|---|---|---:|---:|---:|---|
| **V3** | V2.2, 16 SWA seeds | response-weighted Optuna trial 15, narrow-domain fit | `18 < r < 25.8`, `0.5 < Re < 1.5 arcsec` | 0.962559 | 0.826627 | 0.132608 | `m = +0.3488 +/- 0.1773%`, N=5,642,349 |
| **V3b** | V2 `dom6x6`, 16 SWA seeds | same fixed recipe refit on broad V2 domain | `18 < r < 26.0`, `0.3 < Re < 1.5 arcsec` | 0.860502 | 0.725809 | 0.143064 | `m = -0.9599 +/- 0.2149%`, N=11,674,408 |

The error bars combine flow-seed and case-blocked simulation SEM in quadrature. The
sign convention is always `m = R_sim / (R_flow + R_blend) - 1`.

## Frozen artifact identity

`sbsi.models` provides optional path-only presets and validates all 16 flow
checkpoints plus the emulator hashes. It does not configure the workflow or
choose any catalogue. In compact form:

- V3 flow: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s{seed}_swaavg.pt`
- V3 emulator: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json`, SHA-256 `01decd...c21f`
- V3b flow: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s{seed}_swaavg.pt`
- V3b emulator: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/weighted_model.json`, SHA-256 `3cf70b...553`

The active `results/` directory contains only the two milestone lookup/evaluation
pairs and the V3b half-shear transfer score. Flow response and coupling targets needed
to reproduce training live under `data/calibration/`.

## Interpretation

V3 is the narrow-domain headline. V3b is a named broad-domain transfer milestone, not
an improvement over its older emulator baseline: the frozen broader emulator overshoots
ConstGold even though its retrospective half-shear vector slope is `1.0077 +/- 0.0040`.
Both names identify compositions and provenance; they do not imply equal accuracy.

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
