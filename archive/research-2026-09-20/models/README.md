# SBSI model identities

Large flow and classifier checkpoints remain external. Repository configs and
`sbsi.models` pin paths, target order, ensemble aggregation, and SHA-256 hashes.
Names are conveniences and never dispatch scientific behavior.

## V3.5-like

`V3.5-like` is the active development model identity:

| Component | Artifact | SHA-256 |
| --- | --- | --- |
| Joint measurement flow | `physical_circularized_grid_free_shape_radius_lambda10_v2/paired/epoch154.pt` | `9c5bb028…25a6f` |
| Blend response model | `constgold50_trial9_v1/trial9_base.json` | `97235898…e46d` |
| Blend response metadata | `constgold50_trial9_v1/metadata.json` | `48908a9c…9f2b4` |
| Coherent-`U` classifier seed 20260913 | `flow8_rblend/seed20260913/selected.pt` | `bbbd27a8…e6685` |
| Coherent-`U` classifier seed 20260914 | `flow8_rblend/seed20260914/selected.pt` | `031498aa…028cd` |
| Coherent-`U` classifier seed 20260915 | `flow8_rblend/seed20260915/selected.pt` | `dab1e4cd…bd702` |

The classifier prediction is the arithmetic mean of the three member
probabilities. Each member has the same nine-feature preprocessor and no
missing-value indicators:

```text
e1_input_p
e2_input_p
sersic_n_input_p
r_input_p
circularized_Re_input_p
nbr_flux_near
nbr_flux_far
nbr_flux_max
R_blend
```

The checkpoint stores the joint density in
`(e1,e2,FLUX_RADIUS,flux-from-MAG_AUTO)` coordinates; magnitude and log-radius
are invertible views of the same two stochastic outputs. The model name does
not authorize dropping either output, replacing the flow selection with
`p(S|x)`, or changing the 0.75-arcsec/no-`|e|` sample. The likelihood config
pins that sample as `measured_flux_radius >= 3.75` pixels and
`measured_flux_from_mag_auto > 47.8630092322638` at zero point 30.

Set `SBSI_CACHE_DIR` to the directory containing the two external roots:

```text
plain_complete_flow_response_re037_v1/
minimal_coherent_u_flow_rblend_ablation_v2/
```

Then inspect or validate the path preset with:

```bash
python -m sbsi show-model V3.5-like
python -m sbsi validate-model V3.5-like
```

The exact likelihood configuration is
`configs/likelihood_v3_5_like.json`. V3.5-like is not the default likelihood
and has not demonstrated 0.3% image-level calibration.

## Bundled artifacts

The small JSON artifacts retained under `models/` support the configured
`v3.2-like` default likelihood. Their full hashes are listed in `SHA256SUMS`.
Historical V3.1--V3.3 path presets and their documentation are preserved under
`archive/research-2026-09-14/` and Git history.
