"""Is the mean head the conditional mean?  (Answer: NO -- its LEVEL is unidentified.)

WHY THIS EXISTS.  `ConditionalMeanFlow` is documented as `p(x|c) = p_resid(x - mu(c) | c)`, which
reads as "mu learns the conditional mean, the flow learns the re-centred scatter".  That reading is
WRONG and would mislead anyone who plots `mu` expecting a prediction: on the fiducial checkpoint
<mu(g2)> = +5.01 in standardized units (= +1.79 in raw ellipticity, outside the physical range),
exactly cancelled by a -5.00 mean in the residual flow.

THE CAUSE IS IDENTIFIABILITY, NOT A BUG.  Nothing in training pins the LEVEL of mu:
  * the response pin constrains only DIFFERENCES mu(c+) - mu(c-) (a derivative; any constant cancels);
  * the NLL constrains only the SUM mu + residual.
So mu's level is free to drift and the flow absorbs it.  This is harmless for the graded number
because the residual flow is BLIND to the shape features, hence
      d<x>/d(shape)  ==  d mu/d(shape)        exactly, whatever mu's level,
which is the identity the whole response calculation rests on.  What it does mean:

  **Only DIFFERENCES of mu are interpretable.  Never read mu (or its per-object outputs) as a
  predicted mean, and never plot it as one.**  Code that needs the actual model mean must sample the
  full model -- `MeasurementModelBundle.target_mean_and_gradient` correctly does (`model.sample`).

Also reported: three checks that the residual flow DID learn real conditional structure (base-space
calibration, held-out log-density vs the checkpoint's recorded value, and a context-shuffle test).

CONTEXT-RECONSTRUCTION GUARD.  Rebuilding the training contexts outside the trainer is easy to get
wrong and fails SILENTLY -- dropping the upper `Re < 1.5` selection cut moved the mean log-density
from -1.37 to -1.3e5 while every array still had the right shape.  This script therefore REFUSES to
report anything until every reconstructed feature matches the statistics stored in the checkpoint.

Login-node safe: CPU only, one record batch, ~10 s.

Usage:
    python -u scripts/diag_meanhead_identifiability.py [--checkpoint PATH] [--rows N]
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from sbs_shear.measurement_model import load_measurement_model
from sbs_shear.paths import catalogue
from sbs_shear.selection_model import TabularPreprocessor

FID_CKPT = (
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/"
    "measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt"
)
TRAIN_CAT = catalogue("det_meas_crowd_conc_g0.0_train_full.feather")

# The trainer derives e1/e2_input_p by renaming the rot0 columns (preprocessing._add_shape_components).
RENAME = {"e1_input_rot0_p": "e1_input_p", "e2_input_rot0_p": "e2_input_p"}
TARGET_RAW = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto", "measured_flux_radius"]

# The fiducial training domain, as recorded in the checkpoint's `selection_cuts` / CLI flags.
# The Re UPPER bound is the easy one to forget and the one that silently corrupts the contexts.
DOMAIN = {"r_input_p": (18.0, 26.0), "Re_input_p": (0.3, 1.5)}


def load_rows(catalogue_path, preprocessor, target_names, batch_index=0):
    """One record batch of the training catalogue, cut to the fiducial domain."""
    raw = [k for k, v in RENAME.items()] + [
        n for n in preprocessor.feature_names if n not in RENAME.values()
    ]
    with pa.memory_map(str(catalogue_path), "r") as src:
        reader = ipc.open_file(src)
        frame = reader.get_batch(batch_index).select(raw + TARGET_RAW).to_pandas()
    frame = frame.rename(columns=RENAME)
    for column, (lo, hi) in DOMAIN.items():
        frame = frame[(frame[column] > lo) & (frame[column] < hi)]
    frame = frame.dropna().reset_index(drop=True)
    frame["measured_log_flux_radius"] = np.log(frame["measured_flux_radius"])
    frame = frame[np.isfinite(frame["measured_log_flux_radius"])].reset_index(drop=True)
    missing = [n for n in target_names if n not in frame.columns]
    if missing:
        raise ValueError(f"target columns missing from the catalogue: {missing}")
    return frame


def assert_contexts_match_checkpoint(frame, preprocessor):
    """REFUSE unless the rebuilt features reproduce the checkpoint's own standardisation stats.

    A mismatch means the population is not the one the model was fitted on, and every number below
    would be an out-of-distribution extrapolation reported as if it were a model property.
    """
    bad = []
    print(f"{'feature':<20}{'ckpt mean':>12}{'data mean':>12}{'ckpt scale':>12}{'data sd':>12}   ")
    for i, name in enumerate(preprocessor.feature_names):
        c_mean, c_scale = float(preprocessor.means[i]), float(preprocessor.scales[i])
        d_mean, d_sd = float(frame[name].mean()), float(frame[name].std())
        ok = abs(c_mean - d_mean) < 0.25 * c_scale and 0.5 < d_sd / c_scale < 2.0
        print(f"{name:<20}{c_mean:>12.4f}{d_mean:>12.4f}{c_scale:>12.4f}{d_sd:>12.4f}   "
              f"{'OK' if ok else '<-- MISMATCH'}")
        if not ok:
            bad.append(name)
    if bad:
        raise SystemExit(
            f"\nREFUSING: reconstructed contexts do not match the checkpoint for {bad}. "
            "The domain cuts or the feature derivation are wrong; fix them before reading any "
            "number below (see the module docstring)."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=FID_CKPT)
    parser.add_argument("--catalogue", default=TRAIN_CAT)
    parser.add_argument("--rows", type=int, default=0, help="0 = all in-domain rows in the batch")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    preprocessor = TabularPreprocessor.from_state(checkpoint["condition_preprocessor"])
    transform = checkpoint["target_transform"]
    names = list(transform["target_names"])
    bundle = load_measurement_model(args.checkpoint)
    model = bundle.model if hasattr(bundle, "model") else bundle
    model.eval()

    frame = load_rows(args.catalogue, preprocessor, names)
    if args.rows:
        frame = frame.iloc[: args.rows]
    print(f"checkpoint : {args.checkpoint}")
    print(f"in-domain rows: {len(frame):,}\n")
    assert_contexts_match_checkpoint(frame, preprocessor)

    context = torch.as_tensor(preprocessor.transform_frame(frame), dtype=torch.float32)
    means = np.asarray(transform["means"])
    scales = np.asarray(transform["scales"])
    targets = np.stack([frame[n].to_numpy() for n in names], axis=1)
    x_std = torch.as_tensor((targets - means) / scales, dtype=torch.float32)

    with torch.no_grad():
        mu = model._mu(context)
        residual = x_std - mu
        log_prob = model.flow.log_prob(residual, model._flow_ctx(context))
        z, logdet = model.flow.inverse(residual, model._flow_ctx(context))

    n = len(frame)
    print("\n--- 1. IS mu THE CONDITIONAL MEAN?  (standardized target units) ---")
    print(f"  {'target':<26}{'<mu>':>10}{'<data>':>10}{'<resid>':>12}{'SEM':>8}"
          f"{'sd(resid)':>11}{'sd(data)':>10}")
    for j, name in enumerate(names):
        sem = float(residual[:, j].std()) / np.sqrt(n)
        print(f"  {name:<26}{mu[:, j].mean():>10.4f}{x_std[:, j].mean():>10.4f}"
              f"{residual[:, j].mean():>12.4f}{sem:>8.4f}"
              f"{residual[:, j].std():>11.4f}{x_std[:, j].std():>10.4f}")
    print("  => a large <mu> cancelled by an equal and opposite <resid> means mu's LEVEL has "
          "drifted;\n     it is unidentified by construction. Only DIFFERENCES of mu are meaningful.")

    print("\n--- 2. SHAPE-BLINDNESS: the response identity that makes the drift harmless ---")
    bumped = context.clone()
    bumped[:, 0] += 0.5  # intrinsic e1, a dim the residual flow cannot see
    with torch.no_grad():
        d_mu = (model._mu(bumped) - model._mu(context))[:, 0]
        d_logp = (model.flow.log_prob(residual, model._flow_ctx(bumped)) - log_prob).abs().max()
    print(f"  perturb intrinsic e1 by +0.5 sigma:")
    print(f"    mean head mu(g1) shifts by      {d_mu.mean():+.5f}")
    print(f"    residual log-density shifts by  {float(d_logp):+.5f}   (exactly 0: e1 is not an input)")
    print("  => d<x>/d(shape) == d mu/d(shape) EXACTLY, whatever mu's level.")

    print("\n--- 3. DID THE RESIDUAL FLOW LEARN REAL STRUCTURE? ---")
    recorded = checkpoint.get("metadata", {}).get("val_log_prob", {})
    print(f"  mean log-density on these rows   {log_prob.mean():+.4f}")
    print(f"  checkpoint-recorded val value    {recorded.get('mean_log_prob', float('nan')):+.4f}"
          "   (independent reproduction)")
    generator = torch.Generator().manual_seed(1)
    perm = torch.randperm(n, generator=generator)
    with torch.no_grad():
        shuffled = model.flow.log_prob(residual, model._flow_ctx(context)[perm])
    print(f"  same rows, SHUFFLED contexts     {shuffled.mean():+.4f}"
          f"   -> conditioning is worth {log_prob.mean() - shuffled.mean():+.1f} nats")
    print(f"  log|det J|                       {logdet.mean():+.3f} +- {logdet.std():.3f} "
          "  (0 would mean an identity flow)")
    zz = z.numpy()
    print("  base-space z, should be ~N(0,1) if the density fits real data:")
    for j, name in enumerate(names):
        excess_kurtosis = float(((zz[:, j] - zz[:, j].mean()) ** 4).mean()
                                / max(zz[:, j].var() ** 2, 1e-12) - 3.0)
        print(f"    {name:<26} sd={zz[:, j].std():.3f}  excess kurtosis={excess_kurtosis:+.2f}")


if __name__ == "__main__":
    main()
