"""Test whether the flow's self-response error is the anchor truncation.

The fixed-g0 domain truncates the zero-shear leg on the measured anchor and
deliberately leaves the sheared leg uncut, so the measured self response is a
difference between a *selected* zero-shear mean and an unselected sheared mean.
The response-profile builder predicts an unconditional mean at both legs, which
is not the same quantity.

This script predicts the same objects three ways from one set of common random
numbers and puts each next to the measured response:

``unconditional``
    both legs averaged over every draw -- what the builder does today.
``zero_leg_anchored``
    the zero-shear leg averaged over the draws whose own sampled measurement
    passes the anchor, the sheared leg left unconditional -- what the data is.
``both_legs_anchored``
    both legs anchored, i.e. what a recut sheared leg would look like.

Nothing is fitted and no offset is applied: the anchor rule is the domain's own
rule, evaluated on the flow's own samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import (
    FLOW_FEATURES,
    FLOW_TARGETS,
    FLUX_RADIUS_MIN_PIXELS,
    MAG_AUTO_MAX,
    ZERO_POINT,
    matched_key_indices,
)
from sbsi.measurement_model import load_measurement_model
from scripts.build_fixed_g0_response_profile_predictions import (
    load_flow_case,
    projected_response,
)

MAGNITUDE_EDGES = (-np.inf, 23.0, 24.0, 25.0, 26.0, np.inf)
MAGNITUDE_LABELS = ("r<23", "23-24", "24-25", "25-26", "r>=26")
VARIANTS = ("unconditional", "zero_leg_anchored", "both_legs_anchored")


def anchor_flux_minimum() -> float:
    """The flux the anchor's magnitude limit corresponds to."""

    return 10.0 ** ((ZERO_POINT - MAG_AUTO_MAX) / 2.5)


def passes_anchor(draws: torch.Tensor) -> torch.Tensor:
    """The domain's own anchor rule, applied to sampled measurements."""

    radius = draws[..., FLOW_TARGETS.index("measured_flux_radius")]
    flux = draws[..., FLOW_TARGETS.index("measured_flux_from_mag_auto")]
    return (radius > FLUX_RADIUS_MIN_PIXELS) & (flux > anchor_flux_minimum())


@torch.no_grad()
def physical_context_draws(model, contexts, draws: int, seed: int) -> list[torch.Tensor]:
    """``physical_context_means`` without the final average over draws."""

    if not callable(getattr(model, "forward_physical", None)):
        raise ValueError("paired physical response requires an explicit physical flow")
    if not contexts or draws < 2 or draws % 2:
        raise ValueError("at least one context and an even draw count >=2 are required")
    batch = len(contexts[0])
    generator = torch.Generator(device=contexts[0].device).manual_seed(int(seed))
    half = torch.randn(
        batch, draws // 2, model.target_dim,
        dtype=contexts[0].dtype, device=contexts[0].device, generator=generator,
    )
    base = torch.cat((half, -half), dim=1).reshape(batch * draws, model.target_dim)
    sampled = []
    for context in contexts:
        expanded = context[:, None, :].expand(batch, draws, model.context_dim)
        values, _ = model.forward_physical(base, expanded.reshape(batch * draws, model.context_dim))
        sampled.append(values.reshape(batch, draws, model.target_dim))
    return sampled


def masked_shape_mean(draws: torch.Tensor, keep: torch.Tensor | None) -> np.ndarray:
    """Mean sampled shape, over every draw or over the kept draws only."""

    shapes = draws[..., :2]
    if keep is None:
        return shapes.mean(dim=1).cpu().numpy().astype(np.float64)
    weight = keep.to(shapes.dtype)[..., None]
    total = weight.sum(dim=1)
    #  An object whose every draw falls outside the anchor has no conditional
    #  mean at all; it is left non-finite here and counted in the report rather
    #  than being quietly replaced by the unconditional mean.
    mean = (shapes * weight).sum(dim=1) / total
    mean[total[:, 0] == 0] = float("nan")
    return mean.cpu().numpy().astype(np.float64)


@torch.no_grad()
def predict_case(bundle, zero, sheared, *, draws, batch_size, seed) -> pd.DataFrame:
    left, right, _ = matched_key_indices(
        zero["case"], zero["input_index"], sheared["case"], sheared["input_index"]
    )
    gamma = np.asarray(sheared["gamma"][right], dtype=np.float64)
    measured = np.asarray(sheared["target"][right][:, :2], dtype=np.float64)
    measured = measured - np.asarray(zero["target"][left][:, :2], dtype=np.float64)
    keep = np.isfinite(gamma).all(axis=1) & (np.einsum("ij,ij->i", gamma, gamma) > 0.0)
    keep &= np.isfinite(measured).all(axis=1)
    left, right, gamma, measured = left[keep], right[keep], gamma[keep], measured[keep]

    zero_context = bundle.context_tensor(pd.DataFrame(zero["context"][left], columns=FLOW_FEATURES))
    shear_context = bundle.context_tensor(pd.DataFrame(sheared["context"][right], columns=FLOW_FEATURES))
    rows = len(left)
    predicted = {name: np.empty((rows, 2), dtype=np.float64) for name in VARIANTS}
    outside = np.empty((rows, 2), dtype=np.float64)
    empty = np.zeros(rows, dtype=bool)
    for block, start in enumerate(range(0, rows, batch_size)):
        stop = min(start + batch_size, rows)
        zero_draws, shear_draws = physical_context_draws(
            bundle.model,
            [zero_context[start:stop], shear_context[start:stop]],
            draws,
            seed + 1_000_003 * block,
        )
        zero_keep = passes_anchor(zero_draws)
        shear_keep = passes_anchor(shear_draws)
        outside[start:stop, 0] = 1.0 - zero_keep.to(torch.float64).mean(dim=1).cpu().numpy()
        outside[start:stop, 1] = 1.0 - shear_keep.to(torch.float64).mean(dim=1).cpu().numpy()
        empty[start:stop] = (zero_keep.sum(dim=1) == 0).cpu().numpy()
        zero_plain = masked_shape_mean(zero_draws, None)
        shear_plain = masked_shape_mean(shear_draws, None)
        zero_anchored = masked_shape_mean(zero_draws, zero_keep)
        shear_anchored = masked_shape_mean(shear_draws, shear_keep)
        predicted["unconditional"][start:stop] = shear_plain - zero_plain
        predicted["zero_leg_anchored"][start:stop] = shear_plain - zero_anchored
        predicted["both_legs_anchored"][start:stop] = shear_anchored - zero_anchored

    frame = pd.DataFrame(
        {
            "case": zero["case"][left].astype(np.int16),
            "input_index": zero["input_index"][left].astype(np.int64),
            "R_self_measured": projected_response(measured, gamma),
            "r_input_p": np.asarray(zero["context"][left], dtype=np.float64)[
                :, FLOW_FEATURES.index("r_input_p")
            ],
            "zero_draws_outside_anchor": outside[:, 0],
            "shear_draws_outside_anchor": outside[:, 1],
            "no_draw_inside_anchor": empty,
        }
    )
    squared = np.einsum("ij,ij->i", gamma, gamma)
    for name in VARIANTS:
        frame[f"R_self_{name}"] = np.einsum("ij,ij->i", predicted[name], gamma) / squared
    return frame


def summarize(frame: pd.DataFrame, *, replicates: int, seed: int) -> list[dict]:
    """Equal-weight case means per variant, with a case bootstrap on the bias."""

    grouped = frame.groupby("case", sort=True)
    objects = grouped.size().to_numpy(float)
    measured = (grouped["R_self_measured"].sum().to_numpy(float) / objects)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(objects), size=(replicates, len(objects)))
    rows = []
    for name in VARIANTS:
        model = grouped[f"R_self_{name}"].sum().to_numpy(float) / objects
        ratios = 100.0 * (measured[picks].mean(axis=1) / model[picks].mean(axis=1) - 1.0)
        rows.append(
            {
                "variant": name,
                "cases": int(len(objects)),
                "objects": int(objects.sum()),
                "R_self_measured": float(measured.mean()),
                "R_self_model": float(model.mean()),
                "m_percent": float(100.0 * (measured.mean() / model.mean() - 1.0)),
                "m_percent_standard_error": float(ratios.std(ddof=1)),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = tuple(sorted(set(args.case)))
    bundle = load_measurement_model(args.flow, device=torch.device(args.device))
    if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
        raise RuntimeError("the flow was not conditioned on the fixed-g0 feature set")

    frames = []
    for offset, case in enumerate(cases):
        zero = load_flow_case(args.domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        sheared = load_flow_case(args.domain_root / "flow" / "g05" / f"case{case:03d}.npz")
        frames.append(
            predict_case(
                bundle, zero, sheared,
                draws=args.draws, batch_size=args.batch_size,
                seed=args.sampling_seed + 10_007 * offset,
            )
        )
        print(f"case {case:03d} done", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    frame["magnitude"] = pd.cut(
        frame["r_input_p"], bins=list(MAGNITUDE_EDGES), labels=list(MAGNITUDE_LABELS), right=False
    )

    overall = summarize(frame, replicates=2000, seed=args.sampling_seed)
    bands = {}
    for label in MAGNITUDE_LABELS:
        selected = frame[frame["magnitude"] == label]
        if not selected.empty:
            bands[label] = summarize(selected, replicates=2000, seed=args.sampling_seed)

    mass = frame.groupby("magnitude", observed=True)[
        ["zero_draws_outside_anchor", "shear_draws_outside_anchor", "no_draw_inside_anchor"]
    ].mean()

    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print("############ flow probability mass outside the anchor, by magnitude ############")
        print((100.0 * mass).round(3).to_string())
        print("############ overall ############")
        print(pd.DataFrame(overall).to_string(index=False))
        for label, rows in bands.items():
            print(f"############ true primary r {label} ############")
            print(pd.DataFrame(rows).to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "cases": list(cases),
                "draws": args.draws,
                "flow": str(args.flow),
                "anchor": {
                    "mag_auto_max": MAG_AUTO_MAX,
                    "flux_minimum": anchor_flux_minimum(),
                    "flux_radius_min_pixels": FLUX_RADIUS_MIN_PIXELS,
                },
                "mass_outside_anchor_percent": (100.0 * mass).to_dict(),
                "overall": overall,
                "by_magnitude": bands,
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
