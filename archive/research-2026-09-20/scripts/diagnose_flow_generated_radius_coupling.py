#!/usr/bin/env python3
"""Test the truth-conditioned flow's response--radius coupling.

The existing radius diagnostic bins an object's unconditional model response
on that object's *catalogue* g=0 FLUX_RADIUS.  That is useful for locating the
catalogue residual, but it is not the conditional curve implied by a
truth-conditioned generative model.  The corresponding model curve must bin
each paired flow draw on that draw's own generated g=0 FLUX_RADIUS.

This script reports both extractions from the same common-random-number draws:

``catalogue_radius_unconditional``
    The old extraction: average the flow response per object, then bin on the
    realised catalogue g=0 radius.
``generated_radius``
    Bin every paired flow draw on its generated g=0 radius.  This tests the
    joint response--radius coupling without adding a measured condition.
``generated_radius_anchor``
    The same generated-radius curve after applying the fixed-g0 magnitude and
    radius anchor to the flow's own g=0 draw.  The difference from the previous
    curve measures leakage across the learned hard boundary.
``generated_radius_anchor_permuted_shear``
    Preserve every per-object draw and mean in both legs, but permute the
    sheared-leg draw order before pairing it with g=0.  Training NLL and the
    paired-mean response loss are exactly invariant to this operation.  A
    changed conditional curve therefore exposes an unidentified cross-leg
    latent copula rather than a learned paired distribution.

The response in every curve is decomposed into its projected g=0 and sheared
shape terms.  Nothing is fitted and no empirical correction is applied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS, matched_key_indices
from sbsi.measurement_model import load_measurement_model
from scripts.build_fixed_g0_response_profile_predictions import load_flow_case
from scripts.diagnose_flow_anchor_truncation import (
    anchor_flux_minimum,
    passes_anchor,
    physical_context_draws,
)


PIXEL_SCALE_ARCSEC = 0.2
DEFAULT_RADIUS_EDGES_ARCSEC = (
    0.6000,
    0.6154,
    0.6298,
    0.6436,
    0.6572,
    0.6706,
    0.6840,
    0.6974,
    0.7108,
)
EXTRACTIONS = (
    "measured_catalogue_radius",
    "measured_catalogue_anchor",
    "catalogue_radius_unconditional",
    "generated_radius",
    "generated_radius_anchor",
    "generated_radius_anchor_permuted_shear",
)
COMPONENTS = ("zero", "sheared", "response")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def radius_labels(edges: tuple[float, ...]) -> tuple[str, ...]:
    """Return explicit left-closed labels, including the fixed-g0 boundary."""

    return tuple(
        [f"[{left:.4f},{right:.4f})" for left, right in zip(edges[:-1], edges[1:])]
        + [f">={edges[-1]:.4f}"]
    )


def radius_bin_indices(values, edges: tuple[float, ...]):
    """Map values to [edge_i, edge_i+1), with -1 below the first edge."""

    if isinstance(values, torch.Tensor):
        boundary = torch.as_tensor(edges, dtype=values.dtype, device=values.device)
        return torch.bucketize(values.contiguous(), boundary, right=True) - 1
    values = np.asarray(values, dtype=np.float64)
    return np.searchsorted(np.asarray(edges), values, side="right") - 1


@dataclass
class CaseAccumulator:
    """Sufficient statistics for one case and all requested extractions."""

    bins: int

    def __post_init__(self):
        self.count = {name: np.zeros(self.bins, dtype=np.float64) for name in EXTRACTIONS}
        self.total = {
            name: {
                component: np.zeros(self.bins, dtype=np.float64)
                for component in COMPONENTS
            }
            for name in EXTRACTIONS
        }
        self.generated_draws = 0
        self.generated_below_radius = 0
        self.generated_fails_flux = 0
        self.generated_passes_anchor = 0

    def add_numpy(
        self,
        name: str,
        bins: np.ndarray,
        values: dict[str, np.ndarray],
        keep: np.ndarray | None = None,
    ):
        valid = (bins >= 0) & (bins < self.bins)
        if keep is not None:
            valid &= np.asarray(keep, dtype=bool)
        local = bins[valid]
        self.count[name] += np.bincount(local, minlength=self.bins)
        for component in COMPONENTS:
            self.total[name][component] += np.bincount(
                local,
                weights=np.asarray(values[component], dtype=np.float64)[valid],
                minlength=self.bins,
            )

    def add_torch(
        self,
        name: str,
        bins: torch.Tensor,
        values: dict[str, torch.Tensor],
        keep: torch.Tensor | None = None,
    ):
        valid = (bins >= 0) & (bins < self.bins)
        if keep is not None:
            valid &= keep
        local = bins[valid]
        count = torch.bincount(local, minlength=self.bins).to(torch.float64)
        self.count[name] += count.cpu().numpy()
        for component in COMPONENTS:
            total = torch.bincount(
                local,
                weights=values[component][valid].to(torch.float64),
                minlength=self.bins,
            )
            self.total[name][component] += total.cpu().numpy()


def projected_components(
    zero_shape: torch.Tensor,
    sheared_shape: torch.Tensor,
    gamma: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return projected leg terms whose difference is the forward response."""

    squared = torch.square(gamma).sum(dim=-1)
    if zero_shape.ndim == 3:
        gamma = gamma[:, None, :]
        squared = squared[:, None]
    zero = (zero_shape * gamma).sum(dim=-1) / squared
    sheared = (sheared_shape * gamma).sum(dim=-1) / squared
    return {"zero": zero, "sheared": sheared, "response": sheared - zero}


@torch.no_grad()
def diagnose_case(
    bundle,
    zero: dict[str, np.ndarray],
    sheared: dict[str, np.ndarray],
    *,
    radius_edges_arcsec: tuple[float, ...],
    draws: int,
    batch_size: int,
    seed: int,
) -> CaseAccumulator:
    """Accumulate catalogue-binned and generated-binned curves for one case."""

    left, right, _ = matched_key_indices(
        zero["case"], zero["input_index"], sheared["case"], sheared["input_index"]
    )
    gamma = np.asarray(sheared["gamma"][right], dtype=np.float64)
    zero_target = np.asarray(zero["target"][left], dtype=np.float64)
    sheared_target = np.asarray(sheared["target"][right], dtype=np.float64)
    keep = np.isfinite(gamma).all(axis=1)
    keep &= np.einsum("ij,ij->i", gamma, gamma) > 0.0
    keep &= np.isfinite(zero_target[:, :3]).all(axis=1)
    keep &= np.isfinite(sheared_target[:, :2]).all(axis=1)
    left, right = left[keep], right[keep]
    gamma = gamma[keep]
    zero_target, sheared_target = zero_target[keep], sheared_target[keep]

    accumulator = CaseAccumulator(len(radius_edges_arcsec))
    catalogue_bins = radius_bin_indices(
        PIXEL_SCALE_ARCSEC * zero_target[:, FLOW_TARGETS.index("measured_flux_radius")],
        radius_edges_arcsec,
    )
    squared = np.einsum("ij,ij->i", gamma, gamma)
    measured_zero = np.einsum("ij,ij->i", zero_target[:, :2], gamma) / squared
    measured_sheared = np.einsum("ij,ij->i", sheared_target[:, :2], gamma) / squared
    accumulator.add_numpy(
        "measured_catalogue_radius",
        catalogue_bins,
        {
            "zero": measured_zero,
            "sheared": measured_sheared,
            "response": measured_sheared - measured_zero,
        },
    )
    catalogue_anchor = zero_target[:, FLOW_TARGETS.index("measured_flux_radius")] > 3.0
    catalogue_anchor &= (
        zero_target[:, FLOW_TARGETS.index("measured_flux_from_mag_auto")]
        > anchor_flux_minimum()
    )
    accumulator.add_numpy(
        "measured_catalogue_anchor",
        catalogue_bins,
        {
            "zero": measured_zero,
            "sheared": measured_sheared,
            "response": measured_sheared - measured_zero,
        },
        keep=catalogue_anchor,
    )

    zero_context = bundle.context_tensor(
        pd.DataFrame(zero["context"][left], columns=FLOW_FEATURES)
    )
    sheared_context = bundle.context_tensor(
        pd.DataFrame(sheared["context"][right], columns=FLOW_FEATURES)
    )
    for block, start in enumerate(range(0, len(left), batch_size)):
        stop = min(start + batch_size, len(left))
        zero_draws, sheared_draws = physical_context_draws(
            bundle.model,
            [zero_context[start:stop], sheared_context[start:stop]],
            draws,
            seed + 1_000_003 * block,
        )
        local_gamma = torch.as_tensor(
            gamma[start:stop], dtype=zero_draws.dtype, device=zero_draws.device
        )
        components = projected_components(
            zero_draws[..., :2], sheared_draws[..., :2], local_gamma
        )
        permutation = np.random.default_rng(
            seed + 9_000_001 + 1_000_003 * block
        ).permutation(draws)
        permuted_sheared = sheared_draws[
            :, torch.as_tensor(permutation, device=sheared_draws.device), :2
        ]
        permuted_components = projected_components(
            zero_draws[..., :2], permuted_sheared, local_gamma
        )
        # This is the central invariance check: only the cross-leg draw pairing
        # changes.  Both leg means, and hence the response loss used in
        # training, remain bitwise-equivalent up to summation order.
        if not torch.allclose(
            components["response"].mean(dim=1),
            permuted_components["response"].mean(dim=1),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError("sheared-draw permutation changed a per-object mean")

        object_means = {
            component: values.mean(dim=1).cpu().numpy()
            for component, values in components.items()
        }
        accumulator.add_numpy(
            "catalogue_radius_unconditional",
            catalogue_bins[start:stop],
            object_means,
        )

        generated_radius = (
            PIXEL_SCALE_ARCSEC
            * zero_draws[..., FLOW_TARGETS.index("measured_flux_radius")]
        )
        generated_bins = radius_bin_indices(generated_radius, radius_edges_arcsec)
        anchor = passes_anchor(zero_draws)
        accumulator.add_torch("generated_radius", generated_bins, components)
        accumulator.add_torch(
            "generated_radius_anchor", generated_bins, components, keep=anchor
        )
        accumulator.add_torch(
            "generated_radius_anchor_permuted_shear",
            generated_bins,
            permuted_components,
            keep=anchor,
        )
        accumulator.generated_draws += int(generated_radius.numel())
        accumulator.generated_below_radius += int((generated_bins < 0).sum().item())
        radius_ok = generated_bins >= 0
        accumulator.generated_fails_flux += int((radius_ok & ~anchor).sum().item())
        accumulator.generated_passes_anchor += int(anchor.sum().item())
    return accumulator


def summarize_cases(
    cases: dict[int, CaseAccumulator], labels: tuple[str, ...]
) -> list[dict]:
    """Equal-weight case means and paired case errors for each radius bin."""

    rows = []
    case_ids = sorted(cases)
    for index, label in enumerate(labels):
        for extraction in EXTRACTIONS:
            count = np.array([cases[case].count[extraction][index] for case in case_ids])
            valid = count > 0
            component_case = {
                component: np.array(
                    [cases[case].total[extraction][component][index] for case in case_ids]
                )[valid]
                / count[valid]
                for component in COMPONENTS
            }
            response = component_case["response"]
            entry = {
                "radius_bin": label,
                "extraction": extraction,
                "cases": int(valid.sum()),
                "samples": int(count.sum()),
                "zero_term": float(component_case["zero"].mean()),
                "sheared_term": float(component_case["sheared"].mean()),
                "R_self": float(response.mean()),
                "R_self_case_sem": (
                    float(response.std(ddof=1) / np.sqrt(len(response)))
                    if len(response) > 1
                    else float("nan")
                ),
            }
            reference = (
                "measured_catalogue_anchor"
                if extraction.startswith("generated_radius_anchor")
                else "measured_catalogue_radius"
            )
            reference_count = np.array(
                [cases[case].count[reference][index] for case in case_ids]
            )
            if not extraction.startswith("measured_catalogue_") and (
                valid & (reference_count > 0)
            ).all():
                measured_case = np.array(
                    [
                        cases[case].total[reference]["response"][index]
                        for case in case_ids
                    ]
                ) / reference_count
                difference = response - measured_case
                entry.update(
                    model_minus_measured=float(difference.mean()),
                    model_minus_measured_case_sem=(
                        float(difference.std(ddof=1) / np.sqrt(len(difference)))
                        if len(difference) > 1
                        else float("nan")
                    ),
                )
            rows.append(entry)
    return rows


def summarize_postcut(cases: dict[int, CaseAccumulator]) -> list[dict]:
    """Equal-weight case means after the hard generated/catalogue radius cut."""

    rows = []
    case_ids = sorted(cases)
    for extraction in EXTRACTIONS:
        count = np.array(
            [cases[case].count[extraction].sum() for case in case_ids]
        )
        valid = count > 0
        component_case = {
            component: np.array(
                [cases[case].total[extraction][component].sum() for case in case_ids]
            )[valid]
            / count[valid]
            for component in COMPONENTS
        }
        response = component_case["response"]
        entry = {
            "extraction": extraction,
            "cases": int(valid.sum()),
            "samples": int(count.sum()),
            "zero_term": float(component_case["zero"].mean()),
            "sheared_term": float(component_case["sheared"].mean()),
            "R_self": float(response.mean()),
            "R_self_case_sem": (
                float(response.std(ddof=1) / np.sqrt(len(response)))
                if len(response) > 1
                else float("nan")
            ),
        }
        reference = (
            "measured_catalogue_anchor"
            if extraction.startswith("generated_radius_anchor")
            else "measured_catalogue_radius"
        )
        reference_count = np.array(
            [cases[case].count[reference].sum() for case in case_ids]
        )
        if not extraction.startswith("measured_catalogue_") and (
            valid & (reference_count > 0)
        ).all():
            measured_case = np.array(
                [
                    cases[case].total[reference]["response"].sum()
                    for case in case_ids
                ]
            ) / reference_count
            difference = response - measured_case
            entry.update(
                model_minus_measured=float(difference.mean()),
                model_minus_measured_case_sem=(
                    float(difference.std(ddof=1) / np.sqrt(len(difference)))
                    if len(difference) > 1
                    else float("nan")
                ),
            )
        rows.append(entry)
    return rows


def histogram_rows(
    cases: dict[int, CaseAccumulator], labels: tuple[str, ...]
) -> list[dict]:
    """Compare catalogue and generated g0-radius probability masses."""

    rows = []
    for extraction in (
        "measured_catalogue_radius",
        "measured_catalogue_anchor",
        "generated_radius",
        "generated_radius_anchor",
    ):
        fractions = []
        counts = []
        for case in sorted(cases):
            count = cases[case].count[extraction]
            counts.append(count)
            fractions.append(count / count.sum())
        fractions = np.stack(fractions)
        counts = np.stack(counts)
        for index, label in enumerate(labels):
            rows.append(
                {
                    "radius_bin": label,
                    "extraction": extraction,
                    "samples": int(counts[:, index].sum()),
                    "case_mean_fraction": float(fractions[:, index].mean()),
                    "case_sem_fraction": float(
                        fractions[:, index].std(ddof=1) / np.sqrt(len(fractions))
                    ),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--radius-edge-arcsec", type=float, action="append", default=None
    )
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cases = tuple(sorted(set(args.case)))
    edges = tuple(args.radius_edge_arcsec or DEFAULT_RADIUS_EDGES_ARCSEC)
    if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("at least two unique, increasing finite radius edges required")
    if not np.isclose(edges[0], PIXEL_SCALE_ARCSEC * 3.0):
        raise ValueError("the first radius edge must be the fixed-g0 3-pixel boundary")
    if args.draws < 2 or args.draws % 2:
        raise ValueError("an even draw count >=2 is required")

    bundle = load_measurement_model(args.flow, device=torch.device(args.device))
    if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
        raise RuntimeError("the flow was not conditioned on the fixed-g0 truth features")

    accumulated = {}
    for offset, case in enumerate(cases):
        zero = load_flow_case(args.domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        sheared = load_flow_case(
            args.domain_root / "flow" / "g05" / f"case{case:03d}.npz"
        )
        accumulated[case] = diagnose_case(
            bundle,
            zero,
            sheared,
            radius_edges_arcsec=edges,
            draws=args.draws,
            batch_size=args.batch_size,
            seed=args.sampling_seed + 10_000_019 * offset,
        )
        print(f"case {case:03d} done", flush=True)

    labels = radius_labels(edges)
    total_draws = sum(value.generated_draws for value in accumulated.values())
    below = sum(value.generated_below_radius for value in accumulated.values())
    flux = sum(value.generated_fails_flux for value in accumulated.values())
    passed = sum(value.generated_passes_anchor for value in accumulated.values())
    result = {
        "format_version": 1,
        "purpose": "truth-conditioned flow response versus its own generated g0 radius",
        "flow": str(args.flow.resolve()),
        "domain_root": str(args.domain_root.resolve()),
        "cases": list(cases),
        "draws": args.draws,
        "sampling_seed": args.sampling_seed,
        "radius_edges_arcsec": list(edges) + [None],
        "rows": summarize_cases(accumulated, labels),
        "postcut_rows": summarize_postcut(accumulated),
        "histogram": histogram_rows(accumulated, labels),
        "generated_anchor_accounting": {
            "draws": total_draws,
            "below_3px": below,
            "below_3px_fraction": below / total_draws,
            "radius_ok_but_flux_fails": flux,
            "radius_ok_but_flux_fails_fraction": flux / total_draws,
            "passes_anchor": passed,
            "passes_anchor_fraction": passed / total_draws,
        },
        "notes": [
            "All model curves condition only on the declared fixed-g0 truth features.",
            "Generated-radius curves use common latent draws across g0 and g05.",
            "Generated-radius-anchor curves are compared with catalogue rows passing the same hard g0 radius and magnitude cuts.",
            "The permuted-shear curve preserves every leg's draws and per-object mean; only the untrained cross-leg latent pairing changes.",
            "The first bin is explicitly [0.6000,0.6154) arcsec, not all radii below 0.6154.",
            "Uncertainty is the standard error over equal-weight case means; it includes case scatter and finite flow draws.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False) + "\n")
    table = pd.DataFrame(result["rows"])
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(table.to_string(index=False), flush=True)
    print(json.dumps(result["generated_anchor_accounting"], indent=2), flush=True)
    print(f"FLOW_GENERATED_RADIUS_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
