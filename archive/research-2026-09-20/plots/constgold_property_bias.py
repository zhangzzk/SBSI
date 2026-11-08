#!/usr/bin/env python3
"""Measure and plot ConstGold ``m`` across fixed-cohort property bins.

The three response branches are evaluated together on identical, leg-invariant
bins:

``matched_usable``
    The same fixed-g0 identities must have a usable shape on both shear legs.
``actual_usable_flags``
    Each measured and model leg uses that leg's actual usable-shape flag.
``modeled_usable``
    The measured comparator uses the actual per-leg flags while the model is
    weighted by its per-leg classifier probability ``p(U|x)``.

Bin membership is fixed before reading either ConstGold shear leg.  It uses
only the fixed-g0 anchor, its truth/context columns, or the deterministic
``R_blend_abs`` lookup.  This avoids turning the diagnostic into a sheared-leg
selection or bin-migration measurement.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.measurement_model import load_measurement_model
from sbsi.selection_model import load_selection_model_ensemble
from sbsi.shear_map import apply_shear_to_ellipticity
from scripts.evaluate_constgold_fixed_g0_response import (
    BRANCHES,
    load_measured_leg,
    model_means,
    shear_frames,
)


PIXEL_SCALE_ARCSEC = 0.2
ZERO_POINT = 30.0
PROPERTY_ORDER = (
    "true_magnitude",
    "true_size",
    "magnitude_boost",
    "g0_radius",
    "neighbour_flux_max",
    "blend_response_abs",
)
PROPERTY_META = {
    "true_magnitude": {
        "title": "True primary magnitude",
        "xlabel": r"True primary $r$",
        "scale": "linear",
    },
    "true_size": {
        "title": "True circularized size",
        "xlabel": r"True circularized $R_e$ (arcsec)",
        "scale": "log",
    },
    "magnitude_boost": {
        "title": "Fixed-g0 flux boost",
        "xlabel": r"True $r$ $-$ measured MAG_AUTO",
        "scale": "linear",
    },
    "g0_radius": {
        "title": "Fixed-g0 measured size",
        "xlabel": r"Measured FLUX_RADIUS (arcsec)",
        "scale": "log",
    },
    "neighbour_flux_max": {
        "title": "Brightest-neighbour flux",
        "xlabel": r"$F_{\rm nbr,max}/F_{\rm primary}$",
        "scale": "symlog",
    },
    "blend_response_abs": {
        "title": "Absolute blending response",
        "xlabel": r"$\sum_j |R_{{\rm blend},j}|$",
        "scale": "symlog",
    },
}
BRANCH_META = {
    "matched_usable": {
        "label": "Matched usable",
        "color": "#000000",
        "marker": "o",
        "linestyle": "-",
    },
    "actual_usable_flags": {
        "label": "Actual per-leg usable",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "--",
    },
    "modeled_usable": {
        "label": r"Classifier-weighted $p(U|x)$",
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "-.",
    },
}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def load_lookup(paths: tuple[Path, ...], cases: tuple[int, ...]) -> dict[int, dict[str, np.ndarray]]:
    frames = []
    for path in paths:
        frame = pd.read_feather(
            path,
            columns=["case", "input_index", "R_blend", "R_blend_abs"],
        )
        frames.append(frame.loc[frame["case"].isin(cases)].copy())
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(["case", "input_index"]).any():
        raise ValueError("R_blend inputs overlap or contain duplicate keys")
    actual_cases = set(map(int, merged["case"].unique()))
    if actual_cases != set(cases):
        raise ValueError(
            f"R_blend case coverage differs: expected={set(cases)}, actual={actual_cases}"
        )
    values = merged[["R_blend", "R_blend_abs"]].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values[:, 1] + 1e-12 < np.abs(values[:, 0])):
        raise ValueError("invalid signed/absolute R_blend values")
    result = {}
    for case, group in merged.groupby("case", sort=True):
        ordered = group.sort_values("input_index", kind="stable")
        result[int(case)] = {
            "input_index": ordered["input_index"].to_numpy(np.int64),
            "R_blend": ordered["R_blend"].to_numpy(np.float64),
            "R_blend_abs": ordered["R_blend_abs"].to_numpy(np.float64),
        }
    return result


def load_anchor(path: Path, lookup: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        ids = np.asarray(stored["input_index"], dtype=np.int64)
        context = np.asarray(stored["context"], dtype=np.float64)
        target = np.asarray(stored["target"], dtype=np.float64)
        gamma = np.asarray(stored["gamma"], dtype=np.float64)
    if (
        ids.ndim != 1
        or context.shape != (len(ids), len(FLOW_FEATURES))
        or target.shape != (len(ids), 4)
        or gamma.shape != (len(ids), 2)
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(context).all()
        or not np.isfinite(target).all()
        or not np.array_equal(gamma, np.zeros_like(gamma))
    ):
        raise ValueError(f"invalid fixed-g0 anchor {path}")
    order = np.argsort(ids, kind="stable")
    ids, context, target = ids[order], context[order], target[order]
    if not np.array_equal(ids, lookup["input_index"]):
        raise RuntimeError(f"R_blend identities do not equal the anchor in {path}")
    return ids, context, target


def population_mask(
    context: np.ndarray,
    true_magnitude_max: float | None,
) -> np.ndarray:
    """Return the leg-invariant declared-truth population mask."""

    if true_magnitude_max is None:
        return np.ones(len(context), dtype=bool)
    if not math.isfinite(true_magnitude_max):
        raise ValueError("true-magnitude maximum must be finite")
    true_magnitude = context[:, FLOW_FEATURES.index("r_input_p")]
    keep = true_magnitude < true_magnitude_max
    if not keep.any():
        raise ValueError("true-magnitude population cut removes every anchor")
    return keep


def property_values(
    context: np.ndarray,
    target: np.ndarray,
    blend_response_abs: np.ndarray,
) -> dict[str, np.ndarray]:
    flux = target[:, 3]
    if np.any(flux <= 0):
        raise ValueError("fixed-g0 anchor contains non-positive flux")
    measured_magnitude = ZERO_POINT - 2.5 * np.log10(flux)
    result = {
        "true_magnitude": context[:, FLOW_FEATURES.index("r_input_p")],
        "true_size": context[:, FLOW_FEATURES.index("circularized_Re_input_p")],
        "magnitude_boost": (
            context[:, FLOW_FEATURES.index("r_input_p")] - measured_magnitude
        ),
        "g0_radius": target[:, 2] * PIXEL_SCALE_ARCSEC,
        "neighbour_flux_max": context[:, FLOW_FEATURES.index("nbr_flux_max")],
        "blend_response_abs": np.asarray(blend_response_abs, dtype=np.float64),
    }
    for name, values in result.items():
        if values.ndim != 1 or len(values) != len(context) or not np.isfinite(values).all():
            raise ValueError(f"invalid property {name}")
    if np.any(result["true_size"] <= 0) or np.any(result["g0_radius"] <= 0):
        raise ValueError("size properties must be positive")
    if np.any(result["neighbour_flux_max"] < 0) or np.any(result["blend_response_abs"] < 0):
        raise ValueError("crowding properties lie outside their physical support")
    return result


def quantile_profile(values: np.ndarray, bins: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or bins < 3:
        raise ValueError("finite one-dimensional values and at least three bins required")
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 4:
        raise ValueError("quantile edges collapse to fewer than three populated bins")
    assignment = np.searchsorted(edges[1:-1], values, side="right")
    centers = np.asarray(
        [np.median(values[assignment == index]) for index in range(len(edges) - 1)]
    )
    counts = np.bincount(assignment, minlength=len(centers))
    if np.any(counts == 0) or not np.isfinite(centers).all():
        raise RuntimeError("quantile construction produced an empty bin")
    return {
        "edges": edges,
        "centers": centers,
        "counts": counts,
    }


def symlog_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0]
    if not len(positive):
        raise ValueError("symlog coordinates require at least one positive value")
    return max(float(positive.min()) / 2.0, 1e-4)


def display_coordinate(
    values: np.ndarray,
    scale: str,
    *,
    linthresh: float | None = None,
    inverse: bool = False,
) -> np.ndarray:
    """Map values to a coordinate suitable for equal-width display histograms."""

    values = np.asarray(values, dtype=np.float64)
    if scale == "linear":
        return values
    if scale == "log":
        if inverse:
            return np.exp(values)
        if np.any(values <= 0):
            raise ValueError("log coordinates require positive values")
        return np.log(values)
    if scale == "symlog":
        if linthresh is None or linthresh <= 0:
            raise ValueError("a positive linthresh is required for symlog coordinates")
        if inverse:
            return linthresh * np.sinh(values)
        return np.arcsinh(values / linthresh)
    raise ValueError(f"unsupported axis scale {scale!r}")


def histogram_edges(profile: dict, bins: int) -> tuple[np.ndarray, float | None]:
    """Choose a robust central plotting range in the panel's display coordinate."""

    if bins < 3:
        raise ValueError("at least three histogram bins required")
    centers = np.asarray(profile["centers"], dtype=np.float64)
    outer = np.asarray([profile["edges"][0], profile["edges"][-1]], dtype=np.float64)
    if len(centers) < 2 or not np.isfinite(centers).all():
        raise ValueError("at least two finite profile centers required")
    scale = str(profile["scale"])
    linthresh = symlog_threshold(centers) if scale == "symlog" else None
    transformed_centers = display_coordinate(centers, scale, linthresh=linthresh)
    transformed_outer = display_coordinate(outer, scale, linthresh=linthresh)
    low = max(
        float(transformed_outer[0]),
        float(transformed_centers[0] - 0.6 * (transformed_centers[1] - transformed_centers[0])),
    )
    high = min(
        float(transformed_outer[1]),
        float(
            transformed_centers[-1]
            + 0.6 * (transformed_centers[-1] - transformed_centers[-2])
        ),
    )
    if not low < high:
        raise RuntimeError("histogram display range collapsed")
    transformed_edges = np.linspace(low, high, bins + 1)
    return (
        display_coordinate(
            transformed_edges,
            scale,
            linthresh=linthresh,
            inverse=True,
        ),
        linthresh,
    )


def build_anchor_histograms(result: dict, bins: int) -> tuple[dict[str, dict], pd.DataFrame]:
    """Re-read fixed anchors and build fine histograms without rerunning the flow."""

    cases = tuple(map(int, result["cases"]))
    lookup_paths = tuple(
        dict.fromkeys(Path(item["path"]).resolve() for item in result["inputs"]["rblend"])
    )
    lookup = load_lookup(lookup_paths, cases)
    true_magnitude_max = result.get("population", {}).get("true_magnitude_max")
    profiles = result["binning"]["properties"]
    histograms = {}
    for name in PROPERTY_ORDER:
        edges, linthresh = histogram_edges(profiles[name], bins)
        histograms[name] = {
            "edges": edges,
            "counts": np.zeros(bins, dtype=np.int64),
            "linthresh": linthresh,
            "total": 0,
        }

    for case in cases:
        anchor_path = Path(result["inputs"]["anchor_pattern"].format(case=case))
        _, context, target = load_anchor(anchor_path, lookup[case])
        keep = population_mask(context, true_magnitude_max)
        context, target = context[keep], target[keep]
        properties = property_values(
            context,
            target,
            lookup[case]["R_blend_abs"][keep],
        )
        for name in PROPERTY_ORDER:
            histograms[name]["counts"] += np.histogram(
                properties[name], bins=histograms[name]["edges"]
            )[0]
            histograms[name]["total"] += len(properties[name])

    rows = []
    for name in PROPERTY_ORDER:
        histogram = histograms[name]
        counts = histogram["counts"]
        maximum = int(counts.max())
        if maximum <= 0:
            raise RuntimeError(f"anchor histogram for {name} is empty")
        histogram["relative_density"] = counts / maximum
        retained = int(counts.sum())
        for index, count in enumerate(counts):
            rows.append(
                {
                    "property": name,
                    "histogram_bin": index,
                    "edge_low": histogram["edges"][index],
                    "edge_high": histogram["edges"][index + 1],
                    "anchor_objects": count,
                    "relative_density": histogram["relative_density"][index],
                    "objects_in_display_range": retained,
                    "total_anchor_objects": histogram["total"],
                    "display_range_fraction": retained / histogram["total"],
                }
            )
    return histograms, pd.DataFrame(rows)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 4 or np.any(edges[1:] <= edges[:-1]):
        raise ValueError("strictly increasing bin edges required")
    if np.any(values < edges[0]) or np.any(values > edges[-1]):
        raise ValueError("property value lies outside the frozen bin edges")
    return np.searchsorted(edges[1:-1], values, side="right")


def binned_sufficient(
    values: np.ndarray,
    weights: np.ndarray,
    assignment: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    assignment = np.asarray(assignment, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or weights.shape != (len(values),)
        or assignment.shape != (len(values),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or np.any((assignment < 0) | (assignment >= bins))
    ):
        raise ValueError("invalid binned sufficient-statistic inputs")
    keep = weights > 0
    if not np.isfinite(values[keep]).all():
        raise ValueError("positive-weight response contains a non-finite value")
    denominator = np.bincount(
        assignment[keep], weights=weights[keep], minlength=bins
    ).astype(np.float64)
    numerator = np.column_stack(
        [
            np.bincount(
                assignment[keep],
                weights=weights[keep] * values[keep, component],
                minlength=bins,
            )
            for component in range(2)
        ]
    )
    if np.any(denominator <= 0) or not np.isfinite(numerator).all():
        raise ValueError("every property bin must carry positive finite weight")
    return denominator, numerator


def empty_statistics(cases: int, bins_by_property: dict[str, int]) -> dict:
    return {
        property_name: {
            branch: {
                source: {
                    direction: {
                        "denominator": np.zeros((cases, bins), dtype=np.float64),
                        "numerator": np.zeros((cases, bins, 2), dtype=np.float64),
                    }
                    for direction in ("plus", "minus")
                }
                for source in ("measured", "model")
            }
            for branch in BRANCHES
        }
        for property_name, bins in bins_by_property.items()
    }


def response_from_arrays(plus: dict, minus: dict, h: float) -> np.ndarray:
    plus_mean = plus["numerator"].sum(axis=0) / plus["denominator"].sum(axis=0)[:, None]
    minus_mean = minus["numerator"].sum(axis=0) / minus["denominator"].sum(axis=0)[:, None]
    return (plus_mean - minus_mean) / (2.0 * h)


def bootstrap_response(
    plus: dict,
    minus: dict,
    case_weights: np.ndarray,
    h: float,
) -> np.ndarray:
    plus_denominator = np.einsum("rc,cb->rb", case_weights, plus["denominator"])
    minus_denominator = np.einsum("rc,cb->rb", case_weights, minus["denominator"])
    plus_numerator = np.einsum("rc,cbj->rbj", case_weights, plus["numerator"])
    minus_numerator = np.einsum("rc,cbj->rbj", case_weights, minus["numerator"])
    if np.any(plus_denominator <= 0) or np.any(minus_denominator <= 0):
        raise RuntimeError("a bootstrap replicate emptied a property bin")
    return (
        plus_numerator / plus_denominator[..., None]
        - minus_numerator / minus_denominator[..., None]
    ) / (2.0 * h)


def summarize_branch(
    branch_stats: dict,
    case_weights: np.ndarray,
    h: float,
) -> dict:
    measured = response_from_arrays(
        branch_stats["measured"]["plus"], branch_stats["measured"]["minus"], h
    )
    model = response_from_arrays(
        branch_stats["model"]["plus"], branch_stats["model"]["minus"], h
    )
    measured_draws = bootstrap_response(
        branch_stats["measured"]["plus"],
        branch_stats["measured"]["minus"],
        case_weights,
        h,
    )
    model_draws = bootstrap_response(
        branch_stats["model"]["plus"],
        branch_stats["model"]["minus"],
        case_weights,
        h,
    )
    m_draws = 100.0 * (measured_draws[..., 0] / model_draws[..., 0] - 1.0)
    return {
        "measured_response": measured,
        "model_response": model,
        "measured_response_standard_error": measured_draws[..., 0].std(
            axis=0, ddof=1
        ),
        "model_response_standard_error": model_draws[..., 0].std(axis=0, ddof=1),
        "m_percent": 100.0 * (measured[:, 0] / model[:, 0] - 1.0),
        "m_standard_error_percentage_points": m_draws.std(axis=0, ddof=1),
        "m_ci95_percent": np.quantile(m_draws, [0.025, 0.975], axis=0).T,
    }


def collapse_bins(branch_stats: dict) -> dict:
    return {
        source: {
            direction: {
                "denominator": values["denominator"].sum(axis=1, keepdims=True),
                "numerator": values["numerator"].sum(axis=1, keepdims=True),
            }
            for direction, values in directions.items()
        }
        for source, directions in branch_stats.items()
    }


def make_case_weights(replicates: int, cases: int, seed: int) -> np.ndarray:
    if replicates < 2 or cases < 2:
        raise ValueError("at least two bootstrap replicates and cases required")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, cases, size=(replicates, cases))
    offsets = cases * np.arange(replicates, dtype=np.int64)[:, None]
    return np.bincount(
        (picks + offsets).ravel(), minlength=replicates * cases
    ).reshape(replicates, cases)


def validate_partitions(statistics: dict) -> None:
    reference = None
    for property_name in PROPERTY_ORDER:
        collapsed = {
            branch: collapse_bins(statistics[property_name][branch]) for branch in BRANCHES
        }
        if reference is None:
            reference = collapsed
            continue
        for branch in BRANCHES:
            for source in ("measured", "model"):
                for direction in ("plus", "minus"):
                    for field in ("denominator", "numerator"):
                        left = reference[branch][source][direction][field]
                        right = collapsed[branch][source][direction][field]
                        np.testing.assert_allclose(left, right, rtol=2e-13, atol=2e-8)


def subset_point_m(
    statistics: dict,
    case_indices: np.ndarray,
    h: float,
) -> dict[str, float]:
    result = {}
    first = statistics[PROPERTY_ORDER[0]]
    for branch in BRANCHES:
        selected = {
            source: {
                direction: {
                    field: values[field][case_indices]
                    for field in ("denominator", "numerator")
                }
                for direction, values in directions.items()
            }
            for source, directions in first[branch].items()
        }
        collapsed = collapse_bins(selected)
        measured = response_from_arrays(
            collapsed["measured"]["plus"], collapsed["measured"]["minus"], h
        )[0, 0]
        model = response_from_arrays(
            collapsed["model"]["plus"], collapsed["model"]["minus"], h
        )[0, 0]
        result[branch] = float(100.0 * (measured / model - 1.0))
    return result


def verify_reference(
    reference_path: Path,
    reference_model: str,
    cases: tuple[int, ...],
    statistics: dict,
    flow_sha: str,
    h: float,
) -> dict:
    reference = json.loads(reference_path.read_text())
    subset = reference["subsets"]["cases40_89"]
    expected_cases = tuple(map(int, subset["cases"]))
    indices = np.asarray([cases.index(case) for case in expected_cases], dtype=np.int64)
    model = reference["models"][reference_model]
    if model["flow_sha256"] != flow_sha:
        raise ValueError("reference result uses a different flow")
    measured = subset["models"][reference_model]
    actual = subset_point_m(statistics, indices, h)
    expected = {branch: float(measured[branch]["m_percent"]) for branch in BRANCHES}
    differences = {branch: actual[branch] - expected[branch] for branch in BRANCHES}
    if any(abs(value) > 2e-10 for value in differences.values()):
        raise RuntimeError(f"cases40-89 reference reproduction failed: {differences}")
    return {
        "path": str(reference_path.resolve()),
        "model": reference_model,
        "cases": list(expected_cases),
        "expected_m_percent": expected,
        "recomputed_m_percent": actual,
        "difference_percentage_points": differences,
    }


def compute(args: argparse.Namespace) -> None:
    cases = tuple(sorted(set(args.case)))
    if len(cases) != len(args.case):
        raise ValueError("duplicate cases are not allowed")
    lookup_paths = tuple(path.resolve() for path in args.rblend)
    lookup = load_lookup(lookup_paths, cases)

    property_chunks = {name: [] for name in PROPERTY_ORDER}
    input_anchor_rows = {}
    anchor_rows = {}
    for case in cases:
        anchor_path = Path(args.anchor_pattern.format(case=case))
        ids, context, target = load_anchor(anchor_path, lookup[case])
        input_anchor_rows[str(case)] = int(len(ids))
        keep = population_mask(context, args.true_magnitude_max)
        ids, context, target = ids[keep], context[keep], target[keep]
        values = property_values(
            context,
            target,
            lookup[case]["R_blend_abs"][keep],
        )
        for name in PROPERTY_ORDER:
            property_chunks[name].append(values[name])
        anchor_rows[str(case)] = int(len(ids))
    profiles = {
        name: quantile_profile(np.concatenate(property_chunks[name]), args.bins)
        for name in PROPERTY_ORDER
    }
    del property_chunks

    bins_by_property = {name: len(profiles[name]["centers"]) for name in PROPERTY_ORDER}
    statistics = empty_statistics(len(cases), bins_by_property)

    device = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    bundle = load_measurement_model(args.flow, device=device)
    if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
        raise ValueError("flow feature contract differs from the fixed-g0 domain")
    classifier = load_selection_model_ensemble(args.classifier, device=device)
    if classifier.preprocessor.feature_names != list(FLOW_FEATURES):
        raise ValueError("classifier feature contract differs from the fixed-g0 domain")

    case_reports = {}
    for case_position, case in enumerate(cases):
        anchor_path = Path(args.anchor_pattern.format(case=case))
        ids, context, target = load_anchor(anchor_path, lookup[case])
        keep = population_mask(context, args.true_magnitude_max)
        ids, context, target = ids[keep], context[keep], target[keep]
        case_lookup = {
            name: values[keep]
            for name, values in lookup[case].items()
        }
        plus_frame, minus_frame = shear_frames(context, args.h)
        legs = {
            "plus": load_measured_leg(args.constgold_root, case, +args.h, ids),
            "minus": load_measured_leg(args.constgold_root, case, -args.h, ids),
        }
        common = legs["plus"]["usable"] & legs["minus"]["usable"]
        probabilities = {
            "plus": np.asarray(classifier.predict_proba(plus_frame), dtype=np.float64),
            "minus": np.asarray(classifier.predict_proba(minus_frame), dtype=np.float64),
        }
        for direction in ("plus", "minus"):
            probability = probabilities[direction]
            if (
                probability.shape != (len(ids),)
                or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))
            ):
                raise ValueError(f"case {case}: invalid classifier probability")

        plus_mean, minus_mean = model_means(
            bundle,
            plus_frame,
            minus_frame,
            draws=args.draws,
            batch_size=args.batch_size,
            seed=args.sampling_seed + 10_000_019 * case,
        )
        base_e1, base_e2 = context[:, 0], context[:, 1]
        plus_e1, plus_e2 = apply_shear_to_ellipticity(
            base_e1, base_e2, args.h, 0.0
        )
        minus_e1, minus_e2 = apply_shear_to_ellipticity(
            base_e1, base_e2, -args.h, 0.0
        )
        truth_delta = {
            "plus": np.column_stack((plus_e1 - base_e1, plus_e2 - base_e2)),
            "minus": np.column_stack((minus_e1 - base_e1, minus_e2 - base_e2)),
        }
        model_values = {
            "plus": plus_mean + case_lookup["R_blend"][:, None] * truth_delta["plus"],
            "minus": minus_mean
            + case_lookup["R_blend"][:, None] * truth_delta["minus"],
        }
        properties = property_values(context, target, case_lookup["R_blend_abs"])

        for property_name in PROPERTY_ORDER:
            bins = bins_by_property[property_name]
            assignment = assign_bins(properties[property_name], profiles[property_name]["edges"])
            for branch in BRANCHES:
                for direction in ("plus", "minus"):
                    actual = legs[direction]["usable"].astype(np.float64)
                    measured_weights = common.astype(np.float64) if branch == "matched_usable" else actual
                    if branch == "matched_usable":
                        model_weights = common.astype(np.float64)
                    elif branch == "actual_usable_flags":
                        model_weights = actual
                    else:
                        model_weights = probabilities[direction]
                    for source, values, weights in (
                        ("measured", legs[direction]["values"], measured_weights),
                        ("model", model_values[direction], model_weights),
                    ):
                        denominator, numerator = binned_sufficient(
                            values, weights, assignment, bins
                        )
                        destination = statistics[property_name][branch][source][direction]
                        destination["denominator"][case_position] = denominator
                        destination["numerator"][case_position] = numerator

        case_reports[str(case)] = {
            "input_anchor_rows": input_anchor_rows[str(case)],
            "anchor_rows": int(len(ids)),
            "matched_usable_rows": int(common.sum()),
            "actual_usable_rows": {
                direction: int(legs[direction]["usable"].sum())
                for direction in ("plus", "minus")
            },
            "mean_modeled_usable_probability": {
                direction: float(probabilities[direction].mean())
                for direction in ("plus", "minus")
            },
        }
        print(
            f"CONSTGOLD_PROPERTY_CASE_DONE case={case} anchor={len(ids)} "
            f"matched={int(common.sum())}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    validate_partitions(statistics)
    case_weights = make_case_weights(args.n_boot, len(cases), args.bootstrap_seed)
    rows = []
    for property_name in PROPERTY_ORDER:
        profile = profiles[property_name]
        for branch in BRANCHES:
            summary = summarize_branch(
                statistics[property_name][branch], case_weights, args.h
            )
            for bin_index in range(bins_by_property[property_name]):
                rows.append(
                    {
                        "property": property_name,
                        "bin": bin_index,
                        "edge_low": profile["edges"][bin_index],
                        "edge_high": profile["edges"][bin_index + 1],
                        "x_median": profile["centers"][bin_index],
                        "anchor_objects": profile["counts"][bin_index],
                        "branch": branch,
                        "measured_R11": summary["measured_response"][bin_index, 0],
                        "model_R11": summary["model_response"][bin_index, 0],
                        "m_percent": summary["m_percent"][bin_index],
                        "m_standard_error_percentage_points": summary[
                            "m_standard_error_percentage_points"
                        ][bin_index],
                        "m_ci95_low_percent": summary["m_ci95_percent"][bin_index, 0],
                        "m_ci95_high_percent": summary["m_ci95_percent"][bin_index, 1],
                        "measured_plus_rows": statistics[property_name][branch]["measured"][
                            "plus"
                        ]["denominator"][:, bin_index].sum(),
                        "measured_minus_rows": statistics[property_name][branch]["measured"][
                            "minus"
                        ]["denominator"][:, bin_index].sum(),
                        "model_plus_effective_rows": statistics[property_name][branch]["model"][
                            "plus"
                        ]["denominator"][:, bin_index].sum(),
                        "model_minus_effective_rows": statistics[property_name][branch]["model"][
                            "minus"
                        ]["denominator"][:, bin_index].sum(),
                    }
                )

    overall = {}
    first = statistics[PROPERTY_ORDER[0]]
    for branch in BRANCHES:
        summary = summarize_branch(collapse_bins(first[branch]), case_weights, args.h)
        overall[branch] = {
            "measured_response": summary["measured_response"][0],
            "model_response": summary["model_response"][0],
            "m_percent": summary["m_percent"][0],
            "m_standard_error_percentage_points": summary[
                "m_standard_error_percentage_points"
            ][0],
            "m_ci95_percent": summary["m_ci95_percent"][0],
        }

    flow_sha = file_sha256(args.flow)
    reference = None
    if args.reference_result is not None:
        reference = verify_reference(
            args.reference_result,
            args.reference_model,
            cases,
            statistics,
            flow_sha,
            args.h,
        )

    result = {
        "format_version": 1,
        "model_label": args.model_label,
        "purpose": (
            "property-binned ConstGold m for matched, actual-per-leg, and "
            "classifier-weighted usability branches"
        ),
        "cases": list(cases),
        "case_count": len(cases),
        "h": float(args.h),
        "draws": int(args.draws),
        "sampling_seed": int(args.sampling_seed),
        "population": {
            "true_magnitude_max": args.true_magnitude_max,
            "definition": (
                "strict r_input_p < true_magnitude_max, applied to the fixed-g0 "
                "anchor before binning and before either ConstGold shear leg is read"
                if args.true_magnitude_max is not None
                else "complete fixed-g0 anchor population"
            ),
        },
        "bootstrap": {
            "unit": "ConstGold case",
            "replicates": int(args.n_boot),
            "seed": int(args.bootstrap_seed),
            "common_resamples_across_properties_bins_and_branches": True,
        },
        "binning": {
            "bins_requested": int(args.bins),
            "definition": (
                "global equal-anchor-count quantiles over cases, fixed before "
                "reading either ConstGold shear leg"
            ),
            "properties": {
                name: {
                    **PROPERTY_META[name],
                    "edges": profiles[name]["edges"],
                    "centers": profiles[name]["centers"],
                    "counts": profiles[name]["counts"],
                }
                for name in PROPERTY_ORDER
            },
        },
        "inputs": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "constgold_root": str(args.constgold_root.resolve()),
            "anchor_pattern": args.anchor_pattern,
            "flow": str(args.flow.resolve()),
            "flow_sha256": flow_sha,
            "rblend": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in lookup_paths
            ],
            "classifiers": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in args.classifier
            ],
        },
        "input_anchor_rows_by_case": input_anchor_rows,
        "anchor_rows_by_case": anchor_rows,
        "case_reports": case_reports,
        "overall": overall,
        "summary_rows": rows,
        "reference_reproduction": reference,
        "per_case_sufficient": statistics,
        "limitations": [
            "Cases40-139 reuse finite truth scenes used by model development; this is not fresh acceptance evidence.",
            "Bin edges are estimated from this fixed-g0 anchor population and are diagnostic, not analysis cuts.",
            "The modeled-usable branch uses p(U|x), while its measured comparator uses actual per-leg usability.",
            "One flow seed and one common antithetic integration seed do not quantify training or integration uncertainty.",
        ],
    }
    write_json(args.result, result)
    print(f"CONSTGOLD_PROPERTY_COMPLETE result={args.result}", flush=True)


def response_panel_rows(result: dict, properties: tuple[str, ...]) -> pd.DataFrame:
    """Reconstruct response estimates and case-bootstrap errors for selected panels."""

    unknown = set(properties).difference(PROPERTY_ORDER)
    if unknown:
        raise ValueError(f"unknown response properties: {sorted(unknown)}")
    case_weights = make_case_weights(
        result["bootstrap"]["replicates"],
        result["case_count"],
        result["bootstrap"]["seed"],
    )
    rows = []
    summary_rows = pd.DataFrame(result["summary_rows"])
    for property_name in properties:
        property_rows = summary_rows.loc[
            summary_rows["property"] == property_name
        ].set_index(["branch", "bin"])
        for branch in BRANCHES:
            branch_stats = {
                source: {
                    direction: {
                        field: np.asarray(values, dtype=np.float64)
                        for field, values in sufficient.items()
                    }
                    for direction, sufficient in directions.items()
                }
                for source, directions in result["per_case_sufficient"][property_name][
                    branch
                ].items()
            }
            summary = summarize_branch(branch_stats, case_weights, result["h"])
            bins = len(summary["measured_response"])
            for bin_index in range(bins):
                source_row = property_rows.loc[(branch, bin_index)]
                rows.append(
                    {
                        "property": property_name,
                        "bin": bin_index,
                        "branch": branch,
                        "edge_low": source_row["edge_low"],
                        "edge_high": source_row["edge_high"],
                        "x_median": source_row["x_median"],
                        "anchor_objects": source_row["anchor_objects"],
                        "measured_R11": summary["measured_response"][bin_index, 0],
                        "measured_R11_standard_error": summary[
                            "measured_response_standard_error"
                        ][bin_index],
                        "model_R11": summary["model_response"][bin_index, 0],
                        "model_R11_standard_error": summary[
                            "model_response_standard_error"
                        ][bin_index],
                    }
                )
    return pd.DataFrame(rows)


def apply_plot_style() -> None:
    """Apply the shared publication-oriented style for this diagnostic."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 350,
        }
    )


def add_anchor_histogram(
    ax: plt.Axes,
    histogram: dict | None,
) -> plt.Axes | None:
    """Add a normalized anchor histogram behind a response or bias panel."""

    if histogram is None:
        return None
    histogram_ax = ax.twinx()
    histogram_ax.stairs(
        histogram["relative_density"],
        histogram["edges"],
        fill=True,
        color="0.55",
        edgecolor="0.45",
        alpha=0.18,
        linewidth=0.45,
        zorder=0,
    )
    histogram_ax.set_ylim(0.0, 1.08)
    histogram_ax.set_yticks([])
    histogram_ax.tick_params(right=False)
    histogram_ax.spines["right"].set_visible(False)
    histogram_ax.spines["top"].set_visible(False)
    histogram_ax.set_zorder(0)
    histogram_ax.patch.set_visible(False)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)
    return histogram_ax


def configure_property_axis(
    ax: plt.Axes,
    property_name: str,
    local: pd.DataFrame,
    histogram: dict | None,
) -> None:
    """Apply the declared property transform and range to one panel."""

    meta = PROPERTY_META[property_name]
    if meta["scale"] == "log":
        ax.set_xscale("log")
    elif meta["scale"] == "symlog":
        threshold = (
            histogram["linthresh"]
            if histogram is not None
            else symlog_threshold(local["x_median"].to_numpy())
        )
        ax.set_xscale("symlog", linthresh=threshold)
    if histogram is not None:
        ax.set_xlim(histogram["edges"][[0, -1]])
    ax.set_title(meta["title"])
    ax.set_xlabel(meta["xlabel"])


def plot_response_panels(
    args: argparse.Namespace,
    result: dict,
    histograms: dict | None,
    histogram_rows: pd.DataFrame | None,
) -> None:
    """Plot R_measured and R_model for a selected subset of property panels."""

    properties = tuple(dict.fromkeys(args.response_property))
    rows = response_panel_rows(result, properties)
    fig, axes = plt.subplots(
        1,
        len(properties),
        figsize=(4.4 * len(properties), 3.8),
        squeeze=False,
        layout="constrained",
    )
    for ax, property_name in zip(axes.ravel(), properties, strict=True):
        histogram = histograms[property_name] if histograms is not None else None
        add_anchor_histogram(ax, histogram)
        ax.axhline(0.0, color="0.45", linewidth=0.8)
        local = rows.loc[rows["property"] == property_name]
        for branch in BRANCHES:
            style = BRANCH_META[branch]
            table = local.loc[local["branch"] == branch].sort_values("bin")
            ax.errorbar(
                table["x_median"],
                table["model_R11"],
                yerr=table["model_R11_standard_error"],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="white",
                markeredgewidth=0.9,
                linestyle="--",
                linewidth=1.1,
                markersize=4.2,
                capsize=2.0,
                capthick=0.8,
                zorder=2,
            )
            # The classifier branch deliberately uses the actual-per-leg measured
            # comparator, so plotting it again would exactly obscure that curve.
            if branch == "modeled_usable":
                actual = local.loc[
                    local["branch"] == "actual_usable_flags"
                ].sort_values("bin")
                np.testing.assert_allclose(
                    table["measured_R11"], actual["measured_R11"], rtol=0.0, atol=0.0
                )
                continue
            ax.errorbar(
                table["x_median"],
                table["measured_R11"],
                yerr=table["measured_R11_standard_error"],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=style["color"],
                linestyle="-",
                linewidth=1.2,
                markersize=4.2,
                capsize=2.0,
                capthick=0.8,
                zorder=3,
            )
        configure_property_axis(ax, property_name, local, histogram)
        ax.set_ylabel(r"Shear response $R_{11}$")
        ax.text(
            0.015,
            0.985,
            chr(ord("A") + PROPERTY_ORDER.index(property_name)),
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
        )
        ax.grid(axis="y", color="0.9", linewidth=0.6)

    branch_handles = [
        Line2D(
            [0],
            [0],
            color=BRANCH_META[branch]["color"],
            marker=BRANCH_META[branch]["marker"],
            linestyle="none",
            label=BRANCH_META[branch]["label"],
        )
        for branch in BRANCHES
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            markerfacecolor="0.25",
            linestyle="-",
            label=r"$R_{\rm meas}$",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            markerfacecolor="white",
            linestyle="--",
            label=r"$R_{\rm model}$",
        ),
    ]
    handles = branch_handles + source_handles
    if histograms is not None:
        handles.append(
            Patch(
                facecolor="0.55",
                edgecolor="0.45",
                alpha=0.22,
                label="Anchor population",
            )
        )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=8.5,
    )
    title = (
        f"ConstGold measured and modeled response — {result['case_count']} cases, "
        f"{result['binning']['bins_requested']} quantile bins"
    )
    true_magnitude_max = result.get("population", {}).get("true_magnitude_max")
    if true_magnitude_max is not None:
        title += rf", true $r<{true_magnitude_max:g}$"
    if result.get("model_label"):
        title += f"\n{result['model_label']}"
    fig.suptitle(title)
    prefix = args.output_prefix
    fig.savefig(prefix.with_suffix(".png"), bbox_inches="tight", dpi=350)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    rows.to_csv(prefix.with_suffix(".csv"), index=False)
    if histogram_rows is not None:
        histogram_rows.loc[
            histogram_rows["property"].isin(properties)
        ].to_csv(prefix.parent / f"{prefix.name}_histograms.csv", index=False)
    print(f"CONSTGOLD_RESPONSE_PANELS_WRITTEN prefix={prefix}", flush=True)


def plot(args: argparse.Namespace) -> None:
    result = json.loads(args.result.read_text())
    rows = pd.DataFrame(result["summary_rows"])
    prefix = args.output_prefix
    outputs = [prefix.with_suffix(suffix) for suffix in (".png", ".pdf", ".csv")]
    histogram_csv = prefix.parent / f"{prefix.name}_histograms.csv"
    if not args.no_histogram:
        outputs.append(histogram_csv)
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite one of {outputs}")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    histograms = None
    histogram_rows = None
    if not args.no_histogram:
        histograms, histogram_rows = build_anchor_histograms(
            result, args.histogram_bins
        )

    apply_plot_style()
    if args.response_property:
        plot_response_panels(args, result, histograms, histogram_rows)
        return

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.5), layout="constrained")
    axes = axes.ravel()
    for panel, (ax, property_name) in enumerate(zip(axes, PROPERTY_ORDER, strict=True)):
        histogram = histograms[property_name] if histograms is not None else None
        add_anchor_histogram(ax, histogram)
        ax.axhspan(-0.3, 0.3, color="#009E73", alpha=0.08, linewidth=0)
        ax.axhline(0.0, color="0.45", linewidth=0.8)
        local = rows.loc[rows["property"] == property_name]
        for branch in BRANCHES:
            style = BRANCH_META[branch]
            table = local.loc[local["branch"] == branch].sort_values("bin")
            ax.errorbar(
                table["x_median"],
                table["m_percent"],
                yerr=table["m_standard_error_percentage_points"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.2,
                markersize=3.8,
                capsize=2.0,
                capthick=0.8,
                label=style["label"],
            )
        configure_property_axis(ax, property_name, local, histogram)
        if panel % 3 == 0:
            ax.set_ylabel(r"$m=R_{\rm measured}/R_{\rm model}-1$ (%)")
        ax.text(
            0.015,
            0.985,
            chr(ord("A") + panel),
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
        )
        ax.grid(axis="y", color="0.9", linewidth=0.6)

    handles = [
        Line2D(
            [0],
            [0],
            color=BRANCH_META[branch]["color"],
            marker=BRANCH_META[branch]["marker"],
            linestyle=BRANCH_META[branch]["linestyle"],
            label=BRANCH_META[branch]["label"],
        )
        for branch in BRANCHES
    ]
    handles.append(
        Line2D([0], [0], color="#009E73", linewidth=6, alpha=0.22, label=r"$|m|<0.3\%$")
    )
    if histograms is not None:
        handles.append(
            Patch(
                facecolor="0.55",
                edgecolor="0.45",
                alpha=0.22,
                label="Anchor population",
            )
        )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
    )
    title = (
        f"ConstGold response bias and anchor distributions — {result['case_count']} cases, "
        f"{result['binning']['bins_requested']} quantile bins"
    )
    true_magnitude_max = result.get("population", {}).get("true_magnitude_max")
    if true_magnitude_max is not None:
        title += rf", true $r<{true_magnitude_max:g}$"
    if result.get("model_label"):
        title += f"\n{result['model_label']}"
    fig.suptitle(title)
    fig.savefig(prefix.with_suffix(".png"), bbox_inches="tight", dpi=350)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    rows.to_csv(prefix.with_suffix(".csv"), index=False)
    if histogram_rows is not None:
        histogram_rows.to_csv(histogram_csv, index=False)
    print(f"CONSTGOLD_PROPERTY_PLOT_WRITTEN prefix={prefix}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compute_parser = commands.add_parser("compute", help="run the binned response calculation")
    compute_parser.add_argument("--constgold-root", type=Path, required=True)
    compute_parser.add_argument(
        "--anchor-pattern", required=True, help="case NPZ format string containing {case}"
    )
    compute_parser.add_argument("--flow", type=Path, required=True)
    compute_parser.add_argument("--rblend", type=Path, action="append", required=True)
    compute_parser.add_argument("--classifier", type=Path, action="append", required=True)
    compute_parser.add_argument("--case", type=int, action="append", required=True)
    compute_parser.add_argument("--result", type=Path, required=True)
    compute_parser.add_argument("--bins", type=int, default=8)
    compute_parser.add_argument("--h", type=float, default=0.02)
    compute_parser.add_argument("--draws", type=int, default=64)
    compute_parser.add_argument("--sampling-seed", type=int, default=7301)
    compute_parser.add_argument("--batch-size", type=int, default=1024)
    compute_parser.add_argument("--n-boot", type=int, default=10000)
    compute_parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    compute_parser.add_argument("--device", default="cuda")
    compute_parser.add_argument("--reference-result", type=Path)
    compute_parser.add_argument("--reference-model", default="new_v2_staged")
    compute_parser.add_argument("--model-label")
    compute_parser.add_argument(
        "--true-magnitude-max",
        type=float,
        help="strict declared-truth r_input_p population boundary",
    )

    plot_parser = commands.add_parser("plot", help="render panels from a completed result")
    plot_parser.add_argument("--result", type=Path, required=True)
    plot_parser.add_argument("--output-prefix", type=Path, required=True)
    plot_parser.add_argument("--histogram-bins", type=int, default=36)
    plot_parser.add_argument("--no-histogram", action="store_true")
    plot_parser.add_argument(
        "--response-property",
        choices=PROPERTY_ORDER,
        action="append",
        help=(
            "render a focused R_measured/R_model panel for this property; repeat "
            "to make a multi-panel response figure"
        ),
    )
    plot_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "compute":
        compute(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
