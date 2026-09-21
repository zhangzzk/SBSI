#!/usr/bin/env python
"""Find and plot ConstGold galaxies whose V3.2/V3.3 shape likelihoods differ most."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sbsi import sample_measurement
from scripts.evaluate_constgold_complete_forward_response import CROWD_COLUMNS
from scripts.plot_flow_likelihood_gallery import (
    COLORS,
    MASS_LEVELS,
    _build_case_frame,
    _contour,
    _sha256,
)


def sliced_wasserstein_1(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    directions: int,
    trim_fraction: float = 0.0,
    object_batch: int = 2048,
) -> np.ndarray:
    """Mean optionally trimmed 1D W1 over evenly spaced projection directions."""

    a = np.asarray(samples_a, dtype=np.float32)
    b = np.asarray(samples_b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 2:
        raise ValueError("samples must be aligned arrays with shape (objects, draws, 2)")
    if directions <= 0 or object_batch <= 0:
        raise ValueError("directions and object_batch must be positive")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim fraction must lie in [0, 0.5)")
    trim = int(np.floor(a.shape[1] * trim_fraction))
    if 2 * trim >= a.shape[1]:
        raise ValueError("trim fraction removes every draw")
    retained = slice(trim, a.shape[1] - trim if trim else None)
    angles = (np.arange(directions, dtype=np.float64) + 0.5) * np.pi / directions
    vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    result = np.empty(len(a), dtype=np.float64)
    for start in range(0, len(a), object_batch):
        stop = min(start + object_batch, len(a))
        score = np.zeros(stop - start, dtype=np.float64)
        for vector in vectors:
            projection_a = np.sort(a[start:stop] @ vector, axis=1)
            projection_b = np.sort(b[start:stop] @ vector, axis=1)
            score += np.mean(
                np.abs(projection_a[:, retained] - projection_b[:, retained]), axis=1
            )
        result[start:stop] = score / directions
    return result


def _validate_identities(source: pd.DataFrame, contexts: pd.DataFrame) -> None:
    columns = ["case", "input_index"]
    if source[columns].duplicated().any() or contexts[columns].duplicated().any():
        raise RuntimeError("duplicate ConstGold identities")
    source_ids = pd.MultiIndex.from_frame(source[columns])
    context_ids = pd.MultiIndex.from_frame(contexts[columns])
    missing = source_ids.difference(context_ids)
    extra = context_ids.difference(source_ids)
    if len(missing) or len(extra) or len(source) != len(contexts):
        raise RuntimeError(
            f"context identity mismatch: missing={len(missing)}, extra={len(extra)}"
        )


def _build_case_task(task):
    case, source, constgold_root, crowd, shear = task
    return case, _build_case_frame(
        source,
        case=case,
        constgold_root=constgold_root,
        crowd=crowd,
        shear=shear,
    )


def _build_all_contexts(
    source: pd.DataFrame,
    cases: list[int],
    *,
    constgold_root: Path,
    crowd: pd.DataFrame,
    shear: float,
    workers: int,
) -> pd.DataFrame:
    if workers <= 0:
        raise ValueError("context workers must be positive")
    tasks = [
        (
            case,
            source.loc[source["case"] == case].copy(),
            constgold_root,
            crowd.loc[crowd["case"] == case].copy(),
            shear,
        )
        for case in cases
    ]
    frames = {}
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            case, frame = _build_case_task(task)
            frames[case] = frame
            if completed % 10 == 0 or completed == len(tasks):
                print(f"context cases complete: {completed}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_case_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                case, frame = future.result()
                frames[case] = frame
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"context cases complete: {completed}/{len(tasks)}", flush=True)
    return pd.concat([frames[case] for case in cases], ignore_index=True)


def _display_extent(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    pooled = np.concatenate((samples_a[..., :2].ravel(), samples_b[..., :2].ravel()))
    robust = float(np.quantile(np.abs(pooled), 0.995))
    return min(2.0, max(1.0, np.ceil(robust * 10.0) / 10.0))


def _render(
    selected: pd.DataFrame,
    samples32: np.ndarray,
    samples33: np.ndarray,
    output: Path,
    *,
    draws: int,
    directions: int,
    trim_fraction: float,
    extent: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        8,
        8,
        figsize=(15.3, 15.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for index, ax in enumerate(axes.flat):
        row = selected.iloc[index]
        _contour(
            ax,
            samples32[index, :, 0],
            samples32[index, :, 1],
            color=COLORS["v3.2"],
            linestyle="solid",
            extent=extent,
        )
        _contour(
            ax,
            samples33[index, :, 0],
            samples33[index, :, 1],
            color=COLORS["v3.3"],
            linestyle="dashed",
            extent=extent,
        )
        ax.scatter(
            row["measured_ngmix_g1"],
            row["measured_ngmix_g2"],
            marker="x",
            s=20,
            linewidth=1.1,
            color="black",
            zorder=5,
        )
        ax.set_title(
            rf"#{index + 1}: $d_{{tSW}}={row['sliced_w1']:.3f}$, "
            rf"$\hat m={row['measured_mag_auto']:.2f}$",
            pad=2.4,
        )
        ax.text(
            0.03,
            0.03,
            f"c{int(row['case'])} / {int(row['input_index'])}",
            transform=ax.transAxes,
            fontsize=5.1,
            color="0.35",
        )
        ax.axhline(0, color="0.88", lw=0.4, zorder=0)
        ax.axvline(0, color="0.88", lw=0.4, zorder=0)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="out", length=2.0, width=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    for row in range(8):
        axes[row, 0].set_ylabel(r"measured $\hat e_2$")
    for column in range(8):
        axes[-1, column].set_xlabel(r"measured $\hat e_1$")

    handles = [
        Line2D([0], [0], color=COLORS["v3.2"], lw=1.7, label="V3.2 flow"),
        Line2D(
            [0], [0], color=COLORS["v3.3"], lw=1.7, ls="--", label="V3.3 flow"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="x",
            lw=0,
            markersize=6,
            label="ConstGold measurement",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.991),
        fontsize=9,
    )
    fig.suptitle(
        "Largest V3.2–V3.3 per-galaxy likelihood differences in 100k ConstGold",
        y=0.999,
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.5,
        0.006,
        f"Ranked by {100 * trim_fraction:.1f}%-tail-trimmed sliced Wasserstein-1 "
        f"in measured shape ({directions} projections; "
        f"{draws:,} common-seed draws). Contours enclose 50%, 80%, and 95%. "
        r"Flow term only: no detection, measured cuts, or $R_{\rm blend}$.",
        ha="center",
        fontsize=8,
        color="0.25",
    )
    fig.subplots_adjust(
        left=0.047,
        right=0.995,
        bottom=0.041,
        top=0.967,
        hspace=0.27,
        wspace=0.11,
    )
    fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sample", type=Path, required=True)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--crowd-lookup", type=Path, required=True)
    parser.add_argument("--v32-model", type=Path, required=True)
    parser.add_argument("--v33-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-draws", type=int, default=128)
    parser.add_argument("--screen-directions", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=512)
    parser.add_argument("--final-draws", type=int, default=8192)
    parser.add_argument("--final-directions", type=int, default=32)
    parser.add_argument("--trim-fraction", type=float, default=0.005)
    parser.add_argument("--show", type=int, default=64)
    parser.add_argument("--context-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--shear", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    outputs = [
        args.output.with_suffix(ext)
        for ext in (".png", ".pdf", ".json", ".parquet", ".npz")
    ]
    outputs.extend(
        [
            args.output.with_name(args.output.name + "_screen.parquet"),
            args.output.with_name(args.output.name + "_shortlist.parquet"),
        ]
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}")
    positive = (
        args.screen_draws,
        args.screen_directions,
        args.shortlist,
        args.final_draws,
        args.final_directions,
        args.show,
        args.context_workers,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("draw, direction, shortlist, and display counts must be positive")
    if args.show > args.shortlist:
        raise ValueError("show cannot exceed shortlist")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("trim fraction must lie in [0, 0.5)")

    source = pd.read_parquet(args.source_sample)
    cases = np.sort(source["case"].unique()).astype(int).tolist()
    crowd = pd.read_feather(
        args.crowd_lookup,
        columns=["case", "input_index", *CROWD_COLUMNS],
    )
    crowd = crowd.loc[crowd["case"].isin(cases)].copy()
    contexts = _build_all_contexts(
        source,
        cases,
        constgold_root=args.constgold_root,
        crowd=crowd,
        shear=args.shear,
        workers=args.context_workers,
    )
    _validate_identities(source, contexts)
    if args.shortlist > len(contexts):
        raise ValueError("shortlist exceeds source sample")

    print("sampling V3.2 screen", flush=True)
    screen32 = sample_measurement(
        [args.v32_model],
        contexts,
        n_samples=args.screen_draws,
        batch_size=1024,
        random_seed=args.seed,
        device=args.device,
    )
    print("sampling V3.3 screen", flush=True)
    screen33 = sample_measurement(
        [args.v33_model],
        contexts,
        n_samples=args.screen_draws,
        batch_size=1024,
        random_seed=args.seed,
        device=args.device,
    )
    if screen32.target_names != screen33.target_names:
        raise RuntimeError("V3.2/V3.3 target order differs")
    print("ranking 100k screen", flush=True)
    screen_score = sliced_wasserstein_1(
        np.asarray(screen32.samples)[..., :2],
        np.asarray(screen33.samples)[..., :2],
        directions=args.screen_directions,
        trim_fraction=args.trim_fraction,
    )
    screen_table = contexts[
        ["case", "input_index", "measured_mag_auto", "r_input_p"]
    ].copy()
    screen_table["screen_sliced_w1"] = screen_score
    screen_table["screen_rank"] = (
        pd.Series(screen_score).rank(method="first", ascending=False).astype(int).to_numpy()
    )
    screen_path = args.output.with_name(args.output.name + "_screen.parquet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    screen_table.to_parquet(screen_path, index=False)

    shortlist_index = np.argsort(-screen_score, kind="stable")[: args.shortlist]
    shortlist = contexts.iloc[shortlist_index].copy().reset_index(drop=True)
    shortlist["screen_sliced_w1"] = screen_score[shortlist_index]
    shortlist["screen_rank"] = screen_table.iloc[shortlist_index][
        "screen_rank"
    ].to_numpy()
    del screen32, screen33

    print("sampling V3.2 shortlist", flush=True)
    final32 = sample_measurement(
        [args.v32_model],
        shortlist,
        n_samples=args.final_draws,
        batch_size=32,
        random_seed=args.seed + 1,
        device=args.device,
    )
    print("sampling V3.3 shortlist", flush=True)
    final33 = sample_measurement(
        [args.v33_model],
        shortlist,
        n_samples=args.final_draws,
        batch_size=32,
        random_seed=args.seed + 1,
        device=args.device,
    )
    samples32 = np.asarray(final32.samples, dtype=np.float32)[..., :2]
    samples33 = np.asarray(final33.samples, dtype=np.float32)[..., :2]
    print("reranking shortlist", flush=True)
    final_score = sliced_wasserstein_1(
        samples32,
        samples33,
        directions=args.final_directions,
        trim_fraction=args.trim_fraction,
        object_batch=64,
    )
    score_even = sliced_wasserstein_1(
        samples32[:, ::2],
        samples33[:, ::2],
        directions=args.final_directions,
        trim_fraction=args.trim_fraction,
        object_batch=64,
    )
    score_odd = sliced_wasserstein_1(
        samples32[:, 1::2],
        samples33[:, 1::2],
        directions=args.final_directions,
        trim_fraction=args.trim_fraction,
        object_batch=64,
    )
    shortlist["sliced_w1"] = final_score
    shortlist["sliced_w1_even"] = score_even
    shortlist["sliced_w1_odd"] = score_odd
    shortlist["final_rank"] = (
        pd.Series(final_score).rank(method="first", ascending=False).astype(int).to_numpy()
    )
    shortlist_path = args.output.with_name(args.output.name + "_shortlist.parquet")
    shortlist.to_parquet(shortlist_path, index=False)

    final_order = np.argsort(-final_score, kind="stable")[: args.show]
    selected = shortlist.iloc[final_order].copy().reset_index(drop=True)
    selected32 = samples32[final_order]
    selected33 = samples33[final_order]
    extent = _display_extent(selected32, selected33)
    selected.to_parquet(args.output.with_suffix(".parquet"), index=False)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        v32=selected32,
        v33=selected33,
    )
    _render(
        selected,
        selected32,
        selected33,
        args.output,
        draws=args.final_draws,
        directions=args.final_directions,
        trim_fraction=args.trim_fraction,
        extent=extent,
    )

    even_top = set(np.argsort(-score_even, kind="stable")[: args.show].tolist())
    odd_top = set(np.argsort(-score_odd, kind="stable")[: args.show].tolist())
    outside32 = np.any(np.abs(selected32) > extent, axis=2)
    outside33 = np.any(np.abs(selected33) > extent, axis=2)
    report = {
        "purpose": "largest V3.2/V3.3 per-galaxy conditional shape-likelihood differences",
        "projection": ["measured_ngmix_g1", "measured_ngmix_g2"],
        "ranking_metric": {
            "name": "trimmed sliced Wasserstein-1",
            "units": "ellipticity",
            "screen_directions": args.screen_directions,
            "final_directions": args.final_directions,
            "normalization": "none",
            "trim_fraction_per_tail": args.trim_fraction,
        },
        "likelihood_terms_excluded": [
            "detection probability",
            "measured selection",
            "R_blend",
        ],
        "source_sample": str(args.source_sample.resolve()),
        "source_sample_sha256": _sha256(args.source_sample),
        "source_rows": len(source),
        "constgold_cases": cases,
        "unmatched_rows": 0,
        "shear": [args.shear, 0.0],
        "screen": {
            "draws_per_model_per_galaxy": args.screen_draws,
            "common_random_seed": args.seed,
            "shortlist_rows": args.shortlist,
        },
        "rerank": {
            "draws_per_model_per_galaxy": args.final_draws,
            "common_random_seed": args.seed + 1,
            "display_rows": args.show,
            "even_odd_top_set_overlap": len(even_top & odd_top),
            "even_odd_spearman": float(spearmanr(score_even, score_odd).statistic),
        },
        "selected_metric_range": [
            float(selected["sliced_w1"].min()),
            float(selected["sliced_w1"].max()),
        ],
        "selected_measured_magnitude_range": [
            float(selected["measured_mag_auto"].min()),
            float(selected["measured_mag_auto"].max()),
        ],
        "contour_mass_levels": list(MASS_LEVELS),
        "axis_extent": [-extent, extent],
        "outside_display_fraction": {
            "v3.2": float(outside32.mean()),
            "v3.3": float(outside33.mean()),
        },
        "models": {
            "v3.2": {
                "path": str(args.v32_model.resolve()),
                "sha256": _sha256(args.v32_model),
            },
            "v3.3": {
                "path": str(args.v33_model.resolve()),
                "sha256": _sha256(args.v33_model),
            },
        },
        "outputs": {},
    }
    artifact_paths = [
        args.output.with_suffix(".png"),
        args.output.with_suffix(".pdf"),
        args.output.with_suffix(".parquet"),
        args.output.with_suffix(".npz"),
        screen_path,
        shortlist_path,
    ]
    report["outputs"] = {
        path.name: {"path": str(path.resolve()), "sha256": _file_sha256(path)}
        for path in artifact_paths
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
