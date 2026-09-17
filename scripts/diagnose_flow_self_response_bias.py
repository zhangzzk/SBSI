"""Compare the flow's predicted self response against the measured one.

The flow is trained on the half-shear legs, so its own self response can be
read directly: match the zero-shear and sheared legs of the fixed-g0 domain,
project the measured shape difference onto the applied shear, and put that
next to the flow's prediction for the same objects.  No blending model and no
ConstGold enter anywhere, so this is the flow on its own terms.

The split recorded in the domain manifest says which cases the flow trained
on, so the same read is reported for trained and held-out cases separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.fixed_g0_domain import FLOW_FEATURES, matched_key_indices

MAGNITUDE_EDGES = (-np.inf, 23.0, 24.0, 25.0, 26.0, np.inf)
MAGNITUDE_LABELS = ("r<23", "23-24", "24-25", "25-26", "r>=26")


def projected_response(delta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Project a shape difference onto the applied shear, as the flow does."""

    squared = np.einsum("ij,ij->i", gamma, gamma)
    return np.einsum("ij,ij->i", delta, gamma) / squared


def load_leg(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        result = {name: stored[name] for name in stored.files}
    missing = {"case", "input_index", "context", "target", "gamma"} - set(result)
    if missing:
        raise RuntimeError(f"{path} is missing {sorted(missing)}")
    return result


def measured_case(zero: dict[str, np.ndarray], sheared: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per matched primary carrying the measured self response."""

    left, right, counts = matched_key_indices(
        zero["case"], zero["input_index"], sheared["case"], sheared["input_index"]
    )
    gamma = np.asarray(sheared["gamma"][right], dtype=np.float64)
    shapes = np.asarray(sheared["target"][right][:, :2], dtype=np.float64)
    shapes = shapes - np.asarray(zero["target"][left][:, :2], dtype=np.float64)
    #  A zero or non-finite shear has no response to project onto, and a
    #  non-finite shape cannot contribute a response either.  Both are dropped
    #  and counted rather than silently carried as a NaN.
    keep = np.isfinite(gamma).all(axis=1) & (np.einsum("ij,ij->i", gamma, gamma) > 0.0)
    keep &= np.isfinite(shapes).all(axis=1)
    counts["zero_or_nonfinite_dropped"] = int((~keep).sum())
    left, right, gamma, shapes = left[keep], right[keep], gamma[keep], shapes[keep]
    context = np.asarray(zero["context"][left], dtype=np.float64)
    frame = pd.DataFrame(
        {
            "case": zero["case"][left].astype(np.int16),
            "input_index": zero["input_index"][left].astype(np.int64),
            "R_self_measured": projected_response(shapes, gamma),
        }
    )
    for name in ("r_input_p", "circularized_Re_input_p", "nbr_flux_max"):
        frame[name] = context[:, FLOW_FEATURES.index(name)]
    return frame, counts


def case_sums(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-case sums, so any split is an equal-weight mean over cases."""

    grouped = frame.groupby("case", sort=True)
    return pd.DataFrame(
        {
            "objects": grouped.size(),
            "measured": grouped["R_self_measured"].sum(),
            "model": grouped["R_self_model"].sum(),
        }
    )


def summarize(sums: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, float]:
    """Equal-weight case means with a case bootstrap on the bias."""

    if sums.empty or sums["objects"].sum() == 0:
        raise ValueError("a split has no objects")
    populated = sums[sums["objects"] > 0]
    measured = (populated["measured"] / populated["objects"]).to_numpy(float)
    model = (populated["model"] / populated["objects"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(measured), size=(replicates, len(measured)))
    ratios = 100.0 * (measured[draws].mean(axis=1) / model[draws].mean(axis=1) - 1.0)
    return {
        "cases": int(len(measured)),
        "objects": int(populated["objects"].sum()),
        "R_self_measured": float(measured.mean()),
        "R_self_model": float(model.mean()),
        "R_self_measured_case_sem": float(measured.std(ddof=1) / np.sqrt(len(measured))),
        "m_percent": float(100.0 * (measured.mean() / model.mean() - 1.0)),
        "m_percent_standard_error": float(ratios.std(ddof=1)),
    }


def report(frame: pd.DataFrame, column: str, labels, *, replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for label in labels:
        selected = frame[frame[column] == label]
        if selected.empty:
            continue
        summary = summarize(case_sums(selected), replicates=replicates, seed=seed)
        rows.append({column: label, **summary})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_feather(args.predictions)
    manifest = json.loads((args.domain_root / "manifest.json").read_text())
    train_cases = set(int(case) for case in manifest["split"]["train_cases"])

    frames, drops = [], {}
    for case in sorted(int(case) for case in predictions["case"].unique()):
        zero = load_leg(args.domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        sheared = load_leg(args.domain_root / "flow" / "g05" / f"case{case:03d}.npz")
        measured, counts = measured_case(zero, sheared)
        frames.append(measured)
        drops[str(case)] = counts
    measured = pd.concat(frames, ignore_index=True)

    merged = measured.merge(predictions, on=("case", "input_index"), how="inner", validate="one_to_one")
    unmatched = {
        "measured_rows": int(len(measured)),
        "prediction_rows": int(len(predictions)),
        "joined_rows": int(len(merged)),
        "measured_unmatched": int(len(measured) - len(merged)),
        "prediction_unmatched": int(len(predictions) - len(merged)),
    }
    if not np.isfinite(merged["R_self_model"].to_numpy(float)).all():
        raise RuntimeError("the prediction column carries a non-finite value")

    merged["split"] = np.where(merged["case"].isin(train_cases), "trained", "held-out")
    merged["magnitude"] = pd.cut(
        merged["r_input_p"], bins=list(MAGNITUDE_EDGES), labels=list(MAGNITUDE_LABELS), right=False
    )
    crowding = merged["nbr_flux_max"].to_numpy(float)
    quartiles = np.quantile(crowding, [0.25, 0.5, 0.75])
    merged["crowding"] = pd.cut(
        crowding,
        bins=[-np.inf, *quartiles, np.inf],
        labels=["q1 quietest", "q2", "q3", "q4 most crowded"],
        right=False,
    )

    overall = summarize(case_sums(merged), replicates=args.replicates, seed=args.seed)
    by_split = report(merged, "split", ("trained", "held-out"), replicates=args.replicates, seed=args.seed)
    by_magnitude = report(merged, "magnitude", MAGNITUDE_LABELS, replicates=args.replicates, seed=args.seed)
    by_crowding = report(
        merged, "crowding", ("q1 quietest", "q2", "q3", "q4 most crowded"),
        replicates=args.replicates, seed=args.seed,
    )
    quiet = merged[merged["crowding"] == "q1 quietest"]
    quiet_by_magnitude = report(
        quiet, "magnitude", MAGNITUDE_LABELS, replicates=args.replicates, seed=args.seed
    )

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print("############ overall ############")
        print(json.dumps(overall, indent=2))
        print("############ trained vs held-out cases ############")
        print(by_split.to_string(index=False))
        print("############ by true primary magnitude ############")
        print(by_magnitude.to_string(index=False))
        print("############ by crowding (nbr_flux_max quartile) ############")
        print(by_crowding.to_string(index=False))
        print("############ quietest crowding quartile, by magnitude ############")
        print(quiet_by_magnitude.to_string(index=False))
        print("############ row accounting ############")
        print(json.dumps(unmatched, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "inputs": {
                    "domain_root": str(args.domain_root),
                    "predictions": str(args.predictions),
                },
                "row_accounting": unmatched,
                "per_case_match_counts": drops,
                "crowding_quartiles": [float(value) for value in quartiles],
                "overall": overall,
                "by_split": by_split.to_dict("records"),
                "by_magnitude": by_magnitude.to_dict("records"),
                "by_crowding": by_crowding.to_dict("records"),
                "quiet_by_magnitude": quiet_by_magnitude.to_dict("records"),
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
