"""Analyze frozen distance-definition variants on coherent-neighbour anchors.

The candidate is predeclared as ``lsst_r_extnbr_indist_wc5``.  No model is fit
here.  Cases 200--299 are the final anchor gate because those latent fields do
not exist in the response-emulator training catalogue.  Earlier splits are
reported to expose reuse of latent input fields and sensitivity to case range.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

CANDIDATE = "lsst_r_extnbr_indist_wc5"
BASELINE = "lsst_r_extnbr_v22"
MODELS = [
    BASELINE,
    "lsst_r_extnbr_indist",
    CANDIDATE,
    "lsst_r_extnbr_indist_wc20",
]
SPLITS = [
    ("heldout_latent_0_39", 0, 39),
    ("training_latent_40_99", 40, 99),
    ("training_latent_100_199", 100, 199),
    ("final_unseen_latent_200_299", 200, 299),
    ("all_0_299", 0, 299),
]


def finite_sem(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def summarize_split(frame: pd.DataFrame) -> dict:
    columns = ["R_blend_truth", *[f"R_blend_{tag}" for tag in MODELS]]
    case = frame.groupby("case", sort=True)[columns].mean()
    truth = case["R_blend_truth"].to_numpy(float)
    result = {
        "n_rows": int(len(frame)),
        "n_cases": int(len(case)),
        "case_min": int(case.index.min()),
        "case_max": int(case.index.max()),
        "truth_case_mean": float(truth.mean()),
        "truth_case_sem": finite_sem(truth),
        "models": {},
    }
    for tag in MODELS:
        pred = case[f"R_blend_{tag}"].to_numpy(float)
        gap = pred - truth
        result["models"][tag] = {
            "prediction_case_mean": float(pred.mean()),
            "prediction_minus_truth": float(gap.mean()),
            "gap_case_sem": finite_sem(gap),
            "prediction_over_truth_minus_one": float(pred.mean() / truth.mean() - 1.0),
        }
    baseline_gap = result["models"][BASELINE]["prediction_minus_truth"]
    candidate_gap = result["models"][CANDIDATE]["prediction_minus_truth"]
    result["candidate_vs_baseline"] = {
        "prediction_shift": float(candidate_gap - baseline_gap),
        "absolute_gap_reduction": float(abs(baseline_gap) - abs(candidate_gap)),
        "fraction_of_baseline_deficit_closed": (
            float((candidate_gap - baseline_gap) / -baseline_gap) if baseline_gap != 0 else float("nan")
        ),
    }
    return result


def conditional_table(frame: pd.DataFrame, key: str, edges: np.ndarray) -> list[dict]:
    baseline = frame[f"R_blend_{BASELINE}"].to_numpy(float)
    candidate = frame[f"R_blend_{CANDIDATE}"].to_numpy(float)
    truth = frame["R_blend_truth"].to_numpy(float)
    values = frame[key].to_numpy(float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        if mask.sum() < 100:
            continue
        rows.append({
            "lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
            "truth": float(truth[mask].mean()),
            "baseline": float(baseline[mask].mean()),
            "candidate": float(candidate[mask].mean()),
            "baseline_gap": float((baseline[mask] - truth[mask]).mean()),
            "candidate_gap": float((candidate[mask] - truth[mask]).mean()),
        })
    return rows


def shell_shifts(frame: pd.DataFrame) -> list[dict]:
    rows = []
    prefix = f"R_blend_{BASELINE}_d"
    for base_col in sorted(c for c in frame if c.startswith(prefix)):
        suffix = base_col.removeprefix(f"R_blend_{BASELINE}_")
        cand_col = f"R_blend_{CANDIDATE}_{suffix}"
        if cand_col not in frame:
            continue
        baseline = frame[base_col].to_numpy(float)
        candidate = frame[cand_col].to_numpy(float)
        rows.append({
            "shell": suffix,
            "baseline": float(baseline.mean()),
            "candidate": float(candidate.mean()),
            "candidate_minus_baseline": float((candidate - baseline).mean()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--overlap-audit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")
    frame = pd.concat([pd.read_feather(path) for path in args.inputs], ignore_index=True)
    if frame.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate anchor keys")
    overlap = json.load(open(args.overlap_audit))

    result = {
        "candidate_predeclared": CANDIDATE,
        "baseline": BASELINE,
        "models": MODELS,
        "overlap_audit": overlap,
        "splits": {},
    }
    for name, lo, hi in SPLITS:
        subset = frame[(frame["case"] >= lo) & (frame["case"] <= hi)]
        if len(subset):
            result["splits"][name] = summarize_split(subset)

    final = frame[(frame["case"] >= 200) & (frame["case"] <= 299)].copy()
    if final["case"].nunique() != 100:
        raise RuntimeError("final gate must contain exactly cases 200--299")
    q = np.unique(np.quantile(final[f"R_blend_{BASELINE}"], np.linspace(0, 1, 6)))
    q[0] = np.nextafter(q[0], -np.inf)
    q[-1] = np.nextafter(q[-1], np.inf)
    result["final_gate_diagnostics"] = {
        "by_baseline_blend_quintile": conditional_table(
            final, f"R_blend_{BASELINE}", q,
        ),
        "by_primary_mag": conditional_table(
            final, "r_input_p_plus", np.asarray([18, 22, 23, 24, 25, 25.4, 25.8]),
        ),
        "by_primary_size": conditional_table(
            final, "Re_input_p_plus", np.asarray([0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.5]),
        ),
        "by_pair_count": conditional_table(
            final, "n_pairs", np.asarray([0, 5, 10, 15, 20, 21]),
        ),
        "candidate_minus_baseline_by_distance_shell": shell_shifts(final),
    }
    gate = result["splits"]["final_unseen_latent_200_299"]
    base = gate["models"][BASELINE]
    cand = gate["models"][CANDIDATE]
    result["acceptance"] = {
        "candidate_reduces_absolute_gap": bool(
            abs(cand["prediction_minus_truth"]) < abs(base["prediction_minus_truth"])
        ),
        "candidate_gap_consistent_with_zero_at_2_case_sem": bool(
            abs(cand["prediction_minus_truth"]) <= 2 * cand["gap_case_sem"]
        ),
        "note": (
            "These gates judge the anchor estimand only. Passing does not authorize a numerical "
            "scale correction or establish constgold closure."
        ),
    }

    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    for split, row in result["splits"].items():
        print(f"\n[{split}] N={row['n_rows']:,} cases={row['n_cases']}")
        print(f"  truth={row['truth_case_mean']:+.6f}")
        for tag in MODELS:
            model = row["models"][tag]
            print(
                f"  {tag:34s} pred={model['prediction_case_mean']:+.6f} "
                f"gap={model['prediction_minus_truth']:+.6f} "
                f"+-{model['gap_case_sem']:.6f} "
                f"rel={100*model['prediction_over_truth_minus_one']:+.3f}%"
            )
    print(f"\nacceptance={result['acceptance']}")
    print(f"wrote {args.output}")
    print("ANCHORBLEND_DISTANCE_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
