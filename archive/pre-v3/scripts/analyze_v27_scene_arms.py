"""Compare directional scene-conditioner arms with matched V2.2 baselines."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


ARMS = (
    "sixshell", "purity", "rblend",
    "rb5_all", "rbc8_s6_cap4m", "rbc8_s6_all", "rbc8_s4_cap4m", "rbc8_s4_all",
    "v30_m10_c16", "v30_m12_c16",
)
DEFAULT_SEEDS = (501, 502)
KEYS = ["case", "input_index"]
SHELLS = [
    "logflux_abs_shell_0_0p5", "logflux_abs_shell_0p5_1",
    "logflux_abs_shell_1_2", "logflux_abs_shell_2_3",
    "logflux_abs_shell_3_5", "logflux_abs_shell_5_10",
]


def axes_for_arm(arm: str) -> list[str]:
    if arm == "sixshell":
        return SHELLS
    if arm == "purity":
        return ["true_blendedness_eq17"]
    if arm.startswith("rb5_") or arm.startswith("rbc8_") or arm.startswith("v30_"):
        return ["r_blend", "r_input_p", "Re_input_p"]
    return ["r_blend"]


def profile_bins(arm: str, axis: str) -> int:
    if arm == "sixshell":
        return 4
    if axis == "Re_input_p":
        return 4
    return 5


def mean_sem(values: list[float]) -> dict:
    value = np.asarray(values, float)
    return {
        "mean": float(value.mean()), "seed_values": value.tolist(),
        "seed_sem": float(value.std(ddof=1) / np.sqrt(len(value))) if len(value) > 1 else None,
    }


def read_unique(paths: list[str]) -> pd.DataFrame:
    out = pd.concat([pd.read_feather(path) for path in sorted(paths)], ignore_index=True)
    if out.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate keys across {paths}")
    return out


def merge_halfshear_frames(new: pd.DataFrame, old: pd.DataFrame,
                           new_flow_column: str, old_flow_column: str) -> pd.DataFrame:
    """Merge matched dumps without colliding same-named seed columns."""
    return new[KEYS + ["r_sim_self", new_flow_column]].rename(
        columns={new_flow_column: "variant"}).merge(
        old[KEYS + [old_flow_column]].rename(columns={old_flow_column: "baseline"}),
        on=KEYS, validate="one_to_one")


def finite_halfshear_rows(frame: pd.DataFrame) -> np.ndarray:
    """Rows on which the self-response comparison is mathematically defined."""
    return np.isfinite(frame[["r_sim_self", "variant", "baseline"]].to_numpy(float)).all(axis=1)


def bins(values: np.ndarray, nbin: int) -> np.ndarray:
    """Quantile-bin without splitting tied feature values across bins.

    Sparse shell fluxes have a genuine point mass at zero.  Dropping repeated
    quantile edges preserves that population as one bin; rank-based binning
    would instead divide identical empty-shell objects arbitrarily.
    """
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite profile feature")
    edges = np.unique(np.quantile(values, np.arange(1, nbin) / nbin))
    return np.digitize(values, edges, right=True)


def profile(frame: pd.DataFrame, feature: str, nbin: int) -> list[dict]:
    group = bins(frame[feature].to_numpy(float), nbin)
    rows = []
    for index in range(int(group.max()) + 1):
        use = group == index
        truth = frame.loc[use, "truth"].to_numpy(float)
        baseline = frame.loc[use, "baseline"].to_numpy(float)
        variant = frame.loc[use, "variant"].to_numpy(float)
        rows.append({
            "bin": index + 1, "n": int(use.sum()),
            "feature_median": float(frame.loc[use, feature].median()),
            "baseline_minus_truth": float(baseline.mean() - truth.mean()),
            "variant_minus_truth": float(variant.mean() - truth.mean()),
            "paired_change": float(variant.mean() - baseline.mean()),
        })
    return rows


def blendness_profile(seed_frames: list[pd.DataFrame], blend_eps: float = 0.02) -> dict:
    """Direct total-model bias in the established V2.2 ``R_blend`` bins.

    The low-response class and positive-response quartiles match
    ``diag_v22_blendness.py``.  Simulation sampling uncertainty is blocked by
    case; the paired model change contains no simulation term because both
    models use the same rows, ``R_sim`` and ``R_blend``.
    """
    if not seed_frames:
        raise RuntimeError("no constgold seed frames for blendness profile")
    frames = [frame.sort_values(KEYS).reset_index(drop=True) for frame in seed_frames]
    base = frames[0]
    base_keys = base[KEYS].to_numpy()
    rb = base["R_blend"].to_numpy(float)
    if not np.isfinite(rb).all():
        raise RuntimeError("non-finite R_blend in constgold profile")
    for frame in frames[1:]:
        if not np.array_equal(frame[KEYS].to_numpy(), base_keys):
            raise RuntimeError("constgold keys differ across seeds")
        if not np.array_equal(frame["R_blend"].to_numpy(float), rb):
            raise RuntimeError("constgold R_blend differs across seeds")
        if not np.array_equal(frame["truth"].to_numpy(float), base["truth"].to_numpy(float)):
            raise RuntimeError("constgold flow-demand truth differs across seeds")

    positive = rb >= blend_eps
    edges = np.quantile(rb[positive], [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[-1] = np.nextafter(edges[-1], np.inf)
    masks = [rb < blend_eps]
    labels = [f"Rbl<{blend_eps:.2f}"]
    for index in range(4):
        masks.append((rb >= edges[index]) & (rb < edges[index + 1]))
        labels.append(f"q{index + 1}")

    cases = base["case"].to_numpy(np.int64)
    r_sim = base["truth"].to_numpy(float) + rb
    rows = []
    for index, (label, use) in enumerate(zip(labels, masks)):
        if not use.any():
            raise RuntimeError(f"empty R_blend bin {label}")
        candidate_m, baseline_m = [], []
        candidate_additive, baseline_additive = [], []
        for frame in frames:
            demand = frame.loc[use, "truth"].to_numpy(float).mean()
            candidate = frame.loc[use, "variant"].to_numpy(float).mean()
            baseline = frame.loc[use, "baseline"].to_numpy(float).mean()
            blend = rb[use].mean()
            sim = demand + blend
            candidate_m.append(sim / (candidate + blend) - 1.0)
            baseline_m.append(sim / (baseline + blend) - 1.0)
            candidate_additive.append(candidate - demand)
            baseline_additive.append(baseline - demand)

        candidate_summary = mean_sem(candidate_m)
        baseline_summary = mean_sem(baseline_m)
        paired_summary = mean_sem((np.asarray(candidate_m) - np.asarray(baseline_m)).tolist())
        # Whole cases are independent render/noise blocks.  This matches the
        # established blendness diagnostic rather than treating millions of
        # galaxies within the same case as independent.
        per_case_rsim = pd.DataFrame({"case": cases[use], "r_sim": r_sim[use]}).groupby(
            "case", sort=False)["r_sim"].mean().to_numpy(float)
        sim_sem = float(per_case_rsim.std(ddof=1) / np.sqrt(len(per_case_rsim)))
        candidate_total = np.mean([
            frame.loc[use, "variant"].to_numpy(float).mean() + rb[use].mean()
            for frame in frames
        ])
        candidate_sim_sem = sim_sem / candidate_total
        candidate_seed_sem = candidate_summary["seed_sem"]
        candidate_total_sem = (candidate_sim_sem if candidate_seed_sem is None else
                               np.hypot(candidate_seed_sem, candidate_sim_sem))
        rows.append({
            "bin": index + 1, "label": label, "n": int(use.sum()),
            "low_inclusive": None if index == 0 else float(edges[index - 1]),
            "high_exclusive": float(blend_eps) if index == 0 else float(edges[index]),
            "R_blend_mean": float(rb[use].mean()),
            "m_variant": candidate_summary,
            "m_variant_sim_sem": float(candidate_sim_sem),
            "m_variant_total_sem": float(candidate_total_sem),
            "m_baseline": baseline_summary,
            "paired_delta_m": paired_summary,
            "flow_minus_demand_variant": mean_sem(candidate_additive),
            "flow_minus_demand_baseline": mean_sem(baseline_additive),
        })
    return {"blend_eps": blend_eps, "positive_quantile_edges": edges.tolist(), "bins": rows}


def halfshear(args, features: pd.DataFrame) -> dict:
    result = {}
    baseline_dir = args.halfshear_baseline_dir
    for arm in args.arms:
        seed_stats = []
        profile_frames = []
        for seed in args.seeds:
            new = pd.read_feather(os.path.join(args.halfshear_dir, f"{arm}_s{seed}.feather"))
            baseline_path = args.halfshear_baseline_template.format(seed=seed)
            if not os.path.isabs(baseline_path):
                baseline_path = os.path.join(baseline_dir, baseline_path)
            old = pd.read_feather(baseline_path)
            ncol = [c for c in new if c.startswith("R_flow_s")]
            ocol = [c for c in old if c.startswith("R_flow_s")]
            if len(ncol) != 1 or len(ocol) != 1:
                raise RuntimeError("expected one flow column per half-shear seed file")
            merged = merge_halfshear_frames(new, old, ncol[0], ocol[0])
            merged = merged.merge(features, on=KEYS, how="left", validate="one_to_one")
            axes = axes_for_arm(arm)
            if merged[axes].isna().any().any():
                raise RuntimeError("half-shear feature coverage incomplete")
            n_total = len(merged)
            merged = merged.loc[finite_halfshear_rows(merged)].copy()
            if merged.empty:
                raise RuntimeError("no finite half-shear comparisons")
            truth = merged["r_sim_self"].to_numpy(float)
            variant = merged["variant"].to_numpy(float)
            baseline = merged["baseline"].to_numpy(float)
            seed_stats.append({
                "seed": seed, "n": len(merged), "n_total": n_total,
                "finite_fraction": float(len(merged) / n_total),
                "R_truth": float(truth.mean()),
                "R_baseline": float(baseline.mean()), "R_variant": float(variant.mean()),
                "baseline_ratio_minus_one": float(baseline.mean() / truth.mean() - 1.0),
                "variant_ratio_minus_one": float(variant.mean() / truth.mean() - 1.0),
                "paired_response_change": float(variant.mean() - baseline.mean()),
            })
            profile_frames.append(pd.DataFrame({
                **{c: merged[c].to_numpy() for c in KEYS + axes},
                "truth": truth, "baseline": baseline, "variant": variant,
            }))
        stacked = pd.concat(profile_frames, ignore_index=True).groupby(KEYS, as_index=False).mean()
        result[arm] = {
            "seeds": seed_stats,
            "variant_ratio_minus_one": mean_sem([s["variant_ratio_minus_one"] for s in seed_stats]),
            "paired_response_change": mean_sem([s["paired_response_change"] for s in seed_stats]),
            "profiles": {axis: profile(stacked, axis, profile_bins(arm, axis))
                         for axis in axes},
        }
    return result


def constgold(args, features: pd.DataFrame) -> dict:
    result = {}
    for arm in args.arms:
        seed_stats = []
        profile_frames = []
        for seed in args.seeds:
            new_paths = glob.glob(os.path.join(args.constgold_dir, f"{arm}_s{seed}_c*.feather"))
            old_paths = glob.glob(os.path.join(args.constgold_baseline_dir,
                                               f"*v22_perobj_s{seed}_c*.feather"))
            if len(new_paths) != 10 or len(old_paths) != 10:
                raise RuntimeError(f"expected ten constgold shards for {arm} seed {seed}")
            new = read_unique(new_paths)
            old = read_unique(old_paths)
            merged = new.merge(old[KEYS + ["R_flow"]], on=KEYS, validate="one_to_one",
                               suffixes=("_new", "_old"))
            axes = axes_for_arm(arm)
            # The scorer writes the exact deployed R_blend, including zero for
            # an object absent from the sparse emulator lookup.  Reuse that
            # value for every R_blend-conditioned arm rather than dropping the
            # object by rejoining the sparse lookup.  Dumps also carry
            # r_input_p; merge only genuinely missing primary axes.
            if "r_blend" in axes:
                merged["r_blend"] = merged["R_blend"].to_numpy(float)
            missing_axes = [axis for axis in axes if axis not in merged.columns]
            if missing_axes:
                merged = merged.merge(features[KEYS + missing_axes], on=KEYS, how="left",
                                      validate="one_to_one")
            truth = merged["r_sim"].to_numpy(float)
            blend = merged["R_blend"].to_numpy(float)
            variant = merged["R_flow_new"].to_numpy(float)
            baseline = merged["R_flow_old"].to_numpy(float)
            m_new = truth.mean() / (variant.mean() + blend.mean()) - 1.0
            m_old = truth.mean() / (baseline.mean() + blend.mean()) - 1.0
            seed_stats.append({
                "seed": seed, "n": len(merged), "R_sim": float(truth.mean()),
                "R_blend": float(blend.mean()), "R_baseline": float(baseline.mean()),
                "R_variant": float(variant.mean()), "m_baseline": float(m_old),
                "m_variant": float(m_new), "paired_delta_m": float(m_new - m_old),
            })
            if merged[axes].isna().any().any():
                raise RuntimeError("constgold feature coverage incomplete")
            profile_frames.append(pd.DataFrame({
                **{c: merged[c].to_numpy() for c in KEYS + axes},
                "R_blend": blend, "truth": truth - blend,
                "baseline": baseline, "variant": variant,
            }))
        stacked = pd.concat(profile_frames, ignore_index=True).groupby(KEYS, as_index=False).mean()
        result[arm] = {
            "seeds": seed_stats,
            "m_variant": mean_sem([s["m_variant"] for s in seed_stats]),
            "paired_delta_m": mean_sem([s["paired_delta_m"] for s in seed_stats]),
            "profile_by_R_blend": blendness_profile(profile_frames),
            "profiles_flow_minus_self_truth": {
                axis: profile(stacked, axis, profile_bins(arm, axis)) for axis in axes},
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-half", required=True)
    ap.add_argument("--features-const")
    ap.add_argument("--halfshear-dir", required=True)
    ap.add_argument("--halfshear-baseline-dir", required=True)
    ap.add_argument("--halfshear-baseline-template", default="part_s{seed}.feather",
                    help="baseline filename template, relative to baseline-dir unless absolute")
    ap.add_argument("--constgold-dir")
    ap.add_argument("--constgold-baseline-dir")
    ap.add_argument("--skip-constgold", action="store_true",
                    help="analyze half-shear only, without reading constgold inputs")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                    help="scene arms to analyze; defaults to both")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help="matched flow seeds; defaults to 501 502")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be a non-empty unique list")
    if not args.skip_constgold and not all(
            [args.features_const, args.constgold_dir, args.constgold_baseline_dir]):
        ap.error("constgold inputs are required unless --skip-constgold is set")
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    columns = KEYS + list(dict.fromkeys(
        feature for arm in args.arms for feature in axes_for_arm(arm)))
    features_half = pd.read_feather(args.features_half, columns=columns)
    result = {
        "status": (f"directional screen with {len(args.seeds)} matched seeds; "
                   "not a final absolute-bias certification"),
        "arms": args.arms,
        "seeds": list(args.seeds), "halfshear": halfshear(args, features_half),
    }
    if not args.skip_constgold:
        # Constgold score dumps already contain r_input_p and exact deployed
        # R_blend.  The feature lookup is only needed for other profile axes.
        const_columns = KEYS + [column for column in columns[len(KEYS):]
                                if column not in {"r_blend", "r_input_p"}]
        features_const = pd.read_feather(args.features_const, columns=const_columns)
        result["constgold"] = constgold(args, features_const)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for arm in args.arms:
        hs = result["halfshear"][arm]
        line = (f"{arm}: half-shear flow/truth-1 = "
                f"{100*hs['variant_ratio_minus_one']['mean']:+.3f}%")
        if not args.skip_constgold:
            cg = result["constgold"][arm]
            line += (f" ; constgold m = {100*cg['m_variant']['mean']:+.3f}% ; "
                     f"paired delta m vs V2.2 = "
                     f"{100*cg['paired_delta_m']['mean']:+.3f}%")
        print(line)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
