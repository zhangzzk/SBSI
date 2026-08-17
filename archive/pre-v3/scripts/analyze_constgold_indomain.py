"""Aggregate the four V2.2 constgold in-domain rerender scores and compare the original cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf


SEEDS = (501, 502, 503, 505)
KEYS = ["case", "input_index"]
VALUES = ["r_sim", "R_flow", "R_blend"]


def load_population(path: str, min_case: int, max_case: int, mag_max: float,
                    re_min: float) -> pd.DataFrame:
    columns = KEYS + ["r_input_p", "Re_input_p"]
    frame = pf.read_table(path, columns=columns).to_pandas()
    frame = frame.loc[(frame["case"] >= min_case) & (frame["case"] < max_case)]
    frame = frame.loc[(frame["r_input_p"] < mag_max) & (frame["Re_input_p"] > re_min)]
    grouped = frame.groupby(KEYS, sort=False)[["r_input_p", "Re_input_p"]]
    spread = grouped.max() - grouped.min()
    if (spread.to_numpy(float) > 1e-12).any():
        raise ValueError(f"primary properties are inconsistent across pair rows in {path}")
    return grouped.first().reset_index()[KEYS]


def load_seed(path: str, population: pd.DataFrame, min_case: int,
              max_case: int) -> pd.DataFrame:
    frame = pf.read_table(path, columns=KEYS + VALUES).to_pandas()
    frame = frame.loc[(frame["case"] >= min_case) & (frame["case"] < max_case)].copy()
    if frame.duplicated(KEYS).any():
        raise ValueError(f"duplicate keys in {path}")
    frame = frame.merge(population, on=KEYS, how="inner", validate="one_to_one")
    return frame.sort_values(KEYS).reset_index(drop=True)


def summarize(paths: list[str], catalogue: str, min_case: int, max_case: int,
              mag_max: float, re_min: float, n_boot: int,
              boot_seed: int) -> dict[str, object]:
    population = load_population(catalogue, min_case, max_case, mag_max, re_min)
    frames = [load_seed(path, population, min_case, max_case) for path in paths]
    reference = frames[0][KEYS]
    for path, frame in zip(paths[1:], frames[1:]):
        if not reference.equals(frame[KEYS]):
            raise ValueError(f"seed key/population mismatch: {path}")
        if not np.allclose(frame["r_sim"], frames[0]["r_sim"], rtol=0, atol=1e-12):
            raise ValueError(f"r_sim differs across seeds: {path}")
        if not np.allclose(frame["R_blend"], frames[0]["R_blend"], rtol=0, atol=1e-12):
            raise ValueError(f"R_blend differs across seeds: {path}")

    r_sim = frames[0]["r_sim"].to_numpy(float)
    r_blend = frames[0]["R_blend"].to_numpy(float)
    flow_stack = np.stack([frame["R_flow"].to_numpy(float) for frame in frames])
    r_flow = flow_stack.mean(axis=0)
    cases = frames[0]["case"].to_numpy(np.int64)
    individual_m = np.array([
        r_sim.mean() / (flow.mean() + r_blend.mean()) - 1.0 for flow in flow_stack
    ])
    ensemble_m = float(r_sim.mean() / (r_flow.mean() + r_blend.mean()) - 1.0)

    unique_cases = np.unique(cases)
    per_case = {
        int(case): (
            float(r_sim[cases == case].sum()),
            float(r_flow[cases == case].sum()),
            float(r_blend[cases == case].sum()),
            int((cases == case).sum()),
        )
        for case in unique_cases
    }
    rng = np.random.default_rng(boot_seed)
    boot = np.empty(n_boot)
    for index in range(n_boot):
        chosen = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        sim_sum = sum(per_case[int(case)][0] for case in chosen)
        flow_sum = sum(per_case[int(case)][1] for case in chosen)
        blend_sum = sum(per_case[int(case)][2] for case in chosen)
        count = sum(per_case[int(case)][3] for case in chosen)
        boot[index] = (sim_sum / count) / ((flow_sum + blend_sum) / count) - 1.0

    case_sem = float(boot.std(ddof=1))
    seed_sem = float(individual_m.std(ddof=1) / np.sqrt(len(individual_m)))
    return {
        "paths": paths,
        "catalogue": catalogue,
        "n_rows": int(len(r_sim)),
        "n_cases": int(len(unique_cases)),
        "means": {
            "R_sim": float(r_sim.mean()),
            "R_flow": float(r_flow.mean()),
            "R_blend": float(r_blend.mean()),
            "R_total": float(r_flow.mean() + r_blend.mean()),
        },
        "m": ensemble_m,
        "m_percent": 100.0 * ensemble_m,
        "individual_seed_m_percent": {
            str(seed): 100.0 * float(value) for seed, value in zip(SEEDS, individual_m)
        },
        "case_bootstrap_sem_percent": 100.0 * case_sem,
        "seed_sem_percent": 100.0 * seed_sem,
        "combined_sem_percent": 100.0 * float(np.hypot(case_sem, seed_sem)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-pattern", required=True, help="format pattern containing {seed}")
    parser.add_argument("--old-pattern", required=True, help="format pattern containing {seed}")
    parser.add_argument("--new-catalogue", required=True)
    parser.add_argument("--old-catalogue", required=True)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument("--max-case", type=int, default=90)
    parser.add_argument("--primary-mag-max", type=float, default=25.8)
    parser.add_argument("--primary-re-min", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=83017)
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    new_paths = [args.new_pattern.format(seed=seed) for seed in SEEDS]
    old_paths = [args.old_pattern.format(seed=seed) for seed in SEEDS]
    for path in new_paths + old_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    new = summarize(new_paths, args.new_catalogue, args.min_case, args.max_case,
                    args.primary_mag_max, args.primary_re_min, args.n_boot, args.seed)
    old = summarize(old_paths, args.old_catalogue, args.min_case, args.max_case,
                    args.primary_mag_max, args.primary_re_min, args.n_boot, args.seed)
    payload: dict[str, object] = {
        "case_range": [args.min_case, args.max_case],
        "primary_selection": {
            "r_input_p_max": args.primary_mag_max,
            "Re_input_p_min": args.primary_re_min,
        },
        "seeds": list(SEEDS),
        "restricted_render": new,
        "original_render": old,
        "delta_m_percent_restricted_minus_original": new["m_percent"] - old["m_percent"],
    }
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as handle:
            payload["filter_manifest"] = json.load(handle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for label, result in (("ORIGINAL", old), ("RESTRICTED", new)):
        means = result["means"]
        print(
            f"{label}: Rsim={means['R_sim']:.6f} Rflow={means['R_flow']:.6f} "
            f"Rblend={means['R_blend']:.6f} m={result['m_percent']:+.4f}% "
            f"+/-case {result['case_bootstrap_sem_percent']:.4f}% "
            f"+/-seed {result['seed_sem_percent']:.4f}%"
        )
    print(f"DELTA restricted-original = {payload['delta_m_percent_restricted_minus_original']:+.4f}%")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
