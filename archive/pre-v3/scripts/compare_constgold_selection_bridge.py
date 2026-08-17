"""Paired comparison of historical and replacement-R_blend selection dumps."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def load(pattern: str) -> dict[str, dict]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no dumps match {pattern!r}")
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            dump = json.load(handle)
        checkpoint = dump["ckpt"]
        if checkpoint in out:
            raise SystemExit(f"duplicate checkpoint {checkpoint} in {pattern!r}")
        out[checkpoint] = dump
    return out


def sem(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="glob for historical per-seed dumps")
    ap.add_argument("--candidate", required=True, help="glob for replacement per-seed dumps")
    ap.add_argument("--output-json")
    ap.add_argument(
        "--max-flow-drift", type=float, default=1e-4,
        help="maximum tolerated absolute repeat drift in a flow response (default 1e-4)",
    )
    args = ap.parse_args()

    old, new = load(args.baseline), load(args.candidate)
    if set(old) != set(new):
        raise SystemExit(f"checkpoint sets differ: old={sorted(old)} new={sorted(new)}")
    checkpoints = sorted(old)
    if len(checkpoints) < 16:
        raise SystemExit(f"expected the 16-seed e-response ensemble, found {len(checkpoints)}")

    reference_old, reference_new = old[checkpoints[0]], new[checkpoints[0]]
    for key in ("n", "case_sum", "idx_sum", "g", "n_all"):
        if reference_old["fingerprint"][key] != reference_new["fingerprint"][key]:
            raise SystemExit(
                f"population drift in fingerprint[{key!r}]: "
                f"{reference_old['fingerprint'][key]} vs {reference_new['fingerprint'][key]}"
            )
    old_cuts = [item["name"] for item in reference_old["cuts"]]
    new_cuts = [item["name"] for item in reference_new["cuts"]]
    if old_cuts != new_cuts:
        raise SystemExit(f"cut lists differ: old={old_cuts} new={new_cuts}")
    keys = ["__nocut__", *old_cuts]

    # The lookup is not a flow conditioner, so flow scoring should repeat up to heterogeneous-GPU
    # numerical drift. Refuse if that drift is large enough to compromise the R_blend comparison.
    max_flow_drift = 0.0
    for checkpoint in checkpoints:
        for key in keys:
            max_flow_drift = max(
                max_flow_drift,
                abs(old[checkpoint]["rf"]["ALL"][key] - new[checkpoint]["rf"]["ALL"][key]),
            )
    if max_flow_drift > args.max_flow_drift:
        raise SystemExit(f"flow contribution drifted by {max_flow_drift:.3g}; not an isolated swap")

    rows = []
    for key in keys:
        old_m, new_m, old_dm, new_dm = [], [], [], []
        for checkpoint in checkpoints:
            od, nd = old[checkpoint], new[checkpoint]
            sim = float(od["sim"]["ALL"][key]["measured"])
            sim0 = float(od["sim"]["ALL"]["__nocut__"]["measured"])
            if sim != float(nd["sim"]["ALL"][key]["measured"]):
                raise SystemExit(f"simulation response drift at {key} for {checkpoint}")
            ot = od["rf"]["ALL"][key] + od["rb"]["ALL"][key]
            nt = nd["rf"]["ALL"][key] + nd["rb"]["ALL"][key]
            ot0 = od["rf"]["ALL"]["__nocut__"] + od["rb"]["ALL"]["__nocut__"]
            nt0 = nd["rf"]["ALL"]["__nocut__"] + nd["rb"]["ALL"]["__nocut__"]
            om = 100.0 * (sim / ot - 1.0)
            nm = 100.0 * (sim / nt - 1.0)
            om0 = 100.0 * (sim0 / ot0 - 1.0)
            nm0 = 100.0 * (sim0 / nt0 - 1.0)
            old_m.append(om)
            new_m.append(nm)
            old_dm.append(om - om0)
            new_dm.append(nm - nm0)
        old_m = np.asarray(old_m)
        new_m = np.asarray(new_m)
        old_dm = np.asarray(old_dm)
        new_dm = np.asarray(new_dm)
        row = {
            "cut": key,
            "m_old_pct": float(old_m.mean()),
            "m_new_pct": float(new_m.mean()),
            "delta_m_pct": float((new_m - old_m).mean()),
            "delta_m_sem_pct": sem(new_m - old_m),
            "dm_old_pct": float(old_dm.mean()),
            "dm_new_pct": float(new_dm.mean()),
            "delta_dm_pct": float((new_dm - old_dm).mean()),
            "delta_dm_sem_pct": sem(new_dm - old_dm),
        }
        rows.append(row)

    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "n_seeds": len(checkpoints),
        "n_rows": reference_new["fingerprint"]["n_all"],
        "old_rblend_sum": reference_old["fingerprint"]["rb_sum"],
        "new_rblend_sum": reference_new["fingerprint"]["rb_sum"],
        "max_flow_drift": max_flow_drift,
        "rows": rows,
    }
    print(
        f"paired bridge: {len(checkpoints)} seeds, {result['n_rows']:,} rows; "
        f"max flow drift={max_flow_drift:.3g}"
    )
    print(
        f"{'cut':>24} | {'m old':>8} {'m new':>8} {'delta':>13} | "
        f"{'dm old':>8} {'dm new':>8} {'delta':>13}"
    )
    for row in rows:
        print(
            f"{row['cut']:>24} | {row['m_old_pct']:+8.3f} {row['m_new_pct']:+8.3f} "
            f"{row['delta_m_pct']:+8.3f}+-{row['delta_m_sem_pct']:.3f} | "
            f"{row['dm_old_pct']:+8.3f} {row['dm_new_pct']:+8.3f} "
            f"{row['delta_dm_pct']:+8.3f}+-{row['delta_dm_sem_pct']:.3f}"
        )
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
