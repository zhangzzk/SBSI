#!/usr/bin/env python3
"""Prepare truth-parent ConstGold keys without requiring a usable g0 anchor.

Reuse the existing full-parent classifier contexts. Keep the declared true
magnitude limit and record g0 membership only as a diagnostic mask, never as
a parent selection. No image measurements are invented for undetected rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.fixed_g0_domain import FLOW_FEATURES
from scripts.train_fixed_g0_flow import sha256, write_json


def parent_case(prepared, zero, magnitude_max=26.0, *, truth_magnitude=None):
    ids = np.asarray(prepared["ids"], dtype=np.int64)
    context = np.asarray(prepared["x0"], dtype=np.float32)
    if (ids.ndim != 1 or context.shape != (len(ids), len(FLOW_FEATURES))
            or len(np.unique(ids)) != len(ids) or not np.isfinite(context).all()
            or not np.isfinite(magnitude_max)):
        raise ValueError("invalid truth-parent contexts or magnitude boundary")
    magnitude = (context[:, FLOW_FEATURES.index("r_input_p")]
                 if truth_magnitude is None else np.asarray(truth_magnitude))
    if magnitude.shape != ids.shape or not np.isfinite(magnitude).all():
        raise ValueError("invalid aligned truth magnitudes")
    keep = magnitude < magnitude_max
    ids, context = ids[keep], context[keep]
    order = np.argsort(ids)
    ids, context = ids[order], context[order]
    if not len(ids):
        raise ValueError("empty truth parent")
    zero_ids = np.asarray(zero["input_index"], dtype=np.int64)
    if len(np.unique(zero_ids)) != len(zero_ids):
        raise ValueError("duplicate zero-shear identities")
    location = np.searchsorted(ids, zero_ids)
    matched = (location < len(ids)) & (ids[np.minimum(location, len(ids) - 1)] == zero_ids)
    location = location[matched]
    if not np.allclose(context[location], zero["context"][matched], rtol=2e-6, atol=2e-6):
        raise ValueError("truth-parent contexts differ from the valid-g0 domain")
    usable = np.zeros(len(ids), dtype=bool)
    selected = np.zeros(len(ids), dtype=bool)
    usable[location] = True
    target = np.asarray(zero["target"])[matched]
    selected[location] = ((target[:, 2] > 3.0)
                          & (target[:, 3] > 10 ** (-0.4 * (25.8 - 30))))
    result = {"input_index": ids, "context": context,
              "gamma": np.zeros((len(ids), 2)),
              "g0_usable": usable, "g0_selected": selected}
    counts = {"input_parent_rows": len(keep), "truth_parent_rows": len(ids),
              "truth_cut_dropped": int((~keep).sum()),
              "g0_usable": int(usable.sum()), "g0_selected": int(selected.sum()),
              "g0_keys_outside_truth_parent_dropped": int((~matched).sum()),
              "g0_unusable_retained": int((~usable).sum())}
    return result, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-prepared", type=Path, required=True)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--truth-pattern", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("parent preparation must run under Slurm")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("duplicate requested cases")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    manifest_path = args.classifier_prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["features"] != list(FLOW_FEATURES) or not set(args.cases) <= set(manifest["cases"]):
        raise ValueError("full-parent source lacks requested features/cases")
    args.output_root.mkdir(parents=True)
    reports = {}
    for case in args.cases:
        source = args.classifier_prepared / "cases" / f"case{case:03d}.npz"
        if sha256(source) != manifest["case_records"][str(case)]["prepared_sha256"]:
            raise ValueError(f"classifier parent hash mismatch: case{case}")
        zero_path = args.domain_root / "flow" / "g0" / f"case{case:03d}.npz"
        truth_path = Path(args.truth_pattern.format(case=case))
        truth = pd.read_feather(truth_path, columns=["index_input", "r_input"])
        if truth["index_input"].duplicated().any():
            raise ValueError("duplicate truth source key")
        with np.load(source, allow_pickle=False) as prepared, np.load(zero_path, allow_pickle=False) as zero:
            magnitude = truth.set_index("index_input")["r_input"].reindex(prepared["ids"]).to_numpy()
            arrays, counts = parent_case(prepared, zero, truth_magnitude=magnitude)
        output = args.output_root / f"case{case:03d}.npz"
        np.savez(output, **arrays)
        reports[str(case)] = counts | {"source_sha256": sha256(source),
                                       "g0_sha256": sha256(zero_path), "sha256": sha256(output),
                                       "truth_sha256": sha256(truth_path)}
        print(f"TRUTH_PARENT_PREPARED case={case} counts={counts}", flush=True)
    write_json(args.output_root / "manifest.json", {
        "population": "complete half-shear target-role truth parent; strict r_input_p<26; no measured selection or g0 usability requirement",
        "features": list(FLOW_FEATURES), "cases": args.cases, "case_reports": reports,
        "classifier_manifest_sha256": sha256(manifest_path),
        "domain_manifest_sha256": sha256(args.domain_root / "manifest.json"),
        "script_sha256": sha256(Path(__file__)),
        "limitations": ["same finite scene development cases as previous ConstGold tests",
                        "g0 masks are diagnostic only; parent rows do not have fabricated measurements"],
    })


if __name__ == "__main__":
    main()
