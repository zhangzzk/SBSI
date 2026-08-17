#!/usr/bin/env python3
"""Reproduce and save the selected sequential model's exact OOF prediction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.tune_crossfit_sequential_rblend_optuna import (
    CrossfitExperiment,
    json_clean,
)
from scripts.v22_grouped_rscene_common import sha256, strict_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    output_path = Path(args.output).resolve()
    audit_path = output_path.with_suffix(".json")
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("refusing existing OOF export")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment = CrossfitExperiment(
        Path(summary["source"]["cache"]),
        Path(summary["source"]["scene_cache"]),
    )
    metrics, prediction = experiment.fit_crossfit(
        summary["best_params"], keep_oof=True
    )
    if prediction is None or prediction.shape != (experiment.n_rows,):
        raise RuntimeError("selected OOF prediction was not returned")
    expected = summary["best_oof_metrics"]["oof_pair_metrics"]
    actual = metrics["oof_pair_metrics"]
    checks = {}
    for key in ("r2", "mse", "case_balanced_residual_mean"):
        difference = abs(float(actual[key]) - float(expected[key]))
        checks[key] = difference
        if difference > 1.0e-5:
            raise RuntimeError(
                f"reproduced {key} differs from summary by {difference:.3e}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.npy"
    )
    np.save(temporary, prediction.astype(np.float32))
    os.replace(temporary, output_path)
    payload = {
        "schema_version": 1,
        "objective_variant": summary["objective_variant"],
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "output": str(output_path),
        "n_rows": int(len(prediction)),
        "dtype": "float32",
        "metric_reproduction_absolute_difference": checks,
        "metrics": json_clean(metrics),
        "coherent_anchor_truth_opened": False,
        "constgold_opened": False,
    }
    strict_json(audit_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("SELECTED_CROSSFIT_OOF_EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
