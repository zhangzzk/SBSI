#!/usr/bin/env python
"""Build keyed ConstGold R_blend predictions from an explicit BlendEMU model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from sbsi.models import ModelPaths, load_emulator


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--input-pattern", required=True, help="format string containing {case}")
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    models = ModelPaths(
        flow_checkpoints=(Path(args.measurement_model),),
        emulator_metadata=Path(args.emulator_metadata),
        emulator_model=Path(args.emulator_model),
    )
    predictor = load_emulator(models, conditions=CONDITIONS, device="cpu")
    parts = []
    for case in args.cases:
        path = Path(args.input_pattern.format(case=case))
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pf.read_table(path).to_pandas()
        frame = frame.rename(columns={name: name.replace("_input", "") for name in frame})
        predicted = predictor.predict_response(frame, frame)
        primary_column = "index_input_p"
        if primary_column not in predicted:
            raise RuntimeError(
                f"response prediction lacks required primary key {primary_column!r}"
            )
        response = predicted.groupby(primary_column, sort=False)["response"].sum()
        parts.append(pd.DataFrame({
            "case": np.full(len(response), case, dtype=np.int64),
            "input_index": response.index.to_numpy(dtype=np.int64),
            "R_blend": response.to_numpy(dtype=np.float64),
        }))
        print(
            f"case {case}: rows={len(response):,}, mean_R_blend={response.mean():.8f}",
            flush=True,
        )
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["case", "input_index"]).any():
        raise RuntimeError("generated lookup contains duplicate keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_feather(output)
    print(f"wrote {output}: rows={len(result):,}, cases={result['case'].nunique()}")


if __name__ == "__main__":
    main()
