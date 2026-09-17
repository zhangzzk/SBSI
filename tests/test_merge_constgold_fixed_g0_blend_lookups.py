import json

import numpy as np
import pandas as pd
import pytest

from scripts.merge_constgold_fixed_g0_blend_lookups import main


def block(path, cases, *, with_optional=True, rows=4):
    frames = []
    for case in cases:
        data = {
            "case": np.full(rows, case, dtype=np.int32),
            "input_index": np.arange(rows, dtype=np.int64),
            "R_blend": np.linspace(-0.01, 0.02, rows),
        }
        if with_optional:
            data["R_blend_abs"] = np.abs(data["R_blend"]) + 0.005
            data["blend_pairs"] = np.arange(rows, dtype=np.int64) + 1
        frames.append(pd.DataFrame(data))
    pd.concat(frames, ignore_index=True).to_feather(path)
    return path


def test_optional_columns_survive_the_merge(tmp_path):
    # The signed R_blend can sit near zero through cancellation, so the split
    # diagnostic bins on R_blend_abs.  The merge used to drop it silently,
    # which removed that option from every merged lookup.
    a = block(tmp_path / "a.feather", [40, 41])
    b = block(tmp_path / "b.feather", [42])
    out = tmp_path / "merged.feather"
    main(
        ["--input", str(a), "--input", str(b), "--output", str(out)]
        + [arg for case in (40, 41, 42) for arg in ("--case", str(case))]
    )
    merged = pd.read_feather(out)
    assert "R_blend_abs" in merged.columns
    assert "blend_pairs" in merged.columns
    assert len(merged) == 12
    report = json.loads((out.with_suffix(".json")).read_text())
    assert report["optional_columns_dropped"] == []
    assert "R_blend_abs" in report["columns"]


def test_optional_columns_are_dropped_only_when_a_block_lacks_them(tmp_path):
    a = block(tmp_path / "a.feather", [40])
    b = block(tmp_path / "b.feather", [41], with_optional=False)
    out = tmp_path / "merged.feather"
    main(
        ["--input", str(a), "--input", str(b), "--output", str(out)]
        + [arg for case in (40, 41) for arg in ("--case", str(case))]
    )
    merged = pd.read_feather(out)
    assert list(merged.columns) == ["case", "input_index", "R_blend"]
    report = json.loads((out.with_suffix(".json")).read_text())
    assert report["optional_columns_dropped"] == ["R_blend_abs", "blend_pairs"]


def test_a_block_missing_a_required_column_is_rejected(tmp_path):
    a = block(tmp_path / "a.feather", [40])
    bad = tmp_path / "bad.feather"
    pd.DataFrame({"case": [41], "input_index": [0]}).to_feather(bad)
    out = tmp_path / "merged.feather"
    with pytest.raises(ValueError, match="missing required columns"):
        main(
            ["--input", str(a), "--input", str(bad), "--output", str(out)]
            + [arg for case in (40, 41) for arg in ("--case", str(case))]
        )


def test_an_absolute_sum_below_the_signed_sum_is_rejected(tmp_path):
    path = tmp_path / "a.feather"
    pd.DataFrame(
        {
            "case": np.int32([40, 40]),
            "input_index": np.int64([0, 1]),
            "R_blend": [0.02, -0.03],
            "R_blend_abs": [0.03, 0.01],  # the second is below |signed|
            "blend_pairs": np.int64([1, 2]),
        }
    ).to_feather(path)
    out = tmp_path / "merged.feather"
    with pytest.raises(ValueError, match="absolute blending sum"):
        main(["--input", str(path), "--output", str(out), "--case", "40"])
