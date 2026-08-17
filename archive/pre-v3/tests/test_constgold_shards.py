import sys

import pandas as pd
import pyarrow.feather as pf

from scripts.concat_response_shards import main as concat_main
from scripts.validate_constant_with_blend import _case_filtered_table


def test_case_filtered_table_filters_before_row_limit(tmp_path):
    path = tmp_path / "input.feather"
    pd.DataFrame({"case": [0, 0, 1, 1, 2, 2], "value": range(6)}).to_feather(path)

    got = _case_filtered_table(
        path, ["case", "value"], min_case=1, max_case=3, max_rows=3
    ).to_pandas()

    assert got.to_dict("list") == {"case": [1, 1, 2], "value": [2, 3, 4]}


def test_concat_response_shards_checks_and_preserves_case_order(tmp_path, monkeypatch):
    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "full"
    shard_dir.mkdir()
    tag = "test_model"
    for lo, hi in ((40, 42), (42, 44)):
        pd.DataFrame({
            "case": [lo, hi - 1],
            "input_index": [lo * 10, (hi - 1) * 10],
            "r_sim": [0.8, 0.9],
            "R_flow": [0.7, 0.8],
            "R_blend": [0.1, 0.1],
        }).to_feather(shard_dir / f"{tag}_perobj_s501_c{lo:03d}-{hi:03d}.feather")

    monkeypatch.setattr(sys, "argv", [
        "concat_response_shards.py",
        "--shard-dir", str(shard_dir),
        "--output-dir", str(output_dir),
        "--tag", tag,
        "--seeds", "501",
        "--min-case", "40",
        "--max-case", "44",
        "--cases-per-shard", "2",
    ])
    concat_main()

    out = pf.read_table(output_dir / f"{tag}_perobj_s501.feather").to_pandas()
    assert out["case"].tolist() == [40, 41, 42, 43]
    assert out["input_index"].tolist() == [400, 410, 420, 430]
