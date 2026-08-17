import json
import sys

import pandas as pd
import pytest

from scripts.augment_catalogue_lookup import main as augment_main


def _inputs(tmp_path):
    source = tmp_path / "source.feather"
    lookup = tmp_path / "lookup.feather"
    pd.DataFrame({
        "case": [40, 40], "input_index": [1, 2], "value": [10, 20],
    }).to_feather(source)
    pd.DataFrame({
        "case": [40], "input_index": [1], "r_blend": [0.3],
    }).to_feather(lookup)
    return source, lookup


def test_augment_catalogue_lookup_is_strict_by_default(tmp_path, monkeypatch):
    source, lookup = _inputs(tmp_path)
    output = tmp_path / "strict.feather"
    monkeypatch.setattr(sys, "argv", [
        "augment_catalogue_lookup.py", "--catalogue", str(source),
        "--lookup", str(lookup), "--columns", "r_blend", "--output", str(output),
    ])

    with pytest.raises(RuntimeError, match="zero-filling scene context is forbidden"):
        augment_main()


def test_augment_catalogue_lookup_can_inner_join_explicitly(tmp_path, monkeypatch):
    source, lookup = _inputs(tmp_path)
    output = tmp_path / "inner.feather"
    monkeypatch.setattr(sys, "argv", [
        "augment_catalogue_lookup.py", "--catalogue", str(source),
        "--lookup", str(lookup), "--columns", "r_blend", "--drop-unmatched",
        "--output", str(output),
    ])

    augment_main()

    result = pd.read_feather(output)
    assert result.to_dict("list") == {
        "case": [40], "input_index": [1], "value": [10], "r_blend": [0.3],
    }
    with open(output.with_suffix(".json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["input_rows"] == 2
    assert metadata["rows"] == metadata["matched"] == 1
    assert metadata["dropped_unmatched"] == 1
    assert metadata["match_fraction"] == 0.5
