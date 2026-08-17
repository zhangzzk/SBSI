import json
import sys

import pandas as pd

from scripts.analyze_anchorblend_random_multilayer import main


def test_multilayer_analysis_combines_disjoint_keys(tmp_path, monkeypatch):
    arguments = ["prog"]
    for layer in range(2):
        root10 = tmp_path / f"l{layer}_r10"
        root15 = tmp_path / f"l{layer}_r15"
        root10.mkdir(); root15.mkdir()
        rows10 = []
        rows15 = []
        for case in range(200, 240):
            index = layer * 1000 + case
            manifest = pd.DataFrame({"index": [index], "u1": [1.0], "u2": [0.0]})
            manifest.to_feather(root10 / f"anchors_case{case}.feather")
            manifest.to_feather(root15 / f"anchors_case{case}.feather")
            common = {
                "case": case, "input_index": index, "R_blend_null": 0.0,
                "R_blend_lsst_r_extnbr_v22": 0.1,
                "r_input_p_plus": 24.0, "Re_input_p_plus": 0.8,
            }
            rows10.append({**common, "R_blend_truth": 0.1})
            rows15.append({**common, "R_blend_truth": 0.102})
        path10 = tmp_path / f"response_l{layer}_r10.feather"
        path15 = tmp_path / f"response_l{layer}_r15.feather"
        pd.DataFrame(rows10).to_feather(path10)
        pd.DataFrame(rows15).to_feather(path15)
        arguments += [
            "--response10", str(path10), "--response15", str(path15),
            "--manifest-root10", str(root10), "--manifest-root15", str(root15),
        ]
    output = tmp_path / "result.json"
    arguments += [
        "--case-min", "200", "--case-max", "239", "--development-max", "219",
        "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    payload = json.loads(output.read_text())
    assert payload["n_layers"] == 2
    assert payload["n_common_rows"] == 80
    assert payload["truth_local10"]["mean"] == 0.1
    assert abs(payload["shell_10_15"]["mean"] - 0.002) < 1.0e-12
    assert payload["constgold_opened"] is False
