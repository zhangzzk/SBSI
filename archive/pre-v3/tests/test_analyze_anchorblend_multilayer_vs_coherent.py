import json
import sys

import pandas as pd

from scripts.analyze_anchorblend_multilayer_vs_coherent import main


def test_multilayer_exact_comparison_combines_disjoint_keys(tmp_path, monkeypatch):
    coherent_rows = []
    arguments = ["prog", "--case-min", "200", "--case-max", "239"]
    for layer in range(2):
        root = tmp_path / f"layer{layer}"
        root.mkdir()
        random10_rows = []
        random15_rows = []
        for case in range(200, 240):
            index = layer * 1000 + case
            pd.DataFrame({"index": [index]}).to_feather(
                root / f"anchors_case{case}.feather"
            )
            base = {
                "case": case, "input_index": index,
                "R_blend_lsst_r_extnbr_v22": 0.1,
            }
            coherent_rows.append({**base, "R_blend_truth": 0.09})
            coherent_rows.append({
                "case": case, "input_index": 10000 + layer * 1000 + case,
                "R_blend_lsst_r_extnbr_v22": 0.1, "R_blend_truth": 0.08,
            })
            random10_rows.append({
                **base, "R_blend_truth": 0.095, "R_blend_null": 0.0,
            })
            random15_rows.append({
                **base, "R_blend_truth": 0.097, "R_blend_null": 0.0,
            })
        path10 = tmp_path / f"random10_l{layer}.feather"
        path15 = tmp_path / f"random15_l{layer}.feather"
        pd.DataFrame(random10_rows).to_feather(path10)
        pd.DataFrame(random15_rows).to_feather(path15)
        arguments += [
            "--random10-response", str(path10),
            "--random15-response", str(path15),
            "--manifest-root", str(root),
        ]
    coherent_path = tmp_path / "coherent.feather"
    pd.DataFrame(coherent_rows).to_feather(coherent_path)
    output = tmp_path / "result.json"
    arguments += [
        "--coherent-response", str(coherent_path), "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    payload = json.loads(output.read_text())
    assert payload["n_layers"] == 2
    assert payload["n_exact_common_rows"] == 80
    assert abs(payload["exact_common_coherent_minus_random10"]["mean"] + 0.005) < 1e-12
    assert abs(payload["exact_common_random_shell_10_15"]["mean"] - 0.002) < 1e-12
    assert payload["constgold_opened"] is False
