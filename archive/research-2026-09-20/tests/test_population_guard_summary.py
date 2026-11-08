import json
import sys

import numpy as np
import pytest

from plots.constgold_property_bias import BRANCHES, PROPERTY_ORDER
from scripts.summarize_population_guard_ab import main


def make_products(root):
    denominator = np.array([[10.0], [20.0], [30.0], [40.0]])
    response = np.array([[0.7], [0.9], [1.1], [1.3]])
    for arm in ("shared", "independent"):
        (root / arm).mkdir()
        (root / arm / "completed.json").write_text("{}")
    for cohort in ("uncut", "fixed_cut"):
        directory = root / "constgold" / cohort
        directory.mkdir(parents=True)
        for arm, scale in (("baseline", 0.98), ("shared", 0.99), ("independent", 1.0)):
            statistics = {}
            for branch in BRANCHES:
                statistics[branch] = {}
                for source, factor in (("measured", 1.0), ("model", scale)):
                    statistics[branch][source] = {}
                    for direction, sign in (("plus", 1), ("minus", -1)):
                        first = denominator * response * 0.02 * sign * factor
                        statistics[branch][source][direction] = {
                            "denominator": denominator.tolist(),
                            "numerator": np.stack((first, first * 0), axis=-1).tolist(),
                        }
            result = {
                "cases": [40, 41, 42, 43], "h": 0.02, "draws": 64,
                "sampling_seed": 7301, "bootstrap": {}, "population": {},
                "anchor_rows_by_case": {}, "binning": {},
                "inputs": {"anchor_pattern": "fixed", "rblend": [], "classifiers": []},
                "overall": {branch: {"m_percent": 100 * (1 / scale - 1)} for branch in BRANCHES},
                "per_case_sufficient": {PROPERTY_ORDER[0]: statistics},
            }
            (directory / f"{arm}.json").write_text(json.dumps(result))


def test_paired_case_summary_preserves_exact_multiplicative_difference(tmp_path, monkeypatch):
    make_products(tmp_path)
    monkeypatch.setattr(sys, "argv", ["summary", "--root", str(tmp_path)])
    main()
    result = json.loads((tmp_path / "comparison.json").read_text())
    delta = result["cohorts"]["uncut"][BRANCHES[0]]["paired_comparisons"]["independent_minus_shared"]
    assert delta["delta_m_percentage_points"] == pytest.approx(-100 * (1 / 0.99 - 1))
    assert delta["paired_case_bootstrap_se_pp"] < 1e-12


def test_summary_rejects_changed_sampling_seed(tmp_path, monkeypatch):
    make_products(tmp_path)
    path = tmp_path / "constgold" / "uncut" / "independent.json"
    result = json.loads(path.read_text())
    result["sampling_seed"] += 1
    path.write_text(json.dumps(result))
    monkeypatch.setattr(sys, "argv", ["summary", "--root", str(tmp_path)])
    with pytest.raises(ValueError, match="sampling_seed"):
        main()
