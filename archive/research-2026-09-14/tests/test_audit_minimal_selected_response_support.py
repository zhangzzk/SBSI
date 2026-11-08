import hashlib
import json
from types import SimpleNamespace

import numpy as np

from scripts import audit_minimal_selected_response_support as support


def test_direct_support_uses_probability_mass_and_physical_disk_bound():
    samples = np.array(
        [
            [[0.1, 0.2, 4.0, 100.0], [0.2, 0.1, 2.0, 100.0]],
            [[0.3, 0.2, 2.0, 100.0], [0.2, 0.3, 2.0, 100.0]],
        ]
    )
    result = support.direct_support_from_physical_samples(
        samples,
        event_probability=np.array([0.8, 0.2]),
        blend_shift=np.array([[0.1, -0.05], [0.2, 0.3]]),
    )
    assert result["zero_pass_row_count"] == 1
    assert result["zero_pass_event_probability_fraction"] == 0.2
    np.testing.assert_allclose(result["missing_numerator_abs_bound"], [0.24, 0.26])
    np.testing.assert_allclose(result["leg_mean_abs_bound"], [0.24, 0.26])


def test_support_review_selects_the_smallest_passing_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "TUNE_CASES", (100, 101))
    monkeypatch.setattr(support, "BUDGETS", (64, 256))
    monkeypatch.setattr(support, "QMC_SEEDS", (1, 2))
    monkeypatch.setattr(support, "PLANNED_RESPONSE_H", 0.05)
    monkeypatch.setattr(support, "MAX_RESPONSE_SUPPORT_BOUND", 5.0e-5)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        support,
        "load_protocol",
        lambda output_root: {"paths": {"output_root": str(tmp_path)}},
    )

    for case in support.TUNE_CASES:
        records = {}
        for seed in support.QMC_SEEDS:
            records[str(seed)] = {}
            for leg in range(2):
                records[str(seed)][f"leg{leg}"] = {
                    "64": {
                        "total_event_probability_mass": 10.0,
                        "zero_pass_event_probability_mass": 0.002,
                        "missing_numerator_abs_bound": [0.002, 0.002],
                    },
                    "256": {
                        "total_event_probability_mass": 10.0,
                        "zero_pass_event_probability_mass": 0.00001,
                        "missing_numerator_abs_bound": [0.00001, 0.00001],
                    },
                }
        (tmp_path / "support").mkdir(exist_ok=True)
        (tmp_path / "support" / f"case{case:03d}.json").write_text(
            json.dumps(
                {
                    "checks_passed": True,
                    "case": case,
                    "response_statistics_computed": False,
                    "response_cases_read": False,
                    "protocol_sha256": protocol_hash,
                    "records": records,
                }
            )
        )

    support.review(SimpleNamespace(output_root=tmp_path))
    decision = json.loads((tmp_path / "support_decision.json").read_text())
    assert decision["summaries"]["64"]["passes_support_gate"] is False
    assert decision["summaries"]["256"]["passes_support_gate"] is True
    assert decision["selected_draws_per_row"] == 256
    assert decision["response_statistics_computed"] is False
