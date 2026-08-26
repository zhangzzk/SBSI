import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_fs2_prior_generation.py"
)
SPEC = importlib.util.spec_from_file_location("audit_fs2_prior_generation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_case(path):
    pd.DataFrame(
        {
            "index": [0, 1],
            "g1": [0.0, 0.0],
            "g2": [0.0, 0.0],
            "shear_component_convention": ["sky_cos_sin", "sky_cos_sin"],
        }
    ).to_feather(path)


def test_audit_records_disjoint_case_seed_mapping(tmp_path):
    _write_case(tmp_path / "gals20000_0.0.feather")
    _write_case(tmp_path / "gals20001_0.0.feather")
    reports = MODULE.audit_cases(
        tmp_path,
        cases=(20000, 20001),
        expected_rows_per_case=2,
        seed_offset=123,
        forbidden_cases=range(1000),
    )
    assert [report["generator_seed"] for report in reports] == [20123, 20124]
    assert all(report["n_rows"] == 2 for report in reports)


def test_audit_rejects_forbidden_case_overlap(tmp_path):
    with pytest.raises(ValueError, match="overlap forbidden"):
        MODULE.audit_cases(
            tmp_path,
            cases=(5,),
            expected_rows_per_case=2,
            seed_offset=123,
            forbidden_cases=(5,),
        )
