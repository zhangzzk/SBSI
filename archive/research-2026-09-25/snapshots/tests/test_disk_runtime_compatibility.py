import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("disk_driver", Path(__file__).resolve().parents[1] / "scripts/run_disk_inference.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_cache_reuse_is_limited_to_exact_audited_inference_only_patch():
    current = {"models": {"flow": "pinned"}, "implementation_sha256": driver.implementation()}
    saved = copy.deepcopy(current)
    saved["implementation_sha256"]["scripts/run_disk_inference.py"] = "9a32b73a9ae05582cc455037657e1b19ffb8edb829da780208b18fa3f8a55cb4"
    saved["implementation_sha256"]["sbsi/catalogue_disk_likelihood.py"] = "bfb4e2ba2c8f4be44f844d1001555265b570183f34974c5294b9abc84087dea2"
    assert driver.check_prepared_identity(saved, current) == "audited_two_entry_response_lru_v1"
    assert driver.check_prepared_identity(current, current) == "identical"
    for key in ("sbsi/measurement_model.py", "sbsi/catalogue_disk_cache.py", "sbsi/catalogue_sampling.py"):
        changed = copy.deepcopy(current)
        changed["implementation_sha256"][key] = "tampered"
        with pytest.raises(ValueError, match="unapproved"):
            driver.check_prepared_identity(saved, changed)
    changed = copy.deepcopy(current)
    changed["models"]["flow"] = "another"
    with pytest.raises(ValueError, match="scientific"):
        driver.check_prepared_identity(saved, changed)
