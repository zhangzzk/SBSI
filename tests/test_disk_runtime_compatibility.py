import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("disk_driver", Path(__file__).resolve().parents[1] / "scripts/run_disk_inference.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_run_requires_the_exact_preparation_identity():
    current = {"models": {"flow": "pinned"}, "implementation_sha256": driver.implementation()}
    assert driver.check_prepared_identity(copy.deepcopy(current), current) == "identical"
    changed = copy.deepcopy(current)
    changed["implementation_sha256"]["sbsi/catalogue_disk_likelihood.py"] = "tampered"
    with pytest.raises(ValueError, match="sbsi/catalogue_disk_likelihood.py"):
        driver.check_prepared_identity(current, changed)
    changed = copy.deepcopy(current)
    changed["models"]["flow"] = "another"
    with pytest.raises(ValueError, match="keys=\\['models'\\]"):
        driver.check_prepared_identity(current, changed)
