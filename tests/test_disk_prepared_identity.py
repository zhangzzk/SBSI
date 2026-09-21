"""The V3.6 disk driver's cache-reuse gate.

The gate decides whether a prepared cache may be reused by the tree that is
running now.  It is the only thing standing between a changed implementation
and a silently mismatched 24m-atom cache, and until this file nothing covered
it.
"""

import ast
import hashlib

import pytest

from _script_loader import load_script_module

DRIVER = "scripts/run_disk_inference.py"
DENSITY = "sbsi/catalogue_disk_likelihood.py"
AUDITED_OLD_DRIVER = "9a32b73a9ae05582cc455037657e1b19ffb8edb829da780208b18fa3f8a55cb4"
AUDITED_OLD_DENSITY = "bfb4e2ba2c8f4be44f844d1001555265b570183f34974c5294b9abc84087dea2"
AUDITED_NEW_DENSITY = "7b6b059d4af4e1a99ca54b60970c55033657290eedf29fd6b193cffddf50784c"


@pytest.fixture(scope="module")
def driver():
    return load_script_module("run_disk_inference.py")


def identities(driver, **changes):
    """A saved/current identity pair differing only in the named files."""
    code = {DRIVER: AUDITED_OLD_DRIVER, DENSITY: AUDITED_OLD_DENSITY,
            "sbsi/catalogue_null.py": "a" * 64, "sbsi/catalogue_sampling.py": "b" * 64,
            "sbsi/disk_inference_store.py": "c" * 64, "sbsi/models.py": "d" * 64}
    saved = dict(center=[0.0, 0.0], h=0.001, implementation_sha256=dict(code))
    current = dict(saved, implementation_sha256=dict(code, **changes))
    return saved, current


def test_an_unchanged_tree_is_identical(driver):
    saved, _ = identities(driver)
    assert driver.check_prepared_identity(saved, dict(saved)) == "identical"


def test_a_changed_scientific_input_is_refused(driver):
    saved, current = identities(driver)
    current = dict(current, h=0.002)
    with pytest.raises(ValueError, match="scientific identity differs"):
        driver.check_prepared_identity(saved, current)


def test_a_preparation_stage_change_is_refused(driver):
    """`disk_inference_store` builds the cached artifacts, so it may not move."""
    saved, current = identities(driver, **{
        DRIVER: "e" * 64, DENSITY: AUDITED_NEW_DENSITY,
        "sbsi/disk_inference_store.py": "f" * 64})
    with pytest.raises(ValueError, match="unapproved preparation/runtime"):
        driver.check_prepared_identity(saved, current)


def test_run_stage_changes_are_accepted_and_named(driver):
    """The estimator and the sampler never run before the cache is written."""
    saved, current = identities(driver, **{
        DRIVER: "e" * 64, DENSITY: AUDITED_NEW_DENSITY,
        "sbsi/catalogue_null.py": "1" * 64, "sbsi/catalogue_sampling.py": "2" * 64})
    compatibility = driver.check_prepared_identity(saved, current)
    assert compatibility.startswith("audited_two_entry_response_lru_v1+run_stage:")
    assert "sbsi/catalogue_null.py" in compatibility
    assert "sbsi/catalogue_sampling.py" in compatibility


def test_the_audited_pair_alone_keeps_its_original_compatibility_string(driver):
    saved, current = identities(driver, **{DRIVER: "e" * 64, DENSITY: AUDITED_NEW_DENSITY})
    assert driver.check_prepared_identity(saved, current) == "audited_two_entry_response_lru_v1"


def test_a_run_stage_change_alone_still_requires_the_audited_pair(driver):
    """Run-stage files widen the audited exception; they do not replace it."""
    saved, current = identities(driver, **{"sbsi/catalogue_null.py": "1" * 64})
    with pytest.raises(ValueError, match="unapproved preparation/runtime"):
        driver.check_prepared_identity(saved, current)


def test_the_preparation_functions_are_still_byte_identical(driver):
    """Guards the pin itself: editing these four invalidates every cache."""
    source = (driver.REPO / DRIVER).read_text()
    names = {"identity", "prepare", "assemble", "seed_stream"}
    parts = [ast.get_source_segment(source, node) for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(parts) == len(names)
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    assert digest == "8a4c398b22e7415d0ed19a4df23b53adc221bb43d7dc3bb0705ec325b8bb8ab1"
