import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sbsi.selection_normalization import (
    ExactPopulationNormalization,
    load_population_normalization,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "combine_selection_normalization_shards.py"
SPEC = importlib.util.spec_from_file_location("combine_selection_normalization_shards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BUILD_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_selection_normalization_cache.py"
BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_selection_normalization_cache", BUILD_SCRIPT
)
BUILD_MODULE = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILD_MODULE)


def _write_shard(root, *, index, start, stop, values):
    root.mkdir()
    payload = {
        "version": 1,
        "method": "exact_detected_selected_mass_atom_shard",
        "center": [0.01, -0.02],
        "finite_difference_step": 0.001,
        "identity": {"scene": "test", "cut_key": "cut"},
        "partition": {
            "index": index,
            "count": 2,
            "active_start": start,
            "active_stop": stop,
            "n_active_total": 5,
            "n_atoms_total": 7,
            "row_chunk": 2,
        },
        "points": [
            {
                "name": name,
                "g1": g1,
                "g2": g2,
                "partial_detected_and_selected_mass": mass,
            }
            for name, g1, g2, mass in values
        ],
    }
    (root / "selection_normalization_shard.json").write_text(
        json.dumps(payload) + "\n"
    )


def test_combine_exact_normalization_shards(tmp_path):
    keys0 = [("zero", 0.01, -0.02, 0.2), ("g1_plus", 0.011, -0.02, 0.21)]
    keys1 = [("zero", 0.01, -0.02, 0.3), ("g1_plus", 0.011, -0.02, 0.31)]
    first = tmp_path / "p0"
    second = tmp_path / "p1"
    _write_shard(first, index=0, start=0, stop=2, values=keys0)
    _write_shard(second, index=1, start=2, stop=5, values=keys1)
    combined = MODULE.combine([second, first])
    assert combined.log_mass(0.01, -0.02) == pytest.approx(math.log(0.5))
    assert combined.log_mass(0.011, -0.02) == pytest.approx(math.log(0.52))
    assert combined.source["n_shards"] == 2


def test_combine_refuses_gap(tmp_path):
    keys = [("zero", 0.01, -0.02, 0.2)]
    first = tmp_path / "p0"
    second = tmp_path / "p1"
    _write_shard(first, index=0, start=0, stop=2, values=keys)
    _write_shard(second, index=1, start=3, stop=5, values=keys)
    with pytest.raises(ValueError, match="not contiguous"):
        MODULE.combine([first, second])


def test_quadratic_builder_reads_distributed_exact_cache(tmp_path):
    center = (0.01, -0.02)
    h = 0.001

    def log_mass(g1, g2):
        x, y = g1 - center[0], g2 - center[1]
        return -0.4 + 0.3 * x - 0.2 * y + x * x + 0.5 * x * y - 0.5 * y * y

    shears = [
        center,
        (center[0] + h, center[1]),
        (center[0] - h, center[1]),
        (center[0], center[1] + h),
        (center[0], center[1] - h),
        (center[0] + h, center[1] + h),
        (center[0] + h, center[1] - h),
        (center[0] - h, center[1] + h),
        (center[0] - h, center[1] - h),
    ]
    exact = ExactPopulationNormalization(
        points={point: math.exp(log_mass(*point)) for point in shears},
        finite_difference_step=h,
        identity={"scene": "test"},
        source={"n_shards": 2},
    )
    cache = tmp_path / "exact.json"
    exact.save(cache)
    result = {
        "initial_center": {"center": list(center)},
        "config": {"h": h},
        "selection": {
            "probabilities": [],
            "normalization_cache": {
                "method": "exact_distributed_detected_selected_mass",
                "path": str(cache),
                "sha256": BUILD_MODULE._sha256(cache),
            },
        },
    }
    _, _, zero, gradient, hessian = BUILD_MODULE._derivatives(result)
    assert zero == pytest.approx(-0.4)
    assert gradient == pytest.approx([0.3, -0.2])
    np.testing.assert_allclose(hessian, [[2.0, 0.5], [0.5, -1.0]], atol=1e-9)


def test_quadratic_builder_jointly_fits_exact_stencils(tmp_path):
    h = 0.001
    fit_center = np.array([0.02, -0.01])

    def log_mass(g1, g2):
        delta = np.asarray([g1, g2]) - fit_center
        gradient = np.asarray([0.3, -0.2])
        hessian = np.asarray([[2.0, 0.5], [0.5, -1.0]])
        return -0.4 + gradient @ delta + 0.5 * delta @ hessian @ delta

    paths = []
    for index, center in enumerate(([0.01, -0.02], [0.03, 0.0])):
        points = {
            (center[0] + dx, center[1] + dy): math.exp(
                log_mass(center[0] + dx, center[1] + dy)
            )
            for dx in (-h, 0.0, h)
            for dy in (-h, 0.0, h)
        }
        exact = ExactPopulationNormalization(
            points=points,
            finite_difference_step=h,
            identity={"scene": "test", "cut_key": "cut"},
            source={"n_shards": 2},
        )
        path = tmp_path / f"exact{index}.json"
        exact.save(path)
        paths.append(path)

    payload = BUILD_MODULE._joint_fit_exact_caches(
        paths,
        reference_center=fit_center,
        max_log_error=1.0e-10,
        max_gradient_error=1.0e-8,
        max_hessian_error=1.0e-6,
    )
    output = tmp_path / "quadratic.json"
    output.write_text(json.dumps(payload) + "\n")
    fitted = load_population_normalization(output)
    assert fitted.center == pytest.approx(fit_center)
    assert fitted.gradient == pytest.approx([0.3, -0.2])
    np.testing.assert_allclose(fitted.hessian, [[2.0, 0.5], [0.5, -1.0]], atol=1e-8)
    assert fitted.log_mass(0.02, -0.01) == pytest.approx(-0.4)
    assert payload["validation"]["n_exact_points"] == 18
