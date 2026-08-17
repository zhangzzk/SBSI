import numpy as np
import pandas as pd

from scripts.eval_v21_constgold_isolation import (
    case_bootstrap_percent,
    nearest_other_source,
)


def test_nearest_other_source_excludes_the_target_itself():
    frame = pd.DataFrame({
        "index": [10, 20, 30],
        "RA": np.array([0.0, 1.0, 3.0]) / 3600.0,
        "DEC": [0.0, 0.0, 0.0],
    })
    got = nearest_other_source(frame, np.array([10, 20, 30]), workers=1)
    np.testing.assert_allclose(got, [1.0, 1.0, 2.0], rtol=0, atol=1e-10)


def test_case_bootstrap_preserves_an_exact_response_ratio():
    case = np.repeat([40, 41, 42], [2, 3, 4])
    keep = np.ones(len(case), dtype=bool)
    flow = np.arange(1, len(case) + 1, dtype=float)
    sim = 2.0 * flow
    got = case_bootstrap_percent(case, keep, sim, flow, n_boot=100, seed=7)
    assert got < 1e-12
