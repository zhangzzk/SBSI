import numpy as np
import pandas as pd

from _script_loader import load_script_module


GALLERY = load_script_module("plot_flow_likelihood_gallery.py")


def test_gallery_selection_has_twelve_rows_per_magnitude_stratum():
    rows = []
    for case in (10, 20, 30, 40):
        for offset, magnitude in enumerate(np.linspace(20.0, 21.9, 7)):
            rows.append((case, case * 1000 + offset, magnitude))
        for offset, magnitude in enumerate(np.linspace(23.5, 24.29, 7), start=10):
            rows.append((case, case * 1000 + offset, magnitude))
        for offset, magnitude in enumerate(np.linspace(25.0, 25.79, 7), start=20):
            rows.append((case, case * 1000 + offset, magnitude))
    frame = pd.DataFrame(rows, columns=["case", "input_index", "measured_mag_auto"])
    selected = GALLERY.select_gallery_rows(frame, [10, 20, 30, 40])
    assert len(selected) == 36
    assert selected[["case", "input_index"]].duplicated().sum() == 0
    assert selected.groupby("stratum").size().to_dict() == {
        "bright": 12,
        "faint": 12,
        "typical": 12,
    }


def test_density_thresholds_are_strictly_ordered_when_supported():
    counts = np.arange(1, 101, dtype=float).reshape(10, 10)
    levels = GALLERY._density_thresholds(counts)
    assert levels == sorted(set(levels))
    assert all(level > 0 for level in levels)
