import numpy as np

from sbsi.measurement_model import TargetStandardizer
from scripts.train_minimal_selected_shape_flow import PHYSICAL_NAMES, _selected_mask, selftest


def test_selected_mask_uses_exact_frozen_half_open_cuts():
    target = TargetStandardizer(PHYSICAL_NAMES, np.zeros(4), np.ones(4), dtype="float64")
    flux_boundary = 10 ** ((30.0 - 25.8) / 2.5)
    values = np.array(
        [
            [0.0, 0.0, 3.75, flux_boundary * (1.0 + 1.0e-10)],
            [0.0, 0.0, np.nextafter(3.75, -np.inf), 100.0],
            [0.0, 0.0, 4.0, flux_boundary],
            [0.0, 0.0, 4.0, np.nan],
        ]
    )
    np.testing.assert_array_equal(_selected_mask(values, target), [True, False, False, False])


def test_selected_shape_selftest():
    selftest()
