"""The anchor-truncation diagnostic must apply the domain's own rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.fixed_g0_domain import (
    FLUX_RADIUS_MIN_PIXELS,
    MAG_AUTO_MAX,
    fixed_g0_anchor_mask,
)
from scripts.diagnose_flow_anchor_truncation import (
    anchor_flux_minimum,
    masked_shape_mean,
    passes_anchor,
    summarize,
)


def draws_from(radius, flux) -> torch.Tensor:
    """One object, one draw per entry, with the two shape columns unused."""

    rows = len(radius)
    stacked = torch.zeros(1, rows, 4, dtype=torch.float64)
    stacked[0, :, 2] = torch.tensor(radius, dtype=torch.float64)
    stacked[0, :, 3] = torch.tensor(flux, dtype=torch.float64)
    return stacked


def test_the_sampled_anchor_matches_the_catalogue_anchor():
    radius = [2.9, 3.0, 3.1, 10.0, 10.0]
    flux = [1.0e4, 1.0e4, 1.0e4, anchor_flux_minimum() * 0.99, anchor_flux_minimum() * 1.01]
    keep = passes_anchor(draws_from(radius, flux))[0].numpy()
    #  The same five rows, expressed the way the catalogue expresses them.
    frame = pd.DataFrame(
        {
            "measured_flux_radius": radius,
            "measured_mag_auto": 30.0 - 2.5 * np.log10(flux),
        }
    )
    assert list(keep) == list(fixed_g0_anchor_mask(frame))
    assert list(keep) == [False, False, True, False, True]


def test_the_flux_threshold_is_the_magnitude_limit():
    assert 30.0 - 2.5 * np.log10(anchor_flux_minimum()) == pytest.approx(MAG_AUTO_MAX)
    assert FLUX_RADIUS_MIN_PIXELS == 3.0


def test_the_conditional_mean_uses_only_the_kept_draws():
    stacked = torch.zeros(1, 4, 4, dtype=torch.float64)
    stacked[0, :, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    keep = torch.tensor([[False, False, True, True]])
    assert masked_shape_mean(stacked, None)[0, 0] == pytest.approx(0.5)
    assert masked_shape_mean(stacked, keep)[0, 0] == pytest.approx(1.0)


def test_an_object_with_no_kept_draw_is_left_non_finite():
    stacked = torch.ones(1, 2, 4, dtype=torch.float64)
    keep = torch.tensor([[False, False]])
    assert not np.isfinite(masked_shape_mean(stacked, keep)).any()


def test_summarize_recovers_a_planted_bias_per_variant():
    rows = []
    for case in (0, 1, 2):
        for _ in range(50):
            rows.append(
                {
                    "case": case,
                    "R_self_measured": 0.51,
                    "R_self_unconditional": 0.50,
                    "R_self_zero_leg_anchored": 0.51,
                    "R_self_both_legs_anchored": 0.60,
                }
            )
    summary = {row["variant"]: row for row in summarize(pd.DataFrame(rows), replicates=32, seed=1)}
    assert summary["unconditional"]["m_percent"] == pytest.approx(2.0)
    assert summary["zero_leg_anchored"]["m_percent"] == pytest.approx(0.0)
    assert summary["both_legs_anchored"]["m_percent"] == pytest.approx(-15.0)
    assert summary["unconditional"]["objects"] == 150
