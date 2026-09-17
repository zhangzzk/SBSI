import numpy as np
import pytest
import torch

from sbsi.flow_guard_response import SHAPE_RESPONSE_COMPONENTS, GuardResponsePopulation
from scripts.refine_band_guard_flow import (
    CUMULATIVE_SCALE_FACTOR,
    OFF_DIAGONAL_SCALE_FACTOR,
    RESPONSE_SCALE,
    build_guard_bank,
    build_response_scale,
)


def test_guard_bank_reports_how_many_leading_cuts_are_cumulative():
    cuts, cumulative_count = build_guard_bank()
    assert 0 < cumulative_count < len(cuts)
    # The cumulative cuts come first and none of them closes a band from above.
    for cut in cuts[:cumulative_count]:
        assert cut.get("radius_max") is None
    # At least one band follows, and bands are what carry an upper radius edge.
    assert any(cut.get("radius_max") is not None for cut in cuts[cumulative_count:])


def test_cumulative_cuts_are_up_weighted_against_the_bands():
    cuts, cumulative_count = build_guard_bank()
    scale = build_response_scale(cuts, cumulative_count)
    assert scale.shape == (len(cuts), len(SHAPE_RESPONSE_COMPONENTS))

    diagonal = [
        index
        for index, name in enumerate(SHAPE_RESPONSE_COMPONENTS)
        if name not in {"R12", "R21"}
    ]
    cumulative = scale[:cumulative_count][:, diagonal]
    bands = scale[cumulative_count:][:, diagonal]
    assert np.allclose(cumulative, RESPONSE_SCALE * CUMULATIVE_SCALE_FACTOR)
    assert np.allclose(bands, RESPONSE_SCALE)
    # The loss divides by the scale and squares, so a smaller scale is a
    # larger share of the gradient.  That is the whole point of the change.
    assert CUMULATIVE_SCALE_FACTOR < 1.0
    ratio = (bands[0, 0] / cumulative[0, 0]) ** 2
    assert ratio == pytest.approx(CUMULATIVE_SCALE_FACTOR**-2)


def test_off_diagonal_de_weighting_still_applies_to_both_kinds():
    cuts, cumulative_count = build_guard_bank()
    scale = build_response_scale(cuts, cumulative_count)
    for index, name in enumerate(SHAPE_RESPONSE_COMPONENTS):
        if name not in {"R12", "R21"}:
            continue
        column = scale[:, index]
        expected = np.where(
            np.arange(len(cuts)) < cumulative_count,
            RESPONSE_SCALE * CUMULATIVE_SCALE_FACTOR * OFF_DIAGONAL_SCALE_FACTOR,
            RESPONSE_SCALE * OFF_DIAGONAL_SCALE_FACTOR,
        )
        assert np.allclose(column, expected)


def test_build_response_scale_rejects_a_count_outside_the_bank():
    cuts, _ = build_guard_bank()
    with pytest.raises(ValueError):
        build_response_scale(cuts, len(cuts) + 1)
    with pytest.raises(ValueError):
        build_response_scale(cuts, -1)


def test_the_bank_and_its_scale_are_accepted_by_the_population():
    cuts, cumulative_count = build_guard_bank()
    scale = build_response_scale(cuts, cumulative_count)
    rng = np.random.default_rng(4212)
    rows = 64
    context = torch.as_tensor(
        np.column_stack(
            (
                rng.normal(scale=0.1, size=rows),
                rng.normal(scale=0.1, size=rows),
                rng.uniform(2.0, 9.0, size=rows),
                10.0 ** ((30.0 - rng.uniform(22.0, 26.0, size=rows)) / 2.5),
            )
        ),
        dtype=torch.float64,
    )
    pairs = np.column_stack((np.arange(0, rows, 2), np.arange(1, rows, 2))).astype(
        np.int64
    )
    gamma = rng.normal(scale=0.02, size=(len(pairs), 2))
    population = GuardResponsePopulation(
        context,
        pairs,
        context[pairs[:, 0]].numpy(),
        context[pairs[:, 1]].numpy(),
        gamma,
        cuts,
        radius_softness=0.005,
        magnitude_softness=0.01,
        response_scale=scale,
    )
    assert tuple(population.response_scale.shape) == (
        len(cuts),
        len(SHAPE_RESPONSE_COMPONENTS),
    )
    assert np.allclose(
        population.response_scale.numpy()[:cumulative_count, 0],
        RESPONSE_SCALE * CUMULATIVE_SCALE_FACTOR,
    )
