import numpy as np
import pandas as pd
import pytest

from sbsi.output_conditioned_response import predict_scene_draw_responses
from sbsi.shear_map import apply_shear_to_ellipticity
from scripts.joint_response_axes import shear_frames
from scripts.matched_model_probe import SourceShearSigns, signed_scene_draw_responses, signed_shear_frames


class CellResponse:
    """An exactly piecewise-constant two-output-cell surrogate for pooling."""
    response_coordinate = 'disk_velocity'
    conditioned = True
    output_thresholds = (np.array([2.], dtype=np.float32), np.array([100.], dtype=np.float32))

    def transformed_outputs(self, radius, flux, **kwargs):
        return np.column_stack((radius, flux))

    def predict_pairs(self, pairs, radius, flux, **kwargs):
        return pairs['b'].to_numpy()*(1+(radius >= 2)+2*(flux >= 100))


def fixture():
    # Primary 30 is isolated; source 91 is outside the truth-parent set.
    pairs = pd.DataFrame({'index_input_p': [10, 10, 20],
                          'index_input_s': [20, 91, 10], 'b': [.2, .5, -.3]})
    draws = np.zeros((3, 4, 4))
    draws[..., 2] = [1., 1., 3., 3.]
    draws[..., 3] = [50., 50., 200., 200.]
    return pairs, draws


def test_signed_pooling_matches_explicit_neighbour_sum_and_uses_all_sources():
    pairs, draws = fixture()
    signs = SourceShearSigns([91, 30, 10, 20], [-1, 1, -1, 1])
    actual, counts = signed_scene_draw_responses(CellResponse(), pairs, [10, 20, 30], draws, signs,
                                                pair_batch_size=1)
    # First primary: .2*(+1) + .5*(-1); second: -.3*(-1).
    np.testing.assert_allclose(actual, [[-.3, -.3, -1.2, -1.2], [.3, .3, 1.2, 1.2], [0, 0, 0, 0]])
    assert counts['predicted_pair_cells'] == 6 < counts['unpooled_pair_draws']
    reordered, _ = signed_scene_draw_responses(CellResponse(), pairs.iloc[::-1], [10, 20, 30], draws, signs)
    np.testing.assert_allclose(reordered, actual)


def test_all_positive_signs_exactly_replay_existing_coherent_model():
    pairs, draws = fixture()
    signs = SourceShearSigns([30, 20, 91, 10], [1, 1, 1, 1])
    actual, counts = signed_scene_draw_responses(CellResponse(), pairs, [10, 20, 30], draws, signs)
    expected, expected_counts = predict_scene_draw_responses(CellResponse(), pairs, [10, 20, 30], draws)
    np.testing.assert_array_equal(actual, expected)
    assert counts == expected_counts
    context = np.array([[.1, .2, 1, 23, 2, 0, 0, 0], [-.2, .1, 2, 24, 1, 0, 0, 0]])
    for axis in (1, 2):
        actual = signed_shear_frames(context, [10, 20], signs, h=.02, axis=axis)
        expected = shear_frames(context, .02, axis=axis)
        for a, b in zip(actual, expected):
            pd.testing.assert_frame_equal(a, b)


def test_signed_context_uses_primary_identity_and_preserves_other_features():
    signs = SourceShearSigns([91, 20, 10], [1, -1, 1])
    context = np.array([[.1, .2, 1, 23, 2, 0, 0, 0], [-.2, .1, 2, 24, 1, 0, 0, 0]])
    before = context.copy()
    for axis in (1, 2):
        frames = signed_shear_frames(context, [20, 10], signs, h=.02, axis=axis)
        for orientation, frame in zip((1, -1), frames):
            for row, sign in enumerate((-1, 1)):
                g = np.zeros(2)
                g[axis-1] = orientation*sign*.02
                expected = apply_shear_to_ellipticity(*context[row, :2], *g)
                np.testing.assert_array_equal(frame.iloc[row, :2].to_numpy(), expected)
            np.testing.assert_array_equal(frame.iloc[:, 2:], context[:, 2:])
    np.testing.assert_array_equal(context, before)


def test_identity_errors_reject_missing_sources_and_self_pairs():
    pairs, draws = fixture()
    with pytest.raises(ValueError, match='absent'):
        signed_scene_draw_responses(CellResponse(), pairs, [10, 20, 30], draws,
                                     SourceShearSigns([10, 20, 30], [1, -1, 1]))
    signs = SourceShearSigns([10, 20, 30, 91], [1, -1, 1, -1])
    pairs.loc[0, 'index_input_s'] = 10
    with pytest.raises(ValueError, match='self pairs'):
        signed_scene_draw_responses(CellResponse(), pairs, [10, 20, 30], draws, signs)
    with pytest.raises(ValueError, match='unique'):
        SourceShearSigns([10, 10], [1, -1])
    with pytest.raises(ValueError, match='Rademacher'):
        SourceShearSigns([10, 20], [1, 0])
    with pytest.raises(ValueError, match='integer'):
        signs.at([10.5])


def test_sign_table_is_independent_of_input_mutation():
    ids, signs = np.array([10, 20]), np.array([1, -1])
    field = SourceShearSigns(ids, signs)
    ids[0], signs[0] = 99, -1
    np.testing.assert_array_equal(field.at([20, 10, 20]), [-1, 1, -1])
