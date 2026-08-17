"""Guards on the ``R_blend`` lookup join.

Each test here pins a failure mode that AGENTS.md records as having already cost real time.
They are cheap, data-free and pure-python, so there is no excuse for the conventions to be
enforced only by an agent remembering to read the docs.
"""

import numpy as np
import pandas as pd
import pytest

from sbsi.blend_lookup import join_blend


def _ref(n=10):
    return pd.DataFrame({"case": np.zeros(n, dtype=int),
                         "input_index": np.arange(n),
                         "r_sim": np.linspace(0.5, 1.5, n)})


def _lookup(idx, value=0.13):
    return pd.DataFrame({"case": np.zeros(len(idx), dtype=int),
                         "input_index": np.asarray(idx),
                         "R_blend": np.full(len(idx), value)})


def test_row_order_is_preserved_so_positional_alignment_holds():
    # Callers mask a per-seed R_flow matrix with `keep`, so the returned arrays must stay
    # positionally aligned with `ref` even when the lookup is in a different order.
    ref = _ref(6)
    shuffled = _lookup([5, 0, 3, 1, 4, 2])
    shuffled["R_blend"] = [0.5, 0.0, 0.3, 0.1, 0.4, 0.2]
    values, keep, frac, _ = join_blend(ref, shuffled, "R_blend", verbose=False)
    assert keep.all() and frac == 1.0
    np.testing.assert_allclose(values, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_unmatched_rows_are_nan_and_flagged_never_zero_filled():
    # Zero-filling unmatched rows is the recorded +28.9% m failure (job 15366950). An
    # unmatched row must be NaN and False in `keep` -- never a silent 0.0.
    ref = _ref(10)
    values, keep, frac, _ = join_blend(ref, _lookup([0, 1, 2, 3]), "R_blend", verbose=False)
    assert frac == pytest.approx(0.4)
    assert keep.sum() == 4
    assert np.isnan(values[4:]).all(), "unmatched rows must be NaN, not 0.0"
    assert not (values[~keep] == 0.0).any()


def test_duplicate_keys_refuse_instead_of_multiplying_rows():
    # A repeated (case, input_index) turns a left merge into a row multiplication that
    # silently reweights every downstream mean.
    dup = pd.concat([_lookup([0, 1, 2]), _lookup([1], value=0.99)], ignore_index=True)
    with pytest.raises(SystemExit, match="duplicate"):
        join_blend(_ref(3), dup, "R_blend", verbose=False)


def test_coverage_floor_refuses_when_given_and_reports_when_not():
    ref, lk = _ref(10), _lookup([0, 1])
    with pytest.raises(SystemExit, match="covers only"):
        join_blend(ref, lk, "R_blend", min_match=0.5, verbose=False)
    # min_match=None is the diagnostic mode: report coverage, do not enforce a floor.
    _, _, frac, _ = join_blend(ref, lk, "R_blend", min_match=None, verbose=False)
    assert frac == pytest.approx(0.2)


def test_missing_column_refuses():
    with pytest.raises(SystemExit, match="missing"):
        join_blend(_ref(3), _lookup([0, 1, 2]), "R_blend_flow", verbose=False)


def test_matches_the_plain_left_merge_it_replaced():
    # Equivalence with the inline merge these call sites used before the guards were shared:
    # the guards may only turn a silent wrong answer into a refusal, never change a value.
    ref, lk = _ref(10), _lookup([0, 2, 4, 6, 8])
    expected = ref[["case", "input_index"]].merge(lk, on=["case", "input_index"], how="left")
    values, keep, _, _ = join_blend(ref, lk, "R_blend", verbose=False)
    np.testing.assert_array_equal(np.isnan(values), expected["R_blend"].isna().to_numpy())
    np.testing.assert_allclose(values[keep], expected["R_blend"].to_numpy()[keep])
