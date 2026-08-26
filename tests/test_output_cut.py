"""`OutputCut`: the selection `W(xhat)` shared by the score pass and the population block.

The failure this file exists to prevent is not an exception, it is a SILENT disagreement.
`eval_score_select.py` applies the cut twice on two different array libraries -- once in
NumPy, to decide which galaxies enter (5.3)'s sums, and once in torch, to build `Pi_k` on
the node bank.  If those two predicates differ anywhere (a `<` against a `<=`, a NaN
convention, a `hypot` that rounds differently from `x^2 + y^2`), then `<s>_sel` and `I_sel`
describe a different population from the one being scored, the correction is subtracted
from the wrong sample, and nothing raises.  `test_numpy_and_torch_agree_exactly` is
therefore the headline test here.

The rest guard the two boundaries the class is responsible for: §4.7 (a cut must name a
flow OUTPUT, or `P_pass` does not exist) and score-cache identity (`key()` must stay the
bare float for a plain `|xhat|` cut, or every shard banked before `--cut-bound` existed
becomes unloadable).
"""

import numpy as np
import pytest
import torch

from sbsi.score_inference import OutputCut


V3_OUTPUTS = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto", "measured_log_flux_radius"]
V1_OUTPUTS = ["measured_ngmix_g1", "measured_ngmix_g2"]


def sample_draws(n=4000, seed=3):
    """Draws with the V3 flow's own scale, so the cuts land mid-distribution."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.normal(0.0, 0.364, n),
            rng.normal(0.0, 0.366, n),
            rng.normal(23.89, 1.214, n),
            rng.normal(1.586, 0.232, n),
        ],
        axis=1,
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(abs_shape=0.6),
        dict(abs_shape=None, bounds=[("measured_mag_auto", None, 24.5)]),
        dict(abs_shape=None, bounds=[("measured_log_flux_radius", 1.45, None)]),
        dict(
            abs_shape=0.6, bounds=[("measured_mag_auto", 22.0, 24.5), ("measured_log_flux_radius", 1.45, 2.0)]
        ),
    ],
)
def test_numpy_and_torch_agree_exactly(kw):
    cut = OutputCut(V3_OUTPUTS, **kw)
    x = sample_draws()
    a = cut(x)
    b = cut(torch.as_tensor(x)).numpy()
    assert a.dtype == bool and b.dtype == bool
    assert np.array_equal(a, b)
    # and the cut has to actually bite, or agreement is vacuous
    assert 0.05 < a.mean() < 0.95


def test_agreement_holds_on_the_stacked_shape_the_population_block_uses():
    """The score pass sees `(N, D)`; `pass_fraction_by_node` sees `(M, S, D)`."""
    cut = OutputCut(V3_OUTPUTS, abs_shape=0.6, bounds=[("measured_mag_auto", None, 24.5)])
    x = sample_draws(n=1200).reshape(150, 8, 4)
    keep = cut(torch.as_tensor(x))
    assert keep.shape == (150, 8)
    assert np.array_equal(cut(x), keep.numpy())
    assert np.array_equal(cut(x.reshape(-1, 4)), keep.numpy().reshape(-1))


def test_bounds_are_half_open():
    cut = OutputCut(V3_OUTPUTS, bounds=[("measured_mag_auto", 23.0, 24.0)])
    x = np.zeros((3, 4))
    x[:, 2] = [23.0, 23.5, 24.0]
    assert list(cut(x)) == [True, True, False]


def test_absent_bound_means_unbounded():
    x = np.zeros((2, 4))
    x[:, 2] = [10.0, 30.0]
    assert list(OutputCut(V3_OUTPUTS, bounds=[("measured_mag_auto", None, 24.0)])(x)) == [True, False]
    assert list(OutputCut(V3_OUTPUTS, bounds=[("measured_mag_auto", 24.0, None)])(x)) == [False, True]


def test_a_cut_on_a_non_output_is_refused():
    """§4.7: `P_pass` is a functional of the flow's density, so the cut must be on it."""
    with pytest.raises(KeyError, match="not a flow output"):
        OutputCut(V3_OUTPUTS, bounds=[("measured_flux_radius", 3.0, None)])
    # the V1-shaped flow predicts shape only -- which is exactly why the mag cut needed V3
    with pytest.raises(KeyError, match="not a flow output"):
        OutputCut(V1_OUTPUTS, bounds=[("measured_mag_auto", None, 24.5)])


def test_shape_cut_needs_a_shape_pair_in_the_outputs():
    with pytest.raises(KeyError):
        OutputCut(["measured_mag_auto"], abs_shape=0.6)


def test_a_cut_that_keeps_everything_is_refused():
    with pytest.raises(ValueError):
        OutputCut(V3_OUTPUTS)


def test_empty_and_inverted_bounds_are_refused():
    with pytest.raises(ValueError):
        OutputCut(V3_OUTPUTS, bounds=[("measured_mag_auto", 25.0, 24.0)])
    with pytest.raises(ValueError):
        OutputCut(V3_OUTPUTS, abs_shape=-0.6)


def test_key_stays_a_bare_float_for_a_plain_shape_cut():
    """Score caches banked before --cut-bound existed must still load (cont.177/178)."""
    assert OutputCut(V3_OUTPUTS, abs_shape=0.6).key() == 0.6
    assert OutputCut(V1_OUTPUTS, abs_shape=0.6).key() == 0.6


def test_key_separates_cuts_that_select_different_samples():
    keys = {
        OutputCut(V3_OUTPUTS, abs_shape=0.6).key(),
        OutputCut(V3_OUTPUTS, abs_shape=0.7).key(),
        OutputCut(V3_OUTPUTS, bounds=[("measured_mag_auto", None, 24.5)]).key(),
        OutputCut(V3_OUTPUTS, bounds=[("measured_mag_auto", None, 25.0)]).key(),
        OutputCut(V3_OUTPUTS, abs_shape=0.6, bounds=[("measured_mag_auto", None, 24.5)]).key(),
    }
    assert len(keys) == 5


def test_from_specs_parses_the_command_line_form():
    cut = OutputCut.from_specs(
        V3_OUTPUTS, abs_shape=None, specs=["measured_mag_auto::24.5", "measured_log_flux_radius:1.45:"]
    )
    assert [(n, lo, hi) for n, _, lo, hi in cut.bounds] == [
        ("measured_mag_auto", None, 24.5),
        ("measured_log_flux_radius", 1.45, None),
    ]
    assert OutputCut.from_specs(V3_OUTPUTS, specs=["measured_mag_auto:-inf:inf"]).bounds == [
        ("measured_mag_auto", 2, None, None)
    ]
    with pytest.raises(ValueError, match="NAME:LO:HI"):
        OutputCut.from_specs(V3_OUTPUTS, specs=["measured_mag_auto<24.5"])


def test_describe_reads_as_the_selection_it_applies():
    cut = OutputCut(
        V3_OUTPUTS,
        abs_shape=0.6,
        bounds=[("measured_mag_auto", None, 24.5), ("measured_log_flux_radius", 1.45, 2.0)],
    )
    assert cut.describe() == (
        "|xhat| < 0.6 and measured_mag_auto < 24.5 and 1.45 <= measured_log_flux_radius < 2"
    )
