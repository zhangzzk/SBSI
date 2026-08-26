"""Archived smoke test for the legacy catalogue-prior loop.

WHY THIS EXISTS, AND WHY THE AST GUARD NEXT DOOR IS NOT ENOUGH.  A patch bound `pre` inside
the draw loop, shadowing `pre = est.bundle.condition_preprocessor` which the same loop reads
on every draw.  Three GPU jobs died 21 s in; the whole suite stayed green, because nothing
in `tests/` ran this function against a real bundle.  The static guard added afterwards
protects ONE name against ONE failure mode -- the next shadowing picks a different name.

This runs the actual loop instead, so it fails on ANY AttributeError, shape error or name
error in it.  No flow, no GPU, no fixtures: the stubs expose exactly the attributes the loop
touches, which is also a readable inventory of what the loop depends on.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.catalogue_prior import marginal_score_pass

G, TARGET_DIM, N_FEAT, DIM = 5, 2, 4, 8


class _Flow:
    def log_prob(self, x, ctx):
        return -(x ** 2).sum(-1) - 0.01 * ctx.sum(-1)


class _Model:
    def __init__(self):
        self.flow = _Flow()

    def _mu(self, flat):
        return 0.01 * flat[:, :TARGET_DIM]

    def _flow_ctx(self, flat):
        return flat


class _Bundle:
    def __init__(self, add_missing):
        self.model = _Model()
        self.target_transform = type("T", (), {
            "transform_array": staticmethod(lambda a: np.asarray(a, dtype=np.float32))})()
        self.condition_preprocessor = type("P", (), {
            "add_missing_indicators": add_missing})()


class _Est:
    def __init__(self, add_missing=True):
        self.bundle = _Bundle(add_missing)
        self.device = torch.device("cpu")
        self.G, self.target_dim, self.n_feat = G, TARGET_DIM, N_FEAT

    def _grid_tiled_context(self, frame):
        b = len(frame)
        return torch.arange(b * G * DIM, dtype=torch.float32).view(b, G, DIM) * 1e-3


def _nodes():
    rng = np.random.default_rng(0)
    ns = type("N", (), {})()
    ns.u = rng.normal(size=(G, 2))
    ns.du = rng.normal(size=(G, 2, 2))
    ns.support = np.ones(G, dtype=bool)
    ns.log_prior = rng.normal(size=G)
    ns.info_delta = 0.0025
    ns.shifted = {(ax, sg): (rng.normal(size=G), rng.normal(size=G, ).repeat(2).reshape(G, 2))
                  for ax in ("g1", "g2") for sg in (+1, -1)}
    return ns


@pytest.mark.parametrize("add_missing", [True, False])
@pytest.mark.parametrize("ladder", [[1], [1, 2], [1, 2, 4]])
def test_the_draw_loop_executes_end_to_end(add_missing, ladder):
    n, m_max = 3, max(ladder)
    frame = pd.DataFrame({"a": np.arange(n, dtype=float)})
    ehat = np.zeros((n, TARGET_DIM), dtype=np.float32)
    pool = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    idx = np.zeros((n, m_max), dtype=np.int64)
    logw = np.zeros((n, m_max), dtype=np.float32)
    out, ess, ess_rung = marginal_score_pass(
        _Est(add_missing), _nodes(), frame, ehat, pool, [0, 1], idx, logw, ladder,
        chunk=2, progress_every=0)
    assert set(out) == {"pin", *ladder}
    for k in out:
        assert out[k][0].shape == (n, 2) and out[k][1].shape == (n, 2, 2)
    assert ess.shape == (n,)
    assert ess_rung.shape == (n, len(ladder)), "per-rung ESS must have one column per rung"
    assert np.isfinite(ess_rung).all()


def test_identical_atoms_give_ESS_equal_to_M_not_one():
    """A value check, not just a smoke check -- and it caught me getting it backwards.

    I first asserted ESS = 1 here, reasoning that repeated draws carry no new information.
    That confuses ESS with the number of DISTINCT scenes.  ESS measures how evenly the
    weight is spread: M identical contributions are perfectly uniform, so ESS = M exactly.
    ESS = 1 is the opposite case, one atom carrying everything.  The production numbers
    read the same way -- ESS/M of 61% is about the SPREAD of the evidence contributions,
    which is why it is well below M even at beta=0 where the proposal weights are uniform.
    """
    n, ladder = 3, [1, 2, 4]
    frame = pd.DataFrame({"a": np.arange(n, dtype=float)})
    _, _, ess_rung = marginal_score_pass(
        _Est(), _nodes(), frame, np.zeros((n, TARGET_DIM), dtype=np.float32),
        np.arange(12, dtype=np.float32).reshape(6, 2), [0, 1],
        np.zeros((n, 4), dtype=np.int64), np.zeros((n, 4), dtype=np.float32),
        ladder, chunk=2, progress_every=0)
    for col, M in enumerate(ladder):
        assert np.allclose(ess_rung[:, col], float(M), rtol=1e-5), (
            f"identical atoms are perfectly uniform weights, so rung M={M} must give "
            f"ESS = {M}, got {ess_rung[:, col]}")
