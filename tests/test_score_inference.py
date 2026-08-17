"""Unit tests for the §5B score machinery, with an analytic likelihood in place of the
flow.  Everything here runs in seconds on a CPU and needs no checkpoint: the flow only
supplies `log p(ehat | e)`, so a Gaussian standing in for it exercises the prior, the
generator, the posterior weights, the information and the estimator end to end.

The decisive test is `test_estimator_recovers_injected_shear`, which is the whole of §5B
in miniature: shear a population, generate data, and check that summing scores over
information returns the shear -- and that `Cov(ehat, s)` returns the same response that
transporting the samples does.
"""

import numpy as np

try:                                       # pytest lives in py31; the flow env is sims1
    import pytest
except ImportError:                        # minimal shim so `python tests/...` also works
    import functools
    import types

    pytest = types.SimpleNamespace(
        fixture=lambda *a, **k: (lambda fn: functools.lru_cache(maxsize=None)(fn)))

from sbsi.posterior_shape import make_e_grid
from sbsi.score_inference import (
    ShapeScoreNodes,
    SmoothRadialPrior,
    generator_closed_form,
    project,
    response_from_score,
    scores_from_loglike,
    shear_estimate,
)
from sbsi.shear_map import apply_shear_to_ellipticity

SIGMA_E = 0.24          # per-component intrinsic ellipticity scatter, ~ the real one
A_RESP = 0.30           # toy "measurement" response, ~ the certified R_flow
SIGMA_N = 0.20          # toy measurement noise


@pytest.fixture(scope="module")
def prior():
    rng = np.random.default_rng(0)
    e = rng.normal(0.0, SIGMA_E, size=(400_000, 2))
    keep = np.hypot(e[:, 0], e[:, 1]) < 0.95
    return SmoothRadialPrior(e[keep, 0], e[keep, 1], n_bins=80, n_knots=6)


@pytest.fixture(scope="module")
def nodes(prior):
    grid, _ = make_e_grid(n=61, emax=0.96, rmax=0.95)
    return ShapeScoreNodes(grid, prior, delta=0.01, info_delta=0.0025)


def test_prior_is_isotropic_and_normalised(prior):
    rng = np.random.default_rng(1)
    e1, e2 = prior.sample(200_000, rng)
    assert abs(np.mean(e1)) < 5e-3 and abs(np.mean(e2)) < 5e-3
    assert abs(np.std(e1) / np.std(e2) - 1.0) < 0.02
    assert abs(prior.norm_error) < 1e-3


def test_generator_matches_closed_form(nodes, prior):
    """The finite difference of the Mobius pullback must reproduce
    `u_a = e_a [4 - 2 psi'(r^2)(1 - r^2)]`.  These share no code path: one
    differentiates the sheared prior numerically, the other assembles
    `-(v . grad log p0 + div v)` by hand."""
    u_closed = generator_closed_form(prior, nodes.grid)
    rms = np.sqrt(np.mean(nodes.u ** 2))
    assert np.abs(nodes.u - u_closed).max() / rms < 1e-3


def test_bartlett_identities_on_the_node_bank(nodes):
    """`E_0[u] = 0` and `E_0[du] + Var_0(u) = 0`.

    Both follow from `integral p_gamma = 1` alone, so they test the prior, the shear map,
    the divergence term and the grid quadrature simultaneously -- with no data.  The
    second is the sharp one: it is the per-object information of a galaxy whose
    likelihood is flat, which must be zero because such a galaxy says nothing about
    shear.  Any residual is a spurious information floor that would bias `ghat` by
    `residual / <I>`.
    """
    b = nodes.bartlett()
    var_u = b["scale"] ** 2
    assert np.abs(b["mean_u"]).max() < 1e-8 * np.sqrt(var_u)
    assert np.abs(b["curvature"]).max() < 5e-3 * var_u


def _toy_experiment(prior, nodes, gamma, n=60_000, seed=3):
    """A whole §5B experiment with a Gaussian in place of the flow.

    Returns the score-route quantities and the transport response measured on the same
    draws, so the two routes to `d<ehat>/dgamma` can be compared directly.
    """
    rng = np.random.default_rng(seed)
    e1i, e2i = prior.sample(n, rng)
    noise = rng.normal(0.0, SIGMA_N, size=(n, 2))          # common random numbers
    grid = nodes.grid
    out = {}
    for sign in (+1, -1):
        e1, e2 = apply_shear_to_ellipticity(e1i, e2i, sign * gamma, 0.0)
        ehat = A_RESP * np.stack([e1, e2], axis=1) + noise
        # log p(ehat | e_k) for every node, up to a per-row constant
        d1 = ehat[:, None, 0] - A_RESP * grid[None, :, 0]
        d2 = ehat[:, None, 1] - A_RESP * grid[None, :, 1]
        ll = -(d1 ** 2 + d2 ** 2) / (2 * SIGMA_N ** 2)
        s, info, _ = scores_from_loglike(ll.astype(np.float32), nodes, device="cpu")
        out[sign] = dict(s=s, info=info, ehat=ehat, e=np.stack([e1, e2], axis=1))
    # transport: the same derivative, taken by moving the samples
    out["R_transport"] = float(
        np.mean(A_RESP * (out[+1]["e"][:, 0] - out[-1]["e"][:, 0])) / (2 * gamma))
    return out


def test_estimator_recovers_injected_shear(prior, nodes):
    """§5B end to end: `ghat = sum s / sum I` must return the shear that was applied."""
    gamma = 0.02
    r = _toy_experiment(prior, nodes, gamma)
    ones = np.ones(len(r[+1]["s"]))
    sp, ip = [], []
    for sign in (+1, -1):
        a, b = project(r[sign]["s"], r[sign]["info"], ones, 0 * ones)
        sp.append(a)
        ip.append(b)
    s_anti = 0.5 * (sp[0] - sp[1])                        # antithetic: kills shape noise
    i_anti = 0.5 * (ip[0] + ip[1])
    ghat, _ = shear_estimate(s_anti, i_anti)
    err = float(np.std(s_anti / np.mean(i_anti)) / np.sqrt(len(s_anti)))
    assert abs(ghat - gamma) < max(4 * err, 0.02 * gamma), (
        f"ghat={ghat:.5f} vs injected {gamma}, err={err:.5f}")


def test_covariance_identity_returns_the_transport_response(prior, nodes):
    """(2.3) with `f = ehat`: `Cov_0(ehat, s)` is the response, and it must agree with
    the response obtained by transporting the samples -- the two sides of §5's opening
    equation, evaluated on the same draws."""
    gamma = 0.02
    r = _toy_experiment(prior, nodes, gamma)
    rs = [response_from_score(r[sign]["ehat"][:, 0], r[sign]["s"][:, 0])
          for sign in (+1, -1)]
    r_score = 0.5 * (rs[0] + rs[1])
    assert abs(r_score / r["R_transport"] - 1.0) < 0.03, (
        f"Cov(ehat,s)={r_score:.4f} vs transport {r['R_transport']:.4f}")


def test_information_finite_difference_matches_louis(prior, nodes):
    """`I = -d_gamma s_gamma` (used because §5C.3 gives the likelihood its own
    gamma-dependence) must reduce to Louis' `-E_w[du] - Var_w(u)` on the plain model."""
    rng = np.random.default_rng(11)
    e1i, e2i = prior.sample(20_000, rng)
    ehat = A_RESP * np.stack([e1i, e2i], 1) + rng.normal(0, SIGMA_N, size=(20_000, 2))
    d1 = ehat[:, None, 0] - A_RESP * nodes.grid[None, :, 0]
    d2 = ehat[:, None, 1] - A_RESP * nodes.grid[None, :, 1]
    ll = (-(d1 ** 2 + d2 ** 2) / (2 * SIGMA_N ** 2)).astype(np.float32)
    _, i_fd, _ = scores_from_loglike(ll, nodes, device="cpu")
    _, i_an, _ = scores_from_loglike(ll, nodes, device="cpu", analytic_info=True)
    assert abs(i_fd[:, 0, 0].mean() / i_an[:, 0, 0].mean() - 1.0) < 0.01


def test_uninformative_data_carries_no_information(prior, nodes):
    """A galaxy with a flat likelihood must get `s = 0` and `I = 0`: the posterior is the
    prior, so Bartlett applies per object.  This is the per-object form of the check in
    `test_bartlett_identities_on_the_node_bank`, and it is what makes it legitimate to
    sum `I_i` over a catalogue containing badly measured objects."""
    ll = np.zeros((16, len(nodes.grid)), dtype=np.float32)
    s, info, _ = scores_from_loglike(ll, nodes, device="cpu")
    var_u = nodes.bartlett()["scale"] ** 2
    assert np.abs(s).max() < 1e-6 * np.sqrt(var_u)
    assert np.abs(info).max() < 5e-3 * var_u


if __name__ == "__main__":                 # `python tests/test_score_inference.py`
    import inspect
    import sys
    import time

    _get = (lambda f: f.__wrapped__ if hasattr(f, "__wrapped__") else f)
    _p = _get(prior)()
    _n = _get(nodes)(_p)
    _fx = {"prior": _p, "nodes": _n}
    fails = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        _t = time.time()
        try:
            _fn(**{k: _fx[k] for k in inspect.signature(_fn).parameters})
            print(f"PASS {_name:<52} {time.time() - _t:5.1f}s")
        except AssertionError as exc:
            fails += 1
            print(f"FAIL {_name:<52} {exc}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
