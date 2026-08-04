"""Conventions of FLOW #2 (`sbs_shear/blend_flow.py`).

These guard the two places a blend-response error would be SILENT: the shear map applied to the
neighbour's shape, and the finite-difference factor that turns mean-head shifts into `R_blend`. Both
produce plausible numbers when wrong, and neither raises.
"""
import numpy as np
import torch

from sbs_shear.blend_flow import (
    RAW, apply_shear, build_features, build_model, feature_names, response_from_contexts,
    shape_column_indices, shifted_shape_columns)
from sbs_shear.shear_map import apply_shear_to_ellipticity


def _frame(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "r_input_p": rng.uniform(20, 26, n),
        "Re_input_p": rng.uniform(0.31, 1.4, n),
        "sersic_n_input_p": rng.uniform(0.6, 4.0, n),
        "e1_input_rot0_p": rng.uniform(-0.6, 0.6, n),
        "e2_input_rot0_p": rng.uniform(-0.6, 0.6, n),
        "r_input_s": rng.uniform(20, 28, n),
        "Re_input_s": rng.uniform(0.05, 1.4, n),
        "sersic_n_input_s": rng.uniform(0.6, 4.0, n),
        "e1_input_rot0_s": rng.uniform(-0.6, 0.6, n),
        "e2_input_rot0_s": rng.uniform(-0.6, 0.6, n),
        "distance": rng.uniform(0.05, 7.0, n),
    }


def test_apply_shear_matches_canonical_map():
    """The local real-component Mobius map must equal sbs_shear.shear_map's complex one."""
    f = _frame()
    a1, a2 = apply_shear(f["e1_input_rot0_s"], f["e2_input_rot0_s"], 0.03, -0.02)
    b1, b2 = apply_shear_to_ellipticity(f["e1_input_rot0_s"], f["e2_input_rot0_s"], 0.03, -0.02)
    assert np.allclose(a1, b1, atol=1e-12)
    assert np.allclose(a2, b2, atol=1e-12)


def test_zero_shift_is_identity():
    f = _frame()
    sh = shifted_shape_columns(f, 0.0)
    for key in ("e1+", "e1-", "e2+", "e2-"):
        assert np.allclose(sh[key][0], f["e1_input_rot0_s"], atol=1e-12)
        assert np.allclose(sh[key][1], f["e2_input_rot0_s"], atol=1e-12)


def test_shift_touches_only_the_neighbour_shape():
    """Every other feature must be invariant under the shift, including the DERIVED ones.

    If a derived feature depended on the neighbour's shape it would open a second, unintended
    response channel and the measured `R_blend` would not be the quantity the label measures.
    """
    f = _frame()
    sidx = shape_column_indices(derived=True)
    X = build_features(f, derived=True)
    g = dict(f)
    g["e1_input_rot0_s"], g["e2_input_rot0_s"] = apply_shear(
        f["e1_input_rot0_s"], f["e2_input_rot0_s"], 0.05, 0.0)
    Xs = build_features(g, derived=True)
    changed = np.where(~np.isclose(X, Xs, atol=1e-12).all(axis=0))[0]
    assert set(changed.tolist()) <= set(sidx), (
        f"the shift changed columns {changed.tolist()} but only {sidx} are the shape channel")


def test_response_is_zero_when_the_mean_head_ignores_the_shape():
    """Zeroing the mean head's weights on the two shape columns must give exactly R_blend = 0.

    This is the arithmetic version of the `Gold-V3.md` blocker: with only spin-0 conditioning the
    shifted and unshifted contexts are identical and no loss term can teach a response. If this
    ever fails, the response is leaking in through a channel that is not the neighbour's shape.
    """
    f = _frame()
    derived = True
    sidx = shape_column_indices(derived)
    X = build_features(f, derived=derived).astype(np.float32)
    model, _ = build_model(X.shape[1], mean_hidden=0, derived=derived)
    with torch.no_grad():
        model.mean_net.weight[:, sidx] = 0.0
    sh = shifted_shape_columns(f, 0.02)
    c0 = torch.as_tensor(X)
    ctxs = {}
    for k in ("e1+", "e1-", "e2+", "e2-"):
        c = c0.clone()
        c[:, sidx[0]] = torch.as_tensor(sh[k][0].astype(np.float32))
        c[:, sidx[1]] = torch.as_tensor(sh[k][1].astype(np.float32))
        ctxs[k] = c
    with torch.no_grad():
        r = response_from_contexts(model, c0, ctxs, (1.0, 1.0), 0.02)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)


def test_response_matches_analytic_linear_derivative():
    """With a LINEAR mean head the response has a closed form; the code must reproduce it.

    For mu = W c + b on standardized context c, the trace/2 responsivity is

        R = 0.5 * ( W[0, i0] * d(c_i0)/d(g1) * sc0 + W[1, i1] * d(c_i1)/d(g2) * sc1 )

    and the Mobius derivative at g=0 is J = [[1 - a, -b], [-b, 1 + a]] with a = e1^2 - e2^2,
    b = 2 e1 e2. Checking the finite difference against this catches a wrong factor (0.5 vs 0.25) or
    a missing 1/delta -- both of which would leave R_blend looking perfectly reasonable.

    Run in FLOAT64 at the production delta. In float32 with a tiny delta the central difference
    loses most of its digits to cancellation (at delta=1e-4 the difference is ~5e-6 against
    order-one values, i.e. ~1% error) and the test fails on its own arithmetic rather than on the
    code. The production delta of 0.02 is nowhere near that regime -- the difference is ~1e-3 there,
    so float32 costs ~6e-5 relative -- but the test should not depend on that margin.
    """
    f = _frame(n=256, seed=3)
    derived = True
    sidx = shape_column_indices(derived)
    X = build_features(f, derived=derived)
    torch.manual_seed(11)          # the mean head is randomly initialised; keep the test reproducible
    model, _ = build_model(X.shape[1], mean_hidden=0, derived=derived)
    model.double()
    delta, sc0, sc1 = 2e-2, 0.31, 0.29
    sh = shifted_shape_columns(f, delta)
    c0 = torch.as_tensor(X, dtype=torch.float64)
    ctxs = {}
    for k in ("e1+", "e1-", "e2+", "e2-"):
        c = c0.clone()
        c[:, sidx[0]] = torch.as_tensor(sh[k][0], dtype=torch.float64)
        c[:, sidx[1]] = torch.as_tensor(sh[k][1], dtype=torch.float64)
        ctxs[k] = c
    with torch.no_grad():
        r = response_from_contexts(model, c0, ctxs, (sc0, sc1), delta).numpy()
        W = model.mean_net.weight.numpy()

    e1, e2 = f["e1_input_rot0_s"], f["e2_input_rot0_s"]
    # d eps'/d g at g=0 in real components: J = [[1 - a, -b], [-b, 1 + a]], a = e1^2 - e2^2,
    # b = 2 e1 e2 (sbs_shear.shear_map.shear_jacobian_at_zero). The e1-shift moves BOTH shape
    # columns, so both weights enter each term.
    a, b = e1 ** 2 - e2 ** 2, 2 * e1 * e2
    d1_dg1, d2_dg1 = 1 - a, -b
    d1_dg2, d2_dg2 = -b, 1 + a
    expect = 0.5 * ((W[0, sidx[0]] * d1_dg1 + W[0, sidx[1]] * d2_dg1) * sc0
                    + (W[1, sidx[0]] * d1_dg2 + W[1, sidx[1]] * d2_dg2) * sc1)
    # The tolerance must be ABSOLUTE, at the level of the central difference's O(delta^2) truncation.
    # Measured across random initialisations that error is a steady ~1e-5 -- independent of the draw,
    # as a truncation error should be -- while `expect` occasionally lands near zero for some rows,
    # which makes any purely RELATIVE tolerance fail on arithmetic that is in fact correct. 1e-4 sits
    # comfortably above the truncation and far below the response scale (~0.03-0.1).
    assert np.allclose(r, expect, rtol=5e-3, atol=1e-4)


def test_feature_names_align_with_shape_indices():
    for derived in (True, False):
        names = feature_names(derived)
        idx = shape_column_indices(derived)
        assert [names[i] for i in idx] == ["e1_input_rot0_s", "e2_input_rot0_s"]
        assert len(names) == build_features(_frame(8), derived=derived).shape[1]


def test_raw_columns_are_what_the_pairset_writes():
    """Guards the contract between build_blend_pairset.py and the model's feature builder."""
    assert RAW[-1] == "distance"
    f = _frame(4)
    for c in RAW:
        assert c in f, f"{c} missing from the pair-set contract"


# ---------------------------------------------------------------------------------------------
# DUAL MODE (2026-08-02i): the primary's shape becomes a SECOND response channel, so the whole
# silent-failure surface the tests above cover for the neighbour now exists for the primary too --
# with more at stake, because the self response is ~6x the blend response.
# ---------------------------------------------------------------------------------------------

def test_primary_shift_touches_only_the_primary_shape():
    """Mirror of the neighbour test, for the self channel."""
    f = _frame()
    pidx = shape_column_indices(derived=True, which="p")
    X = build_features(f, derived=True)
    g = dict(f)
    g["e1_input_rot0_p"], g["e2_input_rot0_p"] = apply_shear(
        f["e1_input_rot0_p"], f["e2_input_rot0_p"], 0.05, 0.0)
    Xs = build_features(g, derived=True)
    changed = np.where(~np.isclose(X, Xs, atol=1e-12).all(axis=0))[0]
    assert set(changed.tolist()) <= set(pidx), (
        f"the primary shift changed columns {changed.tolist()} but only {pidx} are its channel")


def test_the_two_channels_are_disjoint():
    s = shape_column_indices(derived=True, which="s")
    p = shape_column_indices(derived=True, which="p")
    assert not set(s) & set(p), "the self and blend channels must not share a column"
    names = feature_names(derived=True)
    assert [names[i] for i in p] == ["e1_input_rot0_p", "e2_input_rot0_p"]
    assert [names[i] for i in s] == ["e1_input_rot0_s", "e2_input_rot0_s"]


def test_shifted_columns_default_to_the_neighbour():
    """Backwards compatibility: every pre-dual caller omitted `which` and meant the neighbour."""
    f = _frame()
    a = shifted_shape_columns(f, 0.02)
    b = shifted_shape_columns(f, 0.02, which="s")
    for key in ("e1+", "e1-", "e2+", "e2-"):
        assert np.allclose(a[key][0], b[key][0]) and np.allclose(a[key][1], b[key][1])
    p = shifted_shape_columns(f, 0.02, which="p")
    assert not np.allclose(a["e1+"][0], p["e1+"][0]), "the channels must shift different columns"


def test_dual_mode_blinds_the_residual_flow_to_both_shapes():
    """In dual mode BOTH shape channels must be hidden from the residual flow.

    Leaving the primary's shape visible once the self response is supervised would let the density
    term absorb the STRONGER of the two responses. Nothing would raise; the model would simply
    under-predict. The blend-only default must be unchanged.
    """
    d = len(feature_names(derived=True))
    _, cfg_blend = build_model(d)
    _, cfg_dual = build_model(d, blind_flow_to_primary_shape=True)
    assert cfg_blend["flow_drop_indices"] == shape_column_indices(derived=True, which="s")
    assert cfg_dual["flow_drop_indices"] == sorted(
        shape_column_indices(derived=True, which="s")
        + shape_column_indices(derived=True, which="p"))


def test_each_channel_reads_its_own_coefficient():
    """With a linear mean head the trace/2 estimator must return that head's coefficient EXACTLY.

    The trace/2 of the Mobius Jacobian at g=0 is identically 1 -- d(e1')/dg1 = 1 - e1^2 + e2^2 and
    d(e2')/dg2 = 1 + e1^2 - e2^2 average to 1 for every shape -- so a linear head with coefficient
    `a` on a channel must give R = a on that channel, with no shape dependence left over. Wiring
    both channels to the same columns, or mixing up which index belongs to which galaxy, would give
    R_self == R_blend: a plausible-looking number and completely wrong.
    """
    f = _frame(n=256, seed=3)
    X = build_features(f, derived=True)
    d = X.shape[1]
    sidx = shape_column_indices(True, "s")
    pidx = shape_column_indices(True, "p")

    w = torch.zeros(2, d, dtype=torch.float32)
    w[0, sidx[0]] = w[1, sidx[1]] = 0.11       # neighbour channel
    w[0, pidx[0]] = w[1, pidx[1]] = 0.77       # primary channel, deliberately different

    class _LinearHead:
        def _mu(self, c):
            return c @ w.T

    model = _LinearHead()
    delta, ctx = 0.02, torch.as_tensor(X)
    for which, idx, expect in (("s", sidx, 0.11), ("p", pidx, 0.77)):
        sh = shifted_shape_columns(f, delta, which)
        built = {}
        for k in ("e1+", "e1-", "e2+", "e2-"):
            c = ctx.clone()
            c[:, idx[0]] = torch.as_tensor(sh[k][0], dtype=c.dtype)
            c[:, idx[1]] = torch.as_tensor(sh[k][1], dtype=c.dtype)
            built[k] = c
        r = response_from_contexts(model, ctx, built, (1.0, 1.0), delta).numpy()
        assert np.allclose(r, expect, rtol=1e-3, atol=1e-4), (
            f"channel {which}: got {r[:3]}, expected {expect} for every row")


def test_collapse_to_primaries_uses_the_primary_as_the_unit():
    """The self label is one number per primary repeated over its k rows.

    Averaging rows instead of primaries would (a) over-weight crowded primaries, and (b) shrink the
    error bar by ~sqrt(k) against labels that are literally identical. Both are silent.
    """
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "train_blend_flow.py")
    spec = importlib.util.spec_from_file_location("_tbf", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pid = np.array([0, 0, 0, 1, 2, 2])
    label = np.array([0.9, 0.9, 0.9, 0.5, 0.2, 0.2])      # constant within a primary, as built
    pred = np.array([1.0, 0.8, 0.9, 0.5, 0.3, 0.1])
    feat = np.array([21.0, 21.0, 21.0, 24.0, 25.0, 25.0])
    lab, prd, cnt, (ft,) = mod.collapse_to_primaries(pid, label, pred, feat)
    assert np.allclose(lab, [0.9, 0.5, 0.2])
    assert np.allclose(prd, [0.9, 0.5, 0.2])              # the mean over each primary's rows
    assert np.allclose(cnt, [3, 1, 2])
    assert np.allclose(ft, [21.0, 24.0, 25.0])
    # the crowded primary must NOT dominate the population mean
    assert abs(lab.mean() - np.mean([0.9, 0.5, 0.2])) < 1e-12
    assert abs(label.mean() - lab.mean()) > 0.05, "the row mean and primary mean must differ here"


def test_self_and_blend_labels_are_the_same_construction():
    """`self_truth` must be `blend_truth` with the primary's shear direction substituted.

    Both project the SAME measured-shape difference; only the direction differs. If they ever stop
    being the same construction, the two responses stop being the same kind of number and adding
    them -- the whole point of the merge -- becomes meaningless.
    """
    import importlib.util
    import os
    import pandas as pd
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "_bbp", os.path.join(root, "scripts", "build_blend_pairset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(11)
    n = 500
    base = pd.DataFrame({
        "gamma1_input_p": rng.normal(0, 0.05, n), "gamma2_input_p": rng.normal(0, 0.05, n),
        "gamma1_input_s": rng.normal(0, 0.05, n), "gamma2_input_s": rng.normal(0, 0.05, n),
        "measured_ngmix_g1_g": rng.normal(0, 0.3, n), "measured_ngmix_g1_0": rng.normal(0, 0.3, n),
        "measured_ngmix_g2_g": rng.normal(0, 0.3, n), "measured_ngmix_g2_0": rng.normal(0, 0.3, n),
    })
    # feeding the PRIMARY's shear into the neighbour slots must reproduce self_truth exactly
    swapped = base.copy()
    swapped["gamma1_input_s"] = base["gamma1_input_p"]
    swapped["gamma2_input_s"] = base["gamma2_input_p"]
    assert np.allclose(mod.self_truth(base), mod.blend_truth(swapped), atol=1e-12)
    # and the 45-degree null must be the spin-2 rotation of it, not a re-derivation
    assert np.allclose(mod.self_truth(base, rotate45=True),
                       mod.blend_truth(swapped, rotate45=True), atol=1e-12)


def test_crowding_columns_match_the_flow1_definition():
    """Hand-computed shell sums, so the block is the SAME quantity flow #1 conditions on.

    A lookalike that merely correlates with crowding would still fit, and the failure would only show
    up as a disagreement with flow #1 much later. Shells 0-3" and 3-7", zero point 30, normalised by
    the aperture rms -- all copied from `scripts/build_crowding_lookup.py`.
    """
    from sbs_shear.blend_flow import aperture_rms, crowding_columns
    frame = {
        "pid": np.array([0, 0, 1]),
        "r_input_s": np.array([24.0, 25.0, 23.0]),
        "distance": np.array([1.0, 5.0, 2.0]),      # near, far, near
        "k": np.array([2, 2, 1]),
    }
    ar = aperture_rms()
    f24, f25, f23 = [10.0 ** (-0.4 * (m - 30.0)) for m in (24.0, 25.0, 23.0)]
    got = crowding_columns(frame)
    assert np.allclose(got["nbr_flux_near"], np.log10(1 + np.array([f24, f24, f23]) / ar))
    assert np.allclose(got["nbr_flux_far"], np.log10(1 + np.array([f25, f25, 0.0]) / ar))
    assert np.allclose(got["nbr_flux_max"], np.log10(1 + np.array([f24, f24, f23]) / ar))
    assert np.allclose(got["log_k"], np.log([2.0, 2.0, 1.0]))


def test_crowding_uses_the_stored_k_not_the_rows_present():
    """`log_k` must describe the GALAXY, not the sample. Recomputing it from the rows in hand would
    silently disagree with the stored count on any filtered or subsampled frame."""
    from sbs_shear.blend_flow import crowding_columns
    frame = {"pid": np.array([0, 0]), "r_input_s": np.array([24.0, 25.0]),
             "distance": np.array([1.0, 2.0]), "k": np.array([7, 7])}
    assert np.allclose(crowding_columns(frame)["log_k"], np.log([7.0, 7.0]))


def test_crowding_features_are_shear_invariant():
    """Neither shear shift may touch the crowding block.

    The block is built from magnitudes, distances and counts, so it CANNOT carry shape information --
    but if it ever did it would open a third response channel and the measured responses would stop
    being the quantities the labels measure. Nothing would raise.
    """
    f = dict(_frame(n=64, seed=5))
    rng = np.random.default_rng(5)
    f["pid"] = np.sort(rng.integers(0, 16, 64))
    f["k"] = np.bincount(f["pid"])[f["pid"]]
    X = build_features(f, derived=True, crowding=True)
    names = feature_names(derived=True, crowding=True)
    for which in ("s", "p"):
        g = dict(f)
        c1, c2 = f"e1_input_rot0_{which}", f"e2_input_rot0_{which}"
        g[c1], g[c2] = apply_shear(f[c1], f[c2], 0.05, -0.03)
        Xs = build_features(g, derived=True, crowding=True)
        changed = {names[i] for i in np.where(~np.isclose(X, Xs, atol=1e-12).all(axis=0))[0]}
        assert changed <= {c1, c2}, f"shifting {which} changed {changed}"


def test_crowding_appends_and_leaves_shape_indices_put():
    """Adding the block must not move the shape channels, or every existing checkpoint's stored
    indices would silently point at the wrong columns."""
    for which in ("s", "p"):
        assert (shape_column_indices(True, which, crowding=False)
                == shape_column_indices(True, which, crowding=True))


def test_cluster_sem_fast_matches_the_original():
    """The aggregator's precomputed-index clustering must equal `eval_blend_flow.cluster_sem`.

    It exists only because re-deriving the primary index with a sort on every call made the 16-seed
    aggregate intractable. A speed-up that quietly changed the ERROR BAR would rescale every chi2 in
    the ensemble report and nothing would raise, so the equivalence is pinned rather than assumed.
    """
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys as _sys
    _sys.path.insert(0, os.path.join(root, "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_agg", os.path.join(root, "scripts", "agg_blendflow_ensemble.py"))
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    from eval_blend_flow import cluster_sem

    rng = np.random.default_rng(7)
    n = 5000
    pid = np.sort(rng.integers(0, 900, n)) * 7 + 13     # deliberately NOT 0..n-1 and not contiguous
    vals = rng.normal(0.02, 3.9, n)
    _, inv = np.unique(pid, return_inverse=True)
    n_prim = int(inv.max()) + 1
    for _ in range(5):
        mask = rng.random(n) < rng.uniform(0.15, 0.95)
        a = cluster_sem(vals, pid, mask)
        b = agg.cluster_sem_fast(vals, inv, n_prim, mask)
        assert np.isclose(a, b, rtol=1e-12, atol=0), f"{a} != {b}"


if __name__ == "__main__":
    # `pytest` is NOT installed in the `sims1` env, so this file also runs standalone:
    #   PYTHONPATH=<repo>:<blendemu> python tests/test_blend_flow.py
    import traceback

    _fns = sorted(k for k in dict(globals()) if k.startswith("test_"))
    _bad = 0
    for _f in _fns:
        try:
            globals()[_f]()
            print("PASS", _f)
        except Exception:
            _bad += 1
            print("FAIL", _f)
            traceback.print_exc()
    print(f"\n{len(_fns) - _bad}/{len(_fns)} passed")
    raise SystemExit(1 if _bad else 0)

