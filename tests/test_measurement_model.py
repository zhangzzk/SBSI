import numpy as np
import pandas as pd
import torch

from sbsi.measurement_model import (
    ConditionalAffineFlow,
    ConditionalMeanFlow,
    ConditionalMeanFlowRA,
    MEASUREMENT_CONDITION_FEATURE_SETS,
    TargetStandardizer,
    add_measurement_target_features,
    build_flow,
    raw_columns_for_measurement_targets,
)


def test_rblend_ablation_appends_one_condition_feature():
    baseline = MEASUREMENT_CONDITION_FEATURE_SETS["g0_meas_crowd_conc_szfl_noz"]
    candidate = MEASUREMENT_CONDITION_FEATURE_SETS["g0_meas_crowd_conc_szfl_noz_rblend"]

    assert candidate == [*baseline, "r_blend"]


def test_measurement_target_features_are_finite_spin2_shape():
    frame = pd.DataFrame({
        "measured_flux_auto": [10.0],
        "measured_flux_radius": [2.0],
        "measured_a_image": [3.0],
        "measured_b_image": [1.0],
        "measured_theta_image": [0.0],
    })

    out = add_measurement_target_features(frame)

    assert np.isfinite(out["measured_log_flux_auto"]).all()
    assert np.isfinite(out["measured_log_flux_radius"]).all()
    np.testing.assert_allclose(out["measured_e1_image"], [0.5])
    np.testing.assert_allclose(out["measured_e2_image"], [0.0], atol=1.0e-15)


def test_raw_columns_for_engineered_measurement_targets():
    columns = raw_columns_for_measurement_targets([
        "measured_log_flux_auto",
        "measured_e1_image",
    ])

    assert columns == {
        "measured_flux_auto",
        "measured_a_image",
        "measured_b_image",
        "measured_theta_image",
    }


def test_target_standardizer_round_trips():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
    })

    transform = TargetStandardizer.fit(frame, ["a", "b"])
    scaled = transform.transform_frame(frame)
    recovered = transform.inverse_transform_array(scaled)

    np.testing.assert_allclose(recovered, frame[["a", "b"]].to_numpy(dtype=np.float32))


def test_conditional_affine_flow_inverse_and_log_prob_are_finite():
    torch.manual_seed(5)
    model = ConditionalAffineFlow(
        target_dim=4,
        context_dim=3,
        hidden_dim=8,
        n_layers=1,
        n_flows=2,
    )
    x = torch.randn(6, 4)
    context = torch.randn(6, 3)

    z, inverse_logdet = model.inverse(x, context)
    recovered, forward_logdet = model.forward(z, context)
    log_prob = model.log_prob(x, context)
    samples = model.sample(context, n_samples=3)

    torch.testing.assert_close(recovered, x)
    torch.testing.assert_close(inverse_logdet + forward_logdet, torch.zeros_like(inverse_logdet))
    assert torch.isfinite(log_prob).all()
    assert samples.shape == (6, 3, 4)


# ---------------------------------------------------------------------------
# ConditionalMeanFlow / ConditionalMeanFlowRA (realisation-aware head)
# ---------------------------------------------------------------------------

_RA_KW = dict(target_dim=4, context_dim=5, base_flow="affine", mean_hidden=8,
              hidden_dim=8, n_layers=1, n_flows=4, flow_drop_indices=(0, 1))


def _make_ra_pair(seed=11, ra_hidden=6):
    """A ConditionalMeanFlowRA and a plain ConditionalMeanFlow sharing every non-RA weight."""
    torch.manual_seed(seed)
    ra = ConditionalMeanFlowRA(ra_hidden=ra_hidden, ra_indices=(2, 3), ra_targets=(0, 1), **_RA_KW)
    plain = ConditionalMeanFlow(**_RA_KW)
    plain.load_state_dict({k: v for k, v in ra.state_dict().items() if not k.startswith("ra_")})
    ra.eval()
    plain.eval()
    return ra, plain


def test_ra_head_zero_init_reduces_exactly_to_mean_flow():
    ra, plain = _make_ra_pair()
    x = torch.randn(7, 4)
    context = torch.randn(7, 5)

    assert torch.equal(ra.log_prob(x, context), plain.log_prob(x, context))

    torch.manual_seed(3)
    s_ra = ra.sample(context, n_samples=5)
    torch.manual_seed(3)
    s_plain = plain.sample(context, n_samples=5)
    assert torch.equal(s_ra, s_plain)


def test_ra_head_transform_has_unit_jacobian():
    ra, _ = _make_ra_pair()
    with torch.no_grad():
        for p in ra.ra_net[-1].parameters():
            p.normal_(0.0, 0.4)
    context = torch.randn(1, 5)

    def to_z(x):
        resid = x - ra._mu(context)
        u = resid.index_select(-1, ra.ra_indices)
        return (resid - ra._A(context, u)).squeeze(0)

    jac = torch.autograd.functional.jacobian(to_z, torch.randn(1, 4))
    jac = jac.reshape(4, 4)
    assert abs(float(torch.det(jac)) - 1.0) < 1.0e-5


def test_ra_head_sample_round_trips_through_closed_form_inverse():
    ra, _ = _make_ra_pair()
    with torch.no_grad():
        for p in ra.ra_net[-1].parameters():
            p.normal_(0.0, 0.4)
    context = torch.randn(6, 5)
    x = ra.sample(context, n_samples=3).reshape(-1, 4)
    ctx = context[:, None, :].expand(6, 3, 5).reshape(-1, 5)

    resid = x - ra._mu(ctx)
    z = resid - ra._A(ctx, resid.index_select(-1, ra.ra_indices))          # forward (log_prob) map
    back = z + ra._A(ctx, z.index_select(-1, ra.ra_indices)) + ra._mu(ctx)  # closed-form inverse
    torch.testing.assert_close(back, x, rtol=1.0e-5, atol=1.0e-6)


def test_ra_head_gives_per_draw_response_variation():
    """The whole point: with A engaged the response VARIES between draws of the same object.

    Common random numbers make the residual draw identical at both contexts, so for the plain
    head the per-draw response is exactly mu(c1)-mu(c0) -- constant within an object.
    """
    ra, plain = _make_ra_pair()
    with torch.no_grad():
        for p in ra.ra_net[-1].parameters():
            p.normal_(0.0, 0.4)
    c0 = torch.randn(4, 5)
    c1 = c0.clone()
    c1[:, 0] += 0.05          # shift the (flow-blind) shape context columns
    c1[:, 1] -= 0.03

    def crn(model):
        torch.manual_seed(17)
        a = model.sample(c0, n_samples=64)
        torch.manual_seed(17)
        b = model.sample(c1, n_samples=64)
        return (b - a)[:, :, 0]

    d_plain = crn(plain)
    d_ra = crn(ra)
    assert float(d_plain.std(dim=1).max()) < 1.0e-6      # realisation-blind: no per-draw spread
    assert float(d_ra.std(dim=1).min()) > 1.0e-6         # realisation-aware: it varies


def test_flow_context_is_blind_to_the_shifted_shape_columns():
    """CRN premise: the residual draw does not move when only the blind columns move."""
    ra, _ = _make_ra_pair()
    c0 = torch.randn(4, 5)
    c1 = c0.clone()
    c1[:, 0] += 0.05
    c1[:, 1] -= 0.03
    torch.manual_seed(23)
    z0 = ra.flow.sample(ra._flow_ctx(c0), n_samples=8)
    torch.manual_seed(23)
    z1 = ra.flow.sample(ra._flow_ctx(c1), n_samples=8)
    assert torch.equal(z0, z1)


def test_ra_head_rejects_overlapping_read_and_write_channels():
    # written without pytest.raises so the file also runs under a bare python driver
    try:
        ConditionalMeanFlowRA(ra_hidden=4, ra_indices=(1, 2), ra_targets=(0, 1), **_RA_KW)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping ra_indices/ra_targets must raise (breaks unit Jacobian)")

    try:
        ConditionalMeanFlowRA(ra_hidden=0, ra_indices=(2, 3), ra_targets=(0, 1), **_RA_KW)
    except ValueError:
        pass
    else:
        raise AssertionError("ra_hidden=0 must raise; use flow_type='mean_affine' instead")

    ra, _ = _make_ra_pair()
    try:
        ra.set_ols_mean_and_freeze(np.zeros((3, 5)), np.zeros((3, 4)))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("OLS mean-freeze must be refused for the realisation-aware head")


def test_build_flow_dispatches_the_realisation_aware_type():
    cfg = dict(flow_type="mean_affine_ra", target_dim=4, context_dim=5, mean_hidden=8,
               hidden_dim=8, n_layers=1, n_flows=4, ra_hidden=6,
               ra_indices=[2, 3], ra_targets=[0, 1])
    model = build_flow(cfg)
    assert isinstance(model, ConditionalMeanFlowRA)
    assert float(model.ra_net[-1].weight.abs().sum()) == 0.0
