import torch

from sbsi.flow_training import complete_flow_context_means, epoch_response
from sbsi.measurement_model import ConditionalAffineFlow, ConditionalMeanFlow


def _plain_flow():
    torch.manual_seed(8)
    return ConditionalAffineFlow(
        target_dim=4,
        context_dim=2,
        hidden_dim=12,
        n_layers=2,
        n_flows=2,
    )


def test_complete_flow_means_are_seeded_common_draws_and_differentiable():
    model = _plain_flow()
    with torch.no_grad():
        for layer in model.layers:
            layer.net[-1].weight.normal_(std=0.02)
    first = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
    second = first + torch.tensor([0.05, 0.0])

    means_a = complete_flow_context_means(model, [first, second], 4, seed=31)
    means_b = complete_flow_context_means(model, [first, second], 4, seed=31)

    torch.testing.assert_close(means_a[0], means_b[0])
    torch.testing.assert_close(means_a[1], means_b[1])
    loss = (means_a[1] - means_a[0]).square().sum()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )


def test_complete_flow_response_epoch_updates_plain_flow_without_mean_head():
    model = _plain_flow()
    batch_size = 6
    context = torch.randn(batch_size, 2)
    delta = 0.02
    plus1 = context + torch.tensor([delta, 0.0])
    plus2 = context + torch.tensor([0.0, delta])
    minus1 = context - torch.tensor([delta, 0.0])
    minus2 = context - torch.tensor([0.0, delta])
    batch = (
        torch.randn(batch_size, 4),
        context,
        torch.ones(batch_size),
        context,
        plus1,
        plus2,
        minus1,
        minus2,
        torch.zeros(batch_size, dtype=torch.long),
        torch.zeros(batch_size),
        torch.zeros(batch_size),
        torch.zeros(batch_size),
        torch.zeros(batch_size),
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    nll, response, _, theta, _, _ = epoch_response(
        model,
        [batch],
        torch.device("cpu"),
        [1.0, 1.0],
        delta,
        torch.zeros((1, 4)),
        1.0,
        response_difference="central",
        optimizer=optimizer,
        sc23=[1.0, 1.0],
        lam_theta=1.0,
        response_components="matrix",
        response_estimator="complete_flow",
        response_draws=2,
    )

    assert torch.isfinite(torch.tensor([nll, response, theta])).all()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert not hasattr(model, "_shift")


def test_complete_flow_response_rejects_explicit_mean_flow():
    model = ConditionalMeanFlow(target_dim=4, context_dim=2, hidden_dim=8, n_flows=2)
    context = torch.zeros(2, 2)
    try:
        complete_flow_context_means(model, [context], 2)
    except ValueError as error:
        assert "plain flow" in str(error)
    else:
        raise AssertionError("expected explicit mean flow to be rejected")
