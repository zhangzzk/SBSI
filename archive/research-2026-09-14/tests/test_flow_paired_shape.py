import copy
import itertools

import numpy as np
import pytest
import torch

from sbsi.flow_paired_shape import (
    PairedShapePopulation, paired_cross_score, paired_shape_backward,
    summarize_paired_residuals, evaluate_paired_shape,
)
from scripts.train_circularized_grid_free import paired_shape_epoch, continue_fit
from scripts.continue_circularized_flow import nll_pass
from scripts.trial_circularized_flow import fit_nll_arm
from test_circularized_flow_continuation import training_fixture
from test_flow_physical_disk import config


def fixture():
    model, values, context, pre, target = training_fixture()
    pairs = np.arange(24).reshape(12, 2)
    rng = np.random.default_rng(119)
    pop = PairedShapePopulation(context, pairs, rng.normal(0, .03, (12, 2)),
                                np.tile([.03, .04], (12, 1)))
    return model, values, pop, pre, target


def test_independent_cross_product_has_no_mc_variance_gradient():
    mean = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(mean)
    scores = []
    for a, c in itertools.product((-1., 1.), repeat=2):
        scores.append(paired_cross_score(torch.stack((mean+a*scale, zero))[None],
                                        torch.stack((mean+c*scale, zero))[None]).mean())
    actual = torch.stack(scores).mean()
    torch.testing.assert_close(actual, mean.square()/4)
    gradient = torch.autograd.grad(actual, (mean, scale))
    torch.testing.assert_close(gradient[0], mean/2)
    torch.testing.assert_close(gradient[1], zero, rtol=0, atol=1e-15)


def test_pair_errors_do_not_cancel_and_rotation_preserves_loss():
    residual = torch.tensor([[1., 2.], [-1., -2.]], dtype=torch.float64)
    assert paired_cross_score(residual, residual).mean() == 1.25
    rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
    torch.testing.assert_close(paired_cross_score(residual@rotation, residual@rotation),
                              paired_cross_score(residual, residual))


def test_only_shape_and_correct_measured_forward_difference_enter_loss():
    class Physical(torch.nn.Module):
        target_dim, context_dim = 4, 2
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor([.1, .2, 100., 200.]))
        def forward_physical(self, z, context):
            value = context[:, :1]*self.offset+z
            return value, torch.zeros(len(z))
    model = Physical()
    context = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    pop = PairedShapePopulation(context, [[0, 1], [2, 3]], np.tile([.05, .1], (2, 1)),
                                np.tile([.03, .04], (2, 1)))
    r = pop.residual(model, torch.arange(2), 4, 3)
    torch.testing.assert_close(r, torch.tensor([[1., 2.], [1., 2.]], dtype=torch.float64), rtol=1e-5, atol=1e-5)
    paired_cross_score(r, r).mean().backward()
    torch.testing.assert_close(model.offset.grad[2:], torch.zeros(2), rtol=0, atol=0)
    with pytest.raises(ValueError, match="nonzero"):
        PairedShapePopulation(context, [[0, 1]], [[0., 0.]], [[0., 0.]])


def test_chunked_backward_matches_direct_graph_and_preserves_existing_gradient():
    model, _, pop, _, _ = fixture()
    seed, count, chunk = 51, 7, 3
    rows = torch.as_tensor(np.random.default_rng(seed).integers(len(pop.pairs), size=count))
    scores = []
    for block, start in enumerate(range(0, count, chunk)):
        ids = rows[start:start+chunk]
        scores.append(paired_cross_score(pop.residual(model, ids, 4, seed+100000+2*block),
                                        pop.residual(model, ids, 4, seed+100001+2*block)).sum()/count)
    score = sum(scores)
    score.backward()
    expected = [p.grad.clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    actual = paired_shape_backward(model, pop, count, 4, chunk_size=chunk, seed=seed, weight=.3)
    np.testing.assert_allclose(actual['loss'], float(score.detach()), rtol=1e-10)
    for p, gradient in zip(model.parameters(), expected):
        torch.testing.assert_close(p.grad, 1+.3*gradient, rtol=2e-5, atol=2e-5)


def test_validation_uses_same_pairs_with_all_distinct_latent_groups():
    model, _, pop, _, _ = fixture()
    result = evaluate_paired_shape(model, pop, np.arange(7), draws=4, groups=4, batch_size=3, seed=150)
    residuals = result['group_residuals']
    expected = np.mean([np.sum(a*c, axis=-1)/4 for i,a in enumerate(residuals)
                        for j,c in enumerate(residuals) if i != j], axis=0)
    np.testing.assert_allclose(result['per_pair'], expected, rtol=1e-12, atol=1e-12)
    constant = summarize_paired_residuals(np.broadcast_to([[1., 0.], [-1., 0.]], (4, 2, 2)))
    assert constant['shape_loss'] == .25
    assert constant['mc_jackknife_se'] == 0


def test_zero_weight_grid_free_epoch_preserves_nll_updates():
    model, values, pop, _, _ = fixture()
    clone = copy.deepcopy(model)
    opt = [torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0.) for m in (model, clone)]
    recipe = dict(batch_size=7, pairs_per_step=5, training_draws=4,
                  response_training_chunk_size=3, training_response_seed=12000, shape_weight=0.)
    expected = nll_pass(model, values, pop.context, 7, optimizer=opt[0], seed=51, clip=5.)
    actual = paired_shape_epoch(clone, values, pop.context, pop, opt[1], recipe, 51)
    assert actual['nll'] == expected['nll']
    for a,b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_grid_free_checkpoint_selection_and_committed_resume(tmp_path):
    _, values, pop, pre, target = fixture()
    train, val = (values[:18], pop.context[:18]), (values[18:], pop.context[18:])
    first_recipe = dict(seed=501, learning_rate=1e-4, epochs=1, rows_per_epoch=13,
                        batch_size=6, early_stopping_patience=20)
    fit_nll_arm(tmp_path/'parent', 'parent', config(), pre, target, train, val, first_recipe, {})
    initial = torch.load(tmp_path/'parent/optimizer.pt', weights_only=False)
    parent = torch.load(tmp_path/'parent/epoch001.pt', weights_only=False)
    recipe = first_recipe | dict(max_epoch=3, min_lr=1e-7, diagnostic_interval=2,
        pairs_per_step=5, training_draws=4, response_training_chunk_size=3,
        training_response_seed=12000, shape_weight=1.)
    def diagnostic(model, epoch):
        return evaluate_paired_shape(model, pop, np.arange(3), draws=4, groups=4, batch_size=3, seed=190)
    for name, cap in [('whole', 3), ('resumed', 2)]:
        continue_fit(tmp_path/name, 'paired', parent, initial, None, train, val,
                     recipe | dict(max_epoch=cap), {}, diagnostic, pop, train_epoch=paired_shape_epoch)
    continue_fit(tmp_path/'resumed', 'paired', parent, initial, None, train, val, recipe, {},
                 diagnostic, pop, resume=True, train_epoch=paired_shape_epoch)
    a = torch.load(tmp_path/'whole/optimizer.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed/optimizer.pt', weights_only=False)
    for k in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][k], b['state_dict'][k], rtol=0, atol=0)
    for k in a['optimizer']['state']:
        for key in a['optimizer']['state'][k]:
            torch.testing.assert_close(a['optimizer']['state'][k][key], b['optimizer']['state'][k][key], rtol=0, atol=0)
    assert a['scheduler'] == b['scheduler']
    assert a['best_epoch'] == min(a['history'], key=lambda r:r['selection_metric'])['epoch']
