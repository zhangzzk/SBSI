from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.flow_physical_disk import ConditionalPhysicalDiskFlow
from sbsi.selection_model import SelectionMLP, SelectionModelBundle, TabularPreprocessor
from sbsi.summary_gradients import SummaryGradients


def fixture():
    torch.manual_seed(19)
    pre = TabularPreprocessor(list('abcdefgh'), np.zeros(8), np.zeros(8),
                             np.ones(8), False, ())
    model = ConditionalPhysicalDiskFlow(target_dim=4, context_dim=8, hidden_dim=8,
        n_layers=1, n_flows=2, disk_map='radial_tanh', disk_shape_means=[0, 0],
        disk_shape_scales=[1, 1], disk_coordinate_means=[0, 0], disk_coordinate_scales=[1, 1],
        physical_map='softplus', physical_means=[1, 1], physical_scales=[1, 1],
        physical_units=[1, 1], physical_coordinate_means=[0, 0], physical_coordinate_scales=[1, 1]).double().eval()
    flow = SimpleNamespace(model=model, device=torch.device('cpu'), condition_preprocessor=pre,
        target_transform=SimpleNamespace(scales=np.ones(4), target_names=list('wxyz')))
    detector = SelectionModelBundle(SelectionMLP(input_dim=8, hidden_dim=8, n_layers=1).double(), pre)
    values = np.array([[.2, -.3, 1.2, .8, .5, .1, .2, .3, .4],
                       [-.1, .2, .8, 1.1, .7, .3, .4, .5, -.2]])
    observed = pd.DataFrame([[.1, -.2, 1.1, 2.], [.2, .3, 1.3, 1.8]], columns=list('wxyz'))
    observed['measured_shape_log_disk_gap'] = np.log1p(-np.square(observed[['w','x']]).sum(1))
    return flow, detector, values, observed


@pytest.mark.parametrize('shear', [[0., 0.], [.02, -.01]])
def test_context_and_shear_derivatives_match_independent_central_differences(shear):
    flow, detector, x, y = fixture()
    evaluator = SummaryGradients(flow, detector, batch_size=3)
    before = [p.clone() for p in flow.model.parameters()]
    _, gradient = evaluator.evaluate(x, y, shear)
    for col in range(8):
        up, down = x.copy(), x.copy()
        up[:, col] += 1e-5; down[:, col] -= 1e-5
        finite = (evaluator.evaluate(up, y, shear)[0]-evaluator.evaluate(down, y, shear)[0])/2e-5
        np.testing.assert_allclose(gradient[:, col], finite, rtol=2e-6, atol=2e-8)
    _, gradient = evaluator.evaluate(x, y, shear, wrt='shear')
    for col in range(2):
        up, down = np.array(shear), np.array(shear)
        up[col] += 1e-5; down[col] -= 1e-5
        finite = (evaluator.evaluate(x, y, up)[0]-evaluator.evaluate(x, y, down)[0])/2e-5
        np.testing.assert_allclose(gradient[:, col], finite, rtol=2e-6, atol=2e-8)
    for original, parameter in zip(before, flow.model.parameters()):
        assert torch.equal(original, parameter) and parameter.grad is None


def test_padding_does_not_multiply_last_row_gradient():
    flow, detector, x, y = fixture()
    for wrt in ['context', 'shear']:
        a = SummaryGradients(flow, detector, batch_size=2).evaluate(x, y, [.02, -.01], wrt=wrt)
        b = SummaryGradients(flow, detector, batch_size=5).evaluate(x, y, [.02, -.01], wrt=wrt)
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)


@pytest.mark.parametrize('shear', [[0., 0.], [.02, -.01]])
def test_full_shear_hessian_matches_gradient_differences_and_padding(shear):
    flow, detector, x, y = fixture()
    evaluator = SummaryGradients(flow, detector, batch_size=3)
    density, score, hessian = evaluator.evaluate(x, y, shear, wrt='shear', hessian=True)
    np.testing.assert_allclose(hessian, hessian.transpose(0, 2, 1), atol=1e-11)
    for axis in range(2):
        plus, minus = np.array(shear), np.array(shear)
        plus[axis] += 1e-5; minus[axis] -= 1e-5
        numeric = (evaluator.evaluate(x, y, plus, wrt='shear')[1]
                   -evaluator.evaluate(x, y, minus, wrt='shear')[1])/2e-5
        np.testing.assert_allclose(hessian[:, :, axis], numeric, atol=1e-7, rtol=2e-6)
    padded = SummaryGradients(flow, detector, batch_size=5).evaluate(x, y, shear, wrt='shear', hessian=True)
    for a, b in zip((density, score, hessian), padded):
        np.testing.assert_allclose(a, b, atol=1e-12)
    assert all(p.grad is None for p in flow.model.parameters())
