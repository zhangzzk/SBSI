import copy

import torch
import numpy as np
import pandas as pd
import pytest

from sbsi.flow_training import epoch_response, freeze_except_mean_head
from sbsi.flow_training import transforms_from_reference, _append_to_priority_sample, _finalize_priority_sample
from sbsi.measurement_model import ConditionalMeanFlow
from sbsi.measurement_model import TargetStandardizer
from sbsi.selection_model import TabularPreprocessor


def test_expanded_sample_keeps_reference_transforms_and_uncapped_rows():
    initial = pd.DataFrame({'x':[1.,2.,4.], 'y':[2.,3.,7.]})
    condition=TabularPreprocessor.fit(initial,['x'])
    target=TargetStandardizer.fit(initial,['y'])
    reference={'condition_preprocessor':condition.to_state(),'target_transform':target.to_state()}
    expanded=pd.DataFrame({'x':[20.,100.],'y':[4.,1000.]})
    c,t=transforms_from_reference(reference,['x'],['y'])
    np.testing.assert_array_equal(c.transform_frame(expanded),condition.transform_frame(expanded))
    np.testing.assert_array_equal(t.transform_frame(expanded),target.transform_frame(expanded))
    np.testing.assert_allclose(t.inverse_transform_array(t.transform_frame(expanded)),expanded[['y']],atol=1e-4)
    with pytest.raises(ValueError,match='order'):
        transforms_from_reference(reference,['wrong'],['y'])
    reservoir=None
    for frame in (initial,expanded):
        reservoir=_append_to_priority_sample(reservoir,frame,0,np.random.default_rng(1))
    pd.testing.assert_frame_equal(_finalize_priority_sample(reservoir,0),pd.concat([initial,expanded],ignore_index=True))


class _ToyMeanDensity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_net = torch.nn.Linear(2, 4, bias=False)

    def _shift(self, context, _u=None):
        return self.mean_net(context)

    def log_prob(self, target, context):
        return -0.5 * (target - self._shift(context)).square().sum(dim=1)


def _toy_loader():
    target = torch.zeros((4, 4))
    context = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    weight = torch.ones(4)
    shifted1 = context + torch.tensor([0.1, 0.0])
    shifted2 = context + torch.tensor([0.0, 0.1])
    bin_id = torch.zeros(4, dtype=torch.long)
    return [(target, context, weight, context, shifted1, shifted2, bin_id)]


def test_preservation_metric_is_zero_at_reference_and_positive_after_mean_shift():
    model = _ToyMeanDensity()
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    target_matrix = torch.zeros((1, 4))

    *_, initial = epoch_response(
        model,
        _toy_loader(),
        torch.device("cpu"),
        [1.0, 1.0],
        0.1,
        target_matrix,
        0.0,
        response_components="matrix",
        nll_weight=0.0,
        reference_model=reference,
        preserve_weight=1.0,
    )
    assert initial["preserve"] == 0.0

    with torch.no_grad():
        model.mean_net.weight.add_(0.2)
    *_, shifted = epoch_response(
        model,
        _toy_loader(),
        torch.device("cpu"),
        [1.0, 1.0],
        0.1,
        target_matrix,
        0.0,
        response_components="matrix",
        nll_weight=0.0,
        reference_model=reference,
        preserve_weight=1.0,
    )
    assert shifted["preserve"] > 0.0


def test_freeze_except_mean_head_leaves_only_explicit_mean_trainable():
    model = ConditionalMeanFlow(
        target_dim=4,
        context_dim=3,
        hidden_dim=8,
        n_layers=1,
        n_flows=2,
        mean_hidden=5,
    )
    names = freeze_except_mean_head(model)
    assert names
    assert all(name.startswith("mean_net.") for name in names)
    assert all(
        parameter.requires_grad == name.startswith("mean_net.")
        for name, parameter in model.named_parameters()
    )
