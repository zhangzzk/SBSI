import numpy as np
import pandas as pd
import pytest
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.measurement_model import TargetStandardizer
from sbsi.selection_model import TabularPreprocessor
from scripts.refine_flow_shape_augmentation import eligible_checkpoint, standardized_sign_action


def test_standardized_action_matches_physical_reflection_with_nonzero_means():
    raw_context = np.arange(32, dtype=np.float32).reshape(4, 8)/20
    raw_target = np.arange(16, dtype=float).reshape(4, 4)/10+.1
    condition = TabularPreprocessor.fit(pd.DataFrame(raw_context, columns=FLOW_FEATURES), FLOW_FEATURES)
    target = TargetStandardizer.fit(pd.DataFrame(raw_target, columns=FLOW_TARGETS), FLOW_TARGETS, dtype='float64')
    signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    context = torch.as_tensor(condition.transform_frame(pd.DataFrame(raw_context, columns=FLOW_FEATURES)))
    values = torch.as_tensor(target.transform_array(raw_target))
    changed_context, changed_target = standardized_sign_action(context, values, torch.as_tensor(signs),
        condition.means, condition.scales, target.means, target.scales)
    raw_context[:, :2] *= signs
    raw_target[:, :2] *= signs
    expected_context = condition.transform_frame(pd.DataFrame(raw_context, columns=FLOW_FEATURES))
    np.testing.assert_allclose(changed_context, expected_context, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(changed_target, target.transform_array(raw_target), atol=1e-14)
    torch.testing.assert_close(changed_context[:, 2:], context[:, 2:], rtol=0, atol=0)
    torch.testing.assert_close(changed_target[:, 2:], values[:, 2:], rtol=0, atol=0)


def test_missing_shape_context_cannot_be_augmented():
    context = torch.zeros(2, 16)
    context[0, 8] = 1
    with pytest.raises(ValueError, match='finite observed context'):
        standardized_sign_action(context, torch.zeros(2, 4), torch.ones(2, 2),
                                 np.zeros(8), np.ones(8), np.zeros(4), np.ones(4))


def test_density_improvement_cannot_buy_worse_response_or_nonfinite_score():
    assert eligible_checkpoint(-4.8, .35, -4.7, .36)
    assert not eligible_checkpoint(-4.8, .37, -4.7, .36)
    assert not eligible_checkpoint(-4.6, .35, -4.7, .36)
    assert not eligible_checkpoint(np.nan, .35, -4.7, .36)
