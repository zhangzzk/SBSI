"""Nonzero-shear catalogue context of the factorized like-1.0 flow equals standardizing a sheared frame."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sbsi.finite.likelihood import CatalogueLikelihood
from sbsi.galsbi.features import FEATURES
from sbsi.measurement_model import load_measurement_model
from sbsi.models import get_model
from sbsi.shear_map import apply_shear_to_ellipticity


def test_like_1_0_shear_context_matches_sheared_frame():
    flow = load_measurement_model(get_model('like-1.0-galsbi').flow_checkpoints[0], device='cpu')
    pre = flow.condition_preprocessor
    rng = np.random.default_rng(1)
    n = 64
    e = rng.uniform(0, .7, n)*np.exp(2j*np.pi*rng.random(n))
    frame = pd.DataFrame(np.column_stack((e.real, e.imag, rng.uniform(.5, 4, n), rng.uniform(19, 25, n),
                                          rng.uniform(.1, 1, n), rng.uniform(0, 2, (n, 3)))), columns=FEATURES)
    raw = pre.raw_tensor_from_frame(frame, torch.device('cpu'))
    context = torch.as_tensor(pre.transform_frame(frame))
    fake = SimpleNamespace(flow_model=flow, raw_zero_context_tensor=lambda: raw)
    rows = torch.arange(n)
    for g in ((.002, 0.), (0., -.002), (.0213, .0047)):
        got = CatalogueLikelihood._shear_standardized_context(fake, context, rows, *g)
        sheared = frame.copy()
        sheared['e1_input_p'], sheared['e2_input_p'] = apply_shear_to_ellipticity(
            frame.e1_input_p.to_numpy(), frame.e2_input_p.to_numpy(), *g)
        np.testing.assert_allclose(got.numpy(), pre.transform_frame(sheared), rtol=0, atol=2e-6)
    assert CatalogueLikelihood._shear_standardized_context(fake, context, rows, 0., 0.) is context
