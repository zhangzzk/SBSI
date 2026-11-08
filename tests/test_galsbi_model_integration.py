"""Retained artifact replay and model-identity isolation at inference entry."""
from pathlib import Path
import ast
import importlib.util
import json
import types
import numpy as np
import pandas as pd
import pytest
import torch
from sbsi.measurement_model import load_measurement_model
from sbsi.models import get_model

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name', ['like-1.0-galsbi', 'galsbi-multistage'])
def test_original_factorized_checkpoint_replay(name):
    preset = get_model(name)
    preset.validate()
    d = torch.load(preset.flow_checkpoints[0], map_location='cpu', weights_only=False)
    source = ROOT/'doc/evidence/alignment_factorized_20261105/frame_model.py'
    spec = importlib.util.spec_from_file_location('original_frame_replay', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    # Execute only the original augmentation definitions, avoiding research runtime imports.
    tree = ast.parse((ROOT/'doc/evidence/neighbour_conditioned_flow_20261106/model.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ('ExtraInputLinear', 'augment')]
    scope = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-augment>', 'exec'), scope)
    from sbsi.selection_model import TabularPreprocessor
    base = TabularPreprocessor.from_state(d['condition']['base'])
    old = scope['augment'](original.FrameFlow(d['config'], base, 'cartesian'))
    old.load_state_dict(d['state_dict'])
    old.eval()
    bundle = load_measurement_model(preset.flow_checkpoints[0])
    raw = np.tile([.1, -.07, 1.5, 24.8, .3, .01, .1, .5], (8, 1))
    frame = pd.DataFrame(raw, columns=base.feature_names)
    frame['neighbour_reference_coefficient'] = np.linspace(.01, .4, len(frame))
    expected = base.transform_frame(frame)
    s = d['condition']
    extra = (frame['neighbour_reference_coefficient'].to_numpy()-s['mean'])/s['scale'] if s['use_score'] else np.zeros(len(frame))
    context = torch.tensor(np.column_stack([expected, extra]).astype(np.float32))
    torch.testing.assert_close(bundle.context_tensor(frame), context, rtol=0, atol=0)
    z = torch.randn((8, 4), generator=torch.Generator().manual_seed(812))
    with torch.no_grad():
        y, ld = old.forward_physical(z, context)
        yn, ldn = bundle.model.forward_physical(z, context)
        torch.testing.assert_close(y, yn, rtol=0, atol=0)
        torch.testing.assert_close(ld, ldn, rtol=0, atol=0)
        target = (y - torch.tensor(bundle.target_transform.means)) / torch.tensor(bundle.target_transform.scales)
        torch.testing.assert_close(old.log_prob(target, context), bundle.model.log_prob(target, context), rtol=0, atol=0)
    if s['use_score']:
        with pytest.raises(ValueError, match='reference neighbour'):
            bundle.context_tensor(frame.drop(columns='neighbour_reference_coefficient'))
    else:
        torch.testing.assert_close(bundle.context_tensor(frame.drop(columns='neighbour_reference_coefficient')), context, rtol=0, atol=0)
    draws, gaps = bundle.sample_with_shape_log_gap(frame, n_samples=2)
    assert draws.shape == (8, 2, 4) and np.isfinite(gaps).all()
    assert (draws[..., 2:] > 0).all()


def test_inference_requires_matching_model_reference():
    from _script_loader import load_script_module
    m = load_script_module('run_galsbi_inference.py')
    from sbsi.cli import _model_payload
    a = _model_payload(get_model('like-1.0-galsbi'))
    b = _model_payload(get_model('galsbi-multistage'))
    m.validate_reference_identity({'model_identity': a}, a)
    with pytest.raises(ValueError, match='artifact hashes'):
        m.validate_reference_identity({'model_identity': a}, b)
    with pytest.raises(ValueError, match='artifact hashes'):
        m.validate_reference_identity({}, a)
