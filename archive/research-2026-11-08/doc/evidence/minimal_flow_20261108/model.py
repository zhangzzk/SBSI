"""Smaller, normalized factorized flows with unchanged physical transforms."""
import copy
import types
from pathlib import Path
import torch
from torch import nn

H = Path(__file__).resolve().parent
c = types.ModuleType('minimal_model_common')
c.__file__ = str(H / 'common.py')
exec(compile((H / 'common.py').read_bytes(), c.__file__, 'exec'), c.__dict__)
FORMAT = 'sbsi_experimental_minimal_factorized_v1'

def trim(model, depth, score):
    assert depth in (3, 5)
    width = model.context_dim
    for flow in (model.base.base.photometry, model.base.base.shape):
        flow.layers = nn.ModuleList(list(flow.layers)[:depth])
        if not score:
            for layer in flow.layers:
                # Source-isolated imports can create distinct class identities.
                # Check the frozen wrapper's actual structure and input position.
                assert type(layer.net[0]).__name__ == 'ExtraInputLinear'
                assert isinstance(layer.net[0].original, nn.Linear)
                assert layer.net[0].index == 2 + width - 1
                layer.net[0] = layer.net[0].original
            flow.context_dim -= 1
    if not score:
        model.context_dim = width - 1
    return model

def initialize(teacher, arm):
    return trim(copy.deepcopy(teacher), arm['depth'], arm['score'])

def contexts(x, arm):
    return x if arm['score'] else x[..., :-1]

def save(path, model, condition, target, teacher_config, arm, metadata):
    torch.save(dict(format=FORMAT, state_dict=model.state_dict(),
                    condition=condition.to_state(), target=target.to_state(),
                    teacher_config=teacher_config, arm=arm, metadata=metadata), path)

def load(path, device='cuda'):
    d = torch.load(path, map_location=device, weights_only=False)
    assert d['format'] == FORMAT
    condition = c.old.Condition.from_state(d['condition'])
    target = c.old.TargetStandardizer.from_state(d['target'])
    model = c.old.augment(c.old.c.fm.FrameFlow(d['teacher_config'], condition.base, 'cartesian')).to(device)
    trim(model, d['arm']['depth'], d['arm']['score'])
    model.load_state_dict(d['state_dict'])
    model.eval()
    # Input-free deployment uses the original preprocessor: no score needed.
    return model, condition if d['arm']['score'] else condition.base, target, d
