"""Reuse frozen prediction helpers with separate replication outputs."""
import hashlib
import json
import sys
import types
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / H.name
ORIGINAL = H.parent / 'conditional_cut_probability_20261107'


def load(name, path):
    path = Path(path)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    return mod


prior = load('cutrep_prior_setup', ORIGINAL / 'common.py')
c = prior.c
c.ROOT = ROOT


def manifest():
    return json.loads((H / 'manifest.json').read_text())


c.manifest = manifest


def verify():
    prior.verify()
    for path, expected in json.loads((H / 'source_pins.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    cfg = manifest()
    assert c.sha(ORIGINAL / 'evaluate.py') == cfg['reference_evaluator_sha256']
    for pin in cfg['checkpoints'].values():
        assert c.sha(pin['path']) == pin['sha256']
