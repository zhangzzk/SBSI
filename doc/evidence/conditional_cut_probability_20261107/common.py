"""Isolated trial paths and source-byte loading (no timestamp bytecode cache)."""
import hashlib
import json
import sys
import types
from pathlib import Path

H = Path(__file__).resolve().parent
BASE = H.parent / 'neighbour_moment_guard_20261106'
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / H.name


def load(name, path):
    path = Path(path)
    source = path.read_bytes()
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(source, str(path), 'exec'), mod.__dict__)
    return mod


def manifest():
    return json.loads((H / 'manifest.json').read_text())


def verify():
    pins = dict(manifest()['reference_sha256'])
    pins.update(json.loads((H / 'source_pins.json').read_text()))
    for path, expected in pins.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path


c = load('cutprob_original_helpers', BASE / 'common.py')
c.ROOT = ROOT
c.manifest = manifest


def bound(name):
    mod = load('cutprob_reference_' + name, BASE / (name + '.py'))
    mod.c = c
    mod.HERE = H
    return mod
