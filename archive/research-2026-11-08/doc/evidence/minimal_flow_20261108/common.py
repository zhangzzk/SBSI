"""Isolated compression experiment; execute pinned sources without stale bytecode."""
import hashlib
import json
import sys
import types
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / H.name

def load(name, path):
    path = Path(path)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    return mod

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def config():
    return json.loads((H / 'manifest.json').read_text())

def verify():
    for path, expected in json.loads((H / 'source_pins.json').read_text()).items():
        assert sha(path) == expected, path

old = load('minimal_teacher_definition', H.parent / 'neighbour_conditioned_flow_20261106/model.py')
portable = load('minimal_portable', H.parent / 'alignment_target9_20261106/portable_draws.py')
