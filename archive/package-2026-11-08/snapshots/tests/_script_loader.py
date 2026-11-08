"""Small helper for testing repository scripts as importable modules."""

import importlib.util
from functools import lru_cache
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_script_module(filename: str):
    path = REPOSITORY_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"sbsi_test_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test target {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
