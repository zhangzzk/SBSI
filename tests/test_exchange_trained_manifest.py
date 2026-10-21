"""Guard the prepared trained-model probe's frozen-source preflight."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('exchange_trained_probe',
    Path(__file__).resolve().parents[1]/'scripts/benchmark_exchange_trained.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_manifest_rejects_changed_or_incomplete_source(tmp_path):
    path = tmp_path/'a.py'
    path.write_text('original')
    manifest = tmp_path/'SHA256SUMS'
    manifest.write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  a.py\n')
    with pytest.raises(ValueError, match='missing'):
        probe.check_manifest(tmp_path, manifest)
    path.write_text('changed')
    with pytest.raises(ValueError, match='mismatch'):
        probe.check_manifest(tmp_path, manifest)
