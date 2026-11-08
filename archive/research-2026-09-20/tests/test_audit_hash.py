import hashlib

from scripts.train_fixed_g0_flow import sha256


def test_report_checksums_accept_serialized_paths(tmp_path):
    artifact = tmp_path / 'measurement.feather'
    contents = b'fixed artifact contents\x00\xff'
    artifact.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()
    assert sha256(artifact) == expected
    assert sha256(str(artifact)) == expected
