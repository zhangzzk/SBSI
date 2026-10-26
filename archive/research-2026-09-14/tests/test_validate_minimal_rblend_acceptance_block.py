from pathlib import Path

import pandas as pd

import scripts.validate_minimal_rblend_acceptance_block as guard


def _write_leg(root: Path, group: str, token: str, g1: float, g2: float) -> None:
    case_root = root / group / f"case1000_{token}"
    catalogue = case_root / "real0" / "catalogues"
    for name in ("input", "Shapes", "CrossMatch"):
        (catalogue / name).mkdir(parents=True, exist_ok=True)
    truth = pd.DataFrame(
        {
            "index_input": [101, 102],
            "RA_input": [180.0, 180.1],
            "DEC_input": [0.0, 0.1],
            "redshift_input": [0.4, 0.8],
            "Re_input": [0.4, 0.7],
            "axis_ratio_input": [0.8, 0.9],
            "position_angle_input": [0.1, 0.2],
            "sersic_n_input": [1.0, 2.0],
            "r_input": [23.0, 24.0],
            "gamma1_input": [g1, g1],
            "gamma2_input": [g2, g2],
        }
    )
    truth.to_feather(catalogue / "input" / "gals_info_tile180.0_-0.5.feather")
    (catalogue / "input" / "noise_info_tile180.0_-0.5.csv").write_text(
        "rms,seeing\n0.312,0.73\n"
    )
    pd.DataFrame({"NUMBER": [1, 2]}).to_feather(
        catalogue
        / "Shapes"
        / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    )
    pd.DataFrame({"id_detec": [1], "id_input": [101]}).to_feather(
        catalogue / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather"
    )
    images = case_root / "real0" / "images"
    images.mkdir(parents=True)
    (images / "image.fits").write_bytes(b"image")
    temporary = root / group / f"case1000_{token}_tmp"
    temporary.mkdir()
    (temporary / "scratch.txt").write_text("scratch")


def test_guard_validates_every_leg_before_exact_pruning(tmp_path, monkeypatch):
    production = tmp_path / "production"
    block = production / "block_00"
    monkeypatch.setattr(guard, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(guard, "CASE_FIRST", 1000)
    monkeypatch.setattr(guard, "BLOCK_SIZE", 1)
    monkeypatch.setattr(guard, "BLOCK_COUNT", 1)
    for group, legs in guard.GROUPS.items():
        for token, g1, g2, _ in legs:
            _write_leg(block, group, token, g1, g2)
    pd.DataFrame(
        {"case": [1000, 1000], "input_index": [101, 102], "R_blend": [0.1, 0.2]}
    ).to_feather(block / "rblend.feather")

    root, receipt, targets = guard.validate_block(0)

    assert receipt["status"] == "validated_before_prune"
    assert receipt["leg_count"] == 9
    assert receipt["rblend"]["truth_support"] == {
        "r_input_p_open_interval": [18.0, 25.8],
        "Re_input_p_open_interval_arcsec": [0.37, 1.5],
    }
    assert receipt["rblend"]["rows_per_case"] == {"1000": 2}
    assert len(receipt["outputs"]["1000"]) == 9
    assert len(targets) == 18
    deleted_bytes, deleted = guard.prune_validated_targets(root, targets)
    assert deleted_bytes > 0
    assert len(deleted) == 18
    assert not any(path.exists() for path in targets)
    assert (
        block
        / "g1_h002"
        / "case1000_0.02"
        / "real0"
        / "catalogues"
        / "input"
        / "gals_info_tile180.0_-0.5.feather"
    ).is_file()


def test_guard_rejects_rblend_keys_from_narrower_radius_support(
    tmp_path, monkeypatch
):
    production = tmp_path / "production"
    block = production / "block_00"
    monkeypatch.setattr(guard, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(guard, "CASE_FIRST", 1000)
    monkeypatch.setattr(guard, "BLOCK_SIZE", 1)
    monkeypatch.setattr(guard, "BLOCK_COUNT", 1)
    for group, legs in guard.GROUPS.items():
        for token, g1, g2, _ in legs:
            _write_leg(block, group, token, g1, g2)
    pd.DataFrame(
        {"case": [1000], "input_index": [102], "R_blend": [0.2]}
    ).to_feather(block / "rblend.feather")

    try:
        guard.validate_block(0)
    except RuntimeError as error:
        assert "missing=1, extra=0" in str(error)
    else:
        raise AssertionError("narrower R_blend support was accepted")
