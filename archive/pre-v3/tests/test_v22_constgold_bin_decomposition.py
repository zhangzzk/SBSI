import numpy as np
import pandas as pd
import json
import sys

from scripts.analyze_v22_constgold_bin_decomposition import power_summary
from scripts.cleanup_v22_constgold_bin_images import main as cleanup_main
from scripts.merge_v22_constgold_bin_decomposition import main as merge_main
from scripts.prepare_v22_constgold_bin_decomposition import (
    IDENTITY_COLUMNS, assign_treatment, candidate_mask, scene_ownership,
)


def toy_frame() -> pd.DataFrame:
    n = 6
    frame = pd.DataFrame({
        "index": np.arange(n, dtype=np.int64),
        "cata_idx": np.arange(n, dtype=np.int64) + 100,
        "RA": 180.0 + np.array([0, 1, 2, 50, 51, 52]) / 3600.0,
        "DEC": np.zeros(n),
        "g1": np.full(n, 0.02), "g2": np.zeros(n),
        "shear_component_convention": ["reduced_shear"] * n,
        "position_angle": np.zeros(n), "redshift": np.ones(n),
        "Re": np.ones(n), "axis_ratio": np.full(n, 0.8),
        "sersic_n": np.ones(n), "r": np.full(n, 24.0),
    })
    return frame


def test_frozen_q3_candidate_mask_uses_lookup_and_domain():
    frame = toy_frame()
    frame.loc[2, "Re"] = 0.4
    lookup = pd.DataFrame({
        "input_index": np.arange(6),
        "R_blend": [0.11, 0.10, 0.20, 0.31, 0.313, np.nan],
    })
    assert candidate_mask(frame, lookup).tolist() == [True, False, False, True, False, False]


def test_disjoint_scene_ownership_and_treatments_partition_sources():
    frame = toy_frame()
    ownership = scene_ownership(frame, np.array([0, 3]), radius_arcsec=3.0)
    assert len(ownership) == 6
    assert not ownership["source_index"].duplicated().any()
    ownership["source_class"] = "other"
    ownership.loc[ownership["source_index"].isin([1, 4]), "source_class"] = "deployed"
    ownership.loc[ownership["source_index"].isin([0, 3]), "source_class"] = "self"
    local = frame.iloc[ownership["source_row"].to_numpy()].copy().merge(
        ownership[["source_index", "anchor_index", "source_class"]],
        left_on="index", right_on="source_index", validate="one_to_one",
    ).drop(columns="source_index")
    local.attrs["reference_columns"] = frame.columns.tolist()
    selected = {}
    for mode in ("total", "self", "deployed", "other"):
        out = assign_treatment(local, mode, +0.02, 0.02)
        assert out.columns.tolist() == frame.columns.tolist()
        selected[mode] = set(out.loc[out["g1"] != 0, "index"])
    assert selected["total"] == set(range(6))
    assert selected["self"] == {0, 3}
    assert selected["deployed"] == {1, 4}
    assert selected["other"] == {2, 5}
    assert selected["self"] | selected["deployed"] | selected["other"] == selected["total"]


def test_power_summary_scales_from_between_case_sd():
    frame = pd.DataFrame({"case": [0, 1, 2, 3], "x": [-1.5, -0.5, 0.5, 1.5]})
    got = power_summary(frame, "x")
    assert got["current_n_cases"] == 4
    assert np.isclose(got["current_sem"], np.std(frame["x"], ddof=1) / 2)
    assert got["targets"]["0.0067"]["cases_for_80pct_power"] > 4


def test_verified_cleanup_removes_only_images(tmp_path, monkeypatch):
    bases = {}
    for mode in ("total", "self", "deployed", "other"):
        base = tmp_path / mode
        bases[mode] = base
        for sign in (0.02, -0.02):
            case = base / f"case48_{float(sign)}" / "real0"
            images = case / "images"
            images.mkdir(parents=True)
            (images / "image.fits").write_bytes(b"image")
            catalogues = case / "catalogues"
            catalogues.mkdir()
            (catalogues / "keep.txt").write_text("keep")
    truth = tmp_path / "truth.feather"
    pd.DataFrame({"case": [48], "input_index": [7]}).to_feather(truth)
    truth.with_suffix(".json").write_text(json.dumps({"cases": [48], "n_rows": 1}))
    receipt = tmp_path / "receipt.json"
    argv = ["cleanup"]
    for mode, base in bases.items():
        argv.extend(["--base", f"{mode}={base}"])
    argv.extend([
        "--truth", str(truth), "--cases", "48", "--signs", "0.02", "-0.02",
        "--receipt", str(receipt),
    ])
    monkeypatch.setattr(sys, "argv", argv)
    cleanup_main()
    got = json.loads(receipt.read_text())
    assert got["target_count"] == 8
    for base in bases.values():
        for sign in (0.02, -0.02):
            real = base / f"case48_{float(sign)}" / "real0"
            assert not (real / "images").exists()
            assert (real / "catalogues" / "keep.txt").read_text() == "keep"


def test_merge_requires_disjoint_complete_cases(tmp_path, monkeypatch):
    inputs = []
    for case in (40, 41):
        path = tmp_path / f"truth_{case}.feather"
        pd.DataFrame({
            "case": [case], "input_index": [case * 10], "R_total_truth": [0.7],
        }).to_feather(path)
        path.with_suffix(".json").write_text(json.dumps({
            "cases": [case], "n_rows": 1, "component_identity": "identity",
        }))
        inputs.append(path)
    output = tmp_path / "merged.feather"
    monkeypatch.setattr(sys, "argv", [
        "merge", "--input", str(inputs[0]), "--input", str(inputs[1]),
        "--cases", "40", "41", "--output", str(output),
    ])
    merge_main()
    got = pd.read_feather(output)
    assert got["case"].tolist() == [40, 41]
    assert json.loads(output.with_suffix(".json").read_text())["n_cases"] == 2
