#!/usr/bin/env python
"""Validate and optionally prune one frozen minimal-R_blend simulation block."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PRODUCTION_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_rblend_fresh_acceptance_v1/production"
)
PROTOCOL = "minimal-rblend-fresh-acceptance-v1"
CASE_FIRST = 1001
BLOCK_SIZE = 20
BLOCK_COUNT = 38
RBLEND_MAG_OPEN_INTERVAL = (18.0, 25.8)
RBLEND_RE_OPEN_INTERVAL = (0.37, 1.5)
GROUPS = {
    "g0": (("0.0", 0.0, 0.0, "g0"),),
    "g1_h002": (
        ("-0.02", -0.02, 0.0, "g1_m02"),
        ("0.02", 0.02, 0.0, "g1_p02"),
    ),
    "g1_h004": (
        ("-0.04", -0.04, 0.0, "g1_m04"),
        ("0.04", 0.04, 0.0, "g1_p04"),
    ),
    "g2_h002": (
        ("-0.02", 0.0, -0.02, "g2_m02"),
        ("0.02", 0.0, 0.02, "g2_p02"),
    ),
    "g2_h004": (
        ("-0.04", 0.0, -0.04, "g2_m04"),
        ("0.04", 0.0, 0.04, "g2_p04"),
    ),
}
FIXED_TRUTH_COLUMNS = (
    "index_input",
    "RA_input",
    "DEC_input",
    "redshift_input",
    "Re_input",
    "axis_ratio_input",
    "position_angle_input",
    "sersic_n_input",
    "r_input",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing, empty, or linked acceptance file: {path}")
    return path


def _assert_under(path: Path, parent: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing linked cleanup target: {path}")
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"cleanup target escapes block root: {path}") from exc


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _block_cases(block_index: int) -> tuple[int, ...]:
    if not 0 <= block_index < BLOCK_COUNT:
        raise ValueError(f"block index must be in [0,{BLOCK_COUNT - 1}]")
    first = CASE_FIRST + BLOCK_SIZE * block_index
    return tuple(range(first, first + BLOCK_SIZE))


def _block_root(block_index: int) -> Path:
    root = PRODUCTION_ROOT / f"block_{block_index:02d}"
    if PRODUCTION_ROOT.is_symlink() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"invalid production block root: {root}")
    if root.resolve(strict=True).parent != PRODUCTION_ROOT.resolve(strict=True):
        raise RuntimeError(f"block root does not resolve directly below production: {root}")
    return root


def validate_block(block_index: int) -> tuple[Path, dict, list[Path]]:
    """Return the validated root, receipt payload, and exact prune targets."""
    cases = _block_cases(block_index)
    root = _block_root(block_index)
    expected_dirs = {
        group: {
            f"case{case}_{token}"
            for case in cases
            for token, _, _, _ in legs
        }
        for group, legs in GROUPS.items()
    }
    for group, expected in expected_dirs.items():
        group_root = root / group
        if group_root.is_symlink() or not group_root.is_dir():
            raise RuntimeError(f"missing or linked group root: {group_root}")
        actual = {
            item.name
            for item in group_root.iterdir()
            if item.is_dir()
            and item.name.startswith("case")
            and not item.name.endswith("_tmp")
        }
        if actual != expected:
            raise RuntimeError(
                f"unexpected case directories in {group_root}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    all_noise_hashes: set[str] = set()
    expected_rblend_ids: dict[int, np.ndarray] = {}
    outputs: dict[str, dict[str, dict]] = {}
    prune_targets: list[Path] = []
    for case in cases:
        reference: pd.DataFrame | None = None
        case_outputs: dict[str, dict] = {}
        for group, legs in GROUPS.items():
            for token, g1, g2, leg_name in legs:
                case_root = root / group / f"case{case}_{token}"
                catalogue = case_root / "real0" / "catalogues"
                truth_path = _required_file(
                    catalogue / "input" / "gals_info_tile180.0_-0.5.feather"
                )
                noise_path = _required_file(
                    catalogue / "input" / "noise_info_tile180.0_-0.5.csv"
                )
                shape_path = _required_file(
                    catalogue
                    / "Shapes"
                    / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
                )
                cross_path = _required_file(
                    catalogue / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather"
                )
                truth = pd.read_feather(
                    truth_path,
                    columns=[*FIXED_TRUTH_COLUMNS, "gamma1_input", "gamma2_input"],
                )
                if truth.empty or truth["index_input"].duplicated().any():
                    raise RuntimeError(f"invalid truth identities: {truth_path}")
                if not np.allclose(truth["gamma1_input"], g1, rtol=0, atol=1e-15):
                    raise RuntimeError(f"wrong g1 values: {truth_path}")
                if not np.allclose(truth["gamma2_input"], g2, rtol=0, atol=1e-15):
                    raise RuntimeError(f"wrong g2 values: {truth_path}")
                fixed = truth.loc[:, FIXED_TRUTH_COLUMNS].reset_index(drop=True)
                if reference is None:
                    reference = fixed
                    mag_min, mag_max = RBLEND_MAG_OPEN_INTERVAL
                    re_min, re_max = RBLEND_RE_OPEN_INTERVAL
                    in_support = (
                        np.isfinite(fixed[["r_input", "Re_input"]]).all(axis=1)
                        & (fixed["r_input"] > mag_min)
                        & (fixed["r_input"] < mag_max)
                        & (fixed["Re_input"] > re_min)
                        & (fixed["Re_input"] < re_max)
                    )
                    expected_rblend_ids[case] = np.sort(
                        fixed.loc[in_support, "index_input"].to_numpy(np.int64)
                    )
                else:
                    pd.testing.assert_frame_equal(fixed, reference, check_exact=True)

                shape = pd.read_feather(shape_path, columns=["NUMBER"])
                cross = pd.read_feather(cross_path, columns=["id_detec", "id_input"])
                if shape.empty or shape["NUMBER"].duplicated().any():
                    raise RuntimeError(f"invalid shape identities: {shape_path}")
                if (
                    cross.empty
                    or cross["id_detec"].duplicated().any()
                    or cross["id_input"].duplicated().any()
                ):
                    raise RuntimeError(f"invalid cross-match identities: {cross_path}")
                shape_ids = shape["NUMBER"].to_numpy()
                cross_ids = cross["id_detec"].to_numpy()
                matched = np.intersect1d(shape_ids, cross_ids, assume_unique=True).size
                noise_hash = _sha256(noise_path)
                all_noise_hashes.add(noise_hash)
                case_outputs[leg_name] = {
                    "truth_rows": len(truth),
                    "shape_rows": len(shape),
                    "crossmatch_rows": len(cross),
                    "crossmatch_rows_without_shape": int(len(cross) - matched),
                    "shape_rows_without_crossmatch": int(len(shape) - matched),
                    "sha256": {
                        "truth": _sha256(truth_path),
                        "noise": noise_hash,
                        "shape": _sha256(shape_path),
                        "crossmatch": _sha256(cross_path),
                    },
                }
                image_root = case_root / "real0" / "images"
                if not image_root.is_dir():
                    raise RuntimeError(f"missing image cleanup target: {image_root}")
                _assert_under(image_root, root)
                prune_targets.append(image_root)
                temporary_root = root / group / f"case{case}_{token}_tmp"
                if temporary_root.exists():
                    if not temporary_root.is_dir():
                        raise RuntimeError(f"invalid temporary cleanup target: {temporary_root}")
                    _assert_under(temporary_root, root)
                    prune_targets.append(temporary_root)
        outputs[str(case)] = case_outputs

    if len(all_noise_hashes) != 1:
        raise RuntimeError("noise metadata differ across cases or shear legs")
    rblend_path = _required_file(root / "rblend.feather")
    rblend = pd.read_feather(rblend_path, columns=["case", "input_index", "R_blend"])
    if (
        rblend.empty
        or rblend.duplicated(["case", "input_index"]).any()
        or not np.isfinite(rblend["R_blend"]).all()
        or set(rblend["case"].unique()) != set(cases)
        or (rblend.groupby("case").size() == 0).any()
    ):
        raise RuntimeError(f"invalid block R_blend lookup: {rblend_path}")
    rblend_rows_per_case: dict[str, int] = {}
    for case in cases:
        actual_ids = np.sort(
            rblend.loc[rblend["case"] == case, "input_index"].to_numpy(np.int64)
        )
        expected_ids = expected_rblend_ids[case]
        if not np.array_equal(actual_ids, expected_ids):
            missing = np.setdiff1d(expected_ids, actual_ids, assume_unique=True)
            extra = np.setdiff1d(actual_ids, expected_ids, assume_unique=True)
            raise RuntimeError(
                f"R_blend keys do not equal the declared truth support for case {case}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        rblend_rows_per_case[str(case)] = len(actual_ids)

    receipt = {
        "status": "validated_before_prune",
        "protocol": PROTOCOL,
        "block_index": block_index,
        "case_first": cases[0],
        "case_last": cases[-1],
        "case_count": len(cases),
        "leg_count": sum(len(legs) for legs in GROUPS.values()),
        "fixed_truth_columns_identical_within_case": list(FIXED_TRUTH_COLUMNS),
        "common_noise_metadata_sha256": next(iter(all_noise_hashes)),
        "rblend": {
            "rows": len(rblend),
            "sha256": _sha256(rblend_path),
            "truth_support": {
                "r_input_p_open_interval": list(RBLEND_MAG_OPEN_INTERVAL),
                "Re_input_p_open_interval_arcsec": list(RBLEND_RE_OPEN_INTERVAL),
            },
            "rows_per_case": rblend_rows_per_case,
        },
        "outputs": outputs,
    }
    return root, receipt, prune_targets


def prune_validated_targets(root: Path, targets: list[Path]) -> tuple[int, list[str]]:
    deleted_bytes = 0
    relative_targets = []
    for target in targets:
        _assert_under(target, root)
        deleted_bytes += _tree_bytes(target)
        relative_targets.append(str(target.relative_to(root)))
    for target in targets:
        shutil.rmtree(target)
    if any(target.exists() for target in targets):
        raise RuntimeError("one or more exact cleanup targets survived pruning")
    return deleted_bytes, relative_targets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument(
        "--prune-images",
        action="store_true",
        help="after complete validation, remove only exact per-leg images and *_tmp trees",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root, receipt, targets = validate_block(args.block_index)
    if not args.prune_images:
        print(json.dumps({**receipt, "prune_target_count": len(targets)}, indent=2))
        return
    validation_path = root / "block_validation_preprune.json"
    final_path = root / "block_receipt.json"
    if validation_path.exists() or final_path.exists():
        raise RuntimeError("refusing to overwrite an existing block receipt")
    validation_path.write_text(json.dumps(receipt, indent=2) + "\n")
    deleted_bytes, relative_targets = prune_validated_targets(root, targets)
    receipt.update(
        status="validated_and_pruned",
        deleted_bytes=deleted_bytes,
        deleted_targets=relative_targets,
        preprune_receipt_sha256=_sha256(validation_path),
    )
    final_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
