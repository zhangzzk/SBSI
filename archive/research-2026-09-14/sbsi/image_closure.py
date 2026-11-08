"""Convert single-leg BlendEMU image measurements into an inference mock.

BlendEMU owns rendering, detection, cross-matching, and shape measurement.
This module only performs the inference-side join of those declared products,
checks their shear and identity, and maps them to the exact measurement-flow
targets.  Pixel blending is already present in the measurements, so image
mocks deliberately carry no catalogue-prior ``scene_row`` or injected
``R_blend`` displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .catalogue_closure import MockCatalogue
from .scene_prior import SHEAR_TRANSFORM


SUPPORTED_IMAGE_TARGETS = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_mag_auto",
    "measured_log_flux_radius",
)


@dataclass(frozen=True)
class ImageCasePaths:
    """The three BlendEMU catalogues needed from one rendered image case."""

    case: int
    input_catalogue: Path
    match_catalogue: Path
    shape_catalogue: Path


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blendemu_case_paths(
    root: str | Path,
    *,
    case: int,
    shear_label: str,
    real: str = "real0",
    tile_name: str = "tile180.0_-0.5",
    shape_name: str | None = None,
) -> ImageCasePaths:
    """Resolve one case without guessing between multiple shape products."""

    case_root = Path(root) / f"case{int(case)}_{shear_label}" / real / "catalogues"
    if shape_name is None:
        candidates = (
            case_root
            / "Shapes"
            / f"shape_catalogue_detect_position_{tile_name}.feather",
            case_root
            / "Shapes"
            / f"shape_catalogue_detect_position_all_{tile_name}.feather",
        )
        present = [path for path in candidates if path.is_file()]
        if len(present) != 1:
            raise FileNotFoundError(
                f"case {case} has {len(present)} recognized shape catalogues; "
                "supply shape_name explicitly"
            )
        shape_path = present[0]
    else:
        shape_path = case_root / "Shapes" / shape_name.format(tile_name=tile_name)
    paths = ImageCasePaths(
        case=int(case),
        input_catalogue=(
            case_root / "input" / f"gals_info_{tile_name}.feather"
        ),
        match_catalogue=(
            case_root / "CrossMatch" / f"{tile_name}_rot0_matched.feather"
        ),
        shape_catalogue=shape_path,
    )
    missing = [str(path) for path in paths.__dict__.values() if isinstance(path, Path) and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"case {case} is missing BlendEMU products: {missing}")
    return paths


def _read_image_case(
    paths: ImageCasePaths,
    *,
    injected_g1: float,
    injected_g2: float,
    primary_mag_bounds: tuple[float, float],
    primary_re_bounds: tuple[float, float],
) -> tuple[pd.DataFrame, dict]:
    input_columns = [
        "index_input",
        "r_input",
        "Re_input",
        "gamma1_input",
        "gamma2_input",
    ]
    shape_columns = ["NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"]
    truth = pd.read_feather(paths.input_catalogue, columns=input_columns)
    match = pd.read_feather(paths.match_catalogue)
    shape = pd.read_feather(paths.shape_catalogue, columns=shape_columns)

    if truth["index_input"].duplicated().any():
        raise ValueError(f"case {paths.case} input catalogue has duplicate index_input")
    if "id_input" not in match or "id_detec" not in match:
        raise KeyError(f"case {paths.case} cross-match lacks id_input/id_detec")
    if "distance_pixel_CM" in match:
        match = match.sort_values("distance_pixel_CM")
    duplicate_matches = int(match["id_input"].duplicated(keep=False).sum())
    match = match.drop_duplicates("id_input", keep="first")
    detection_row = match["id_detec"].to_numpy(np.int64) - 1
    valid_detection = (detection_row >= 0) & (detection_row < len(shape))
    match = match.loc[valid_detection].copy()
    detection_row = detection_row[valid_detection]

    measured = shape.iloc[detection_row].reset_index(drop=True)
    joined = match[["id_input", "id_detec"]].reset_index(drop=True).join(measured)
    joined = joined.merge(
        truth,
        how="inner",
        left_on="id_input",
        right_on="index_input",
        validate="one_to_one",
    )
    expected = np.array([float(injected_g1), float(injected_g2)])
    actual = truth[["gamma1_input", "gamma2_input"]].to_numpy(float)
    shear_ok = np.isclose(actual, expected[None, :], rtol=0, atol=1.0e-14).all(axis=1)
    if not shear_ok.all():
        examples = actual[~shear_ok][:5].tolist()
        raise ValueError(
            f"case {paths.case} input shear does not match {expected.tolist()}; "
            f"examples: {examples}"
        )

    mag_lo, mag_hi = primary_mag_bounds
    re_lo, re_hi = primary_re_bounds
    source_selected = (
        joined["r_input"].between(mag_lo, mag_hi, inclusive="neither")
        & joined["Re_input"].between(re_lo, re_hi, inclusive="neither")
    )
    finite = np.isfinite(
        joined[[*shape_columns]].to_numpy(dtype=float)
    ).all(axis=1)
    positive_radius = joined["FLUX_RADIUS"].to_numpy(float) > 0.0
    # ngmix uses (-1,-1) when fitting fails.  Valid reduced-shear estimates
    # live strictly inside the unit disc, which also rejects that sentinel.
    shape_norm = np.hypot(
        joined["NGMIX_G1"].to_numpy(float),
        joined["NGMIX_G2"].to_numpy(float),
    )
    valid_shape = shape_norm < 1.0
    keep = source_selected.to_numpy() & finite & positive_radius & valid_shape
    selected = joined.loc[keep].copy()
    selected["measured_ngmix_g1"] = selected["NGMIX_G1"].to_numpy(float)
    selected["measured_ngmix_g2"] = selected["NGMIX_G2"].to_numpy(float)
    selected["measured_mag_auto"] = selected["MAG_AUTO"].to_numpy(float)
    selected["measured_log_flux_radius"] = np.log(
        selected["FLUX_RADIUS"].to_numpy(float)
    )
    selected["source_case"] = int(paths.case)
    report = {
        "case": int(paths.case),
        "n_input": int(len(truth)),
        "n_cross_matches": int(len(match)),
        "n_duplicate_match_rows": duplicate_matches,
        "n_source_selected": int(source_selected.sum()),
        "n_invalid_measurements": int((source_selected.to_numpy() & ~(
            finite & positive_radius & valid_shape
        )).sum()),
        "n_retained": int(len(selected)),
        "files": {
            "input": str(paths.input_catalogue),
            "match": str(paths.match_catalogue),
            "shape": str(paths.shape_catalogue),
        },
    }
    return selected, report


def build_image_mock(
    case_paths: Iterable[ImageCasePaths],
    *,
    target_names: Sequence[str],
    injected_g1: float,
    injected_g2: float,
    primary_mag_bounds: tuple[float, float],
    primary_re_bounds: tuple[float, float],
    n_objects: int | None = None,
    sample_seed: int = 0,
) -> tuple[MockCatalogue, dict]:
    """Build a uniformly sampled, image-generated frozen inference mock."""

    target_names = tuple(target_names)
    unsupported = sorted(set(target_names) - set(SUPPORTED_IMAGE_TARGETS))
    if unsupported:
        raise KeyError(f"unsupported image-to-flow targets: {unsupported}")
    if set(target_names) != set(SUPPORTED_IMAGE_TARGETS):
        raise ValueError(
            "the current image adapter requires all four ngmix/magnitude/size targets"
        )
    mag_lo, mag_hi = map(float, primary_mag_bounds)
    re_lo, re_hi = map(float, primary_re_bounds)
    if not (mag_lo < mag_hi and re_lo < re_hi):
        raise ValueError("primary magnitude and size bounds must be ordered")
    frames = []
    reports = []
    for paths in case_paths:
        frame, report = _read_image_case(
            paths,
            injected_g1=injected_g1,
            injected_g2=injected_g2,
            primary_mag_bounds=(mag_lo, mag_hi),
            primary_re_bounds=(re_lo, re_hi),
        )
        frames.append(frame)
        reports.append(report)
    if not frames:
        raise ValueError("at least one image case is required")
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.duplicated(["source_case", "id_input"]).any():
        raise RuntimeError("image cases contain duplicate (case, input) objects")
    available = len(pooled)
    if n_objects is None:
        n_objects = available
    if n_objects <= 0 or n_objects > available:
        raise ValueError(
            f"requested n_objects={n_objects}, but {available} valid objects are available"
        )
    rng = np.random.default_rng(sample_seed)
    rows = rng.choice(available, size=int(n_objects), replace=False)
    pooled = pooled.iloc[rows].reset_index(drop=True)
    measurements = pooled.loc[:, list(target_names)].copy()
    truth = pd.DataFrame(
        {
            "mock_kind": "image",
            "shear_transform": SHEAR_TRANSFORM,
            "source_case": pooled["source_case"].to_numpy(np.int64),
            "source_input_index": pooled["id_input"].to_numpy(np.int64),
            "source_detection_id": pooled["id_detec"].to_numpy(np.int64),
            "injected_g1": float(injected_g1),
            "injected_g2": float(injected_g2),
        }
    )
    mock = MockCatalogue(measurements=measurements, truth=truth)
    report = {
        "mock_kind": mock.kind,
        "shear_transform": SHEAR_TRANSFORM,
        "injected_g1": float(injected_g1),
        "injected_g2": float(injected_g2),
        "target_names": list(target_names),
        "primary_domain": {"mag": [mag_lo, mag_hi], "Re": [re_lo, re_hi]},
        "sample_seed": int(sample_seed),
        "n_available": int(available),
        "n_selected": int(len(mock.measurements)),
        "cases": reports,
    }
    return mock, report


__all__ = [
    "ImageCasePaths",
    "SUPPORTED_IMAGE_TARGETS",
    "blendemu_case_paths",
    "build_image_mock",
    "file_sha256",
]
