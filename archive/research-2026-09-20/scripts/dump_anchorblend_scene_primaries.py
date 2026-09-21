#!/usr/bin/env python3
"""Dump per-primary scene blending response and measured shape, anchor by anchor.

The primary-unsheared anchor simulations hold the anchor galaxy at zero shear
and shear every neighbour coherently by +/-h, so the anchor's measured response
*is* its blending response -- the flow never enters it.  The production run
``anchorblend_g002_measured_cuts_v1`` pooled that response over the whole
selected population and reported one number.  This writes the same quantities
one row per primary instead, so the emulator's accuracy can afterwards be read
as a function of how crowded the primary is, with no re-simulation and no flow.

Selection, cuts, weighting and the emulator call are reproduced exactly from
that run's contract.  Only its ``manifest.json`` is read -- the emulator paths,
their hashes and the rendering conditions -- never its code.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/project/ls-gruen/users/zekang.zhang")
H = 0.02
MAG_MAX = 25.8
RADIUS_MIN_PIXELS = 3.0
FEATURES = (
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
)
INTRINSIC_COLUMNS = (
    "index",
    "RA",
    "DEC",
    "redshift",
    "Re",
    "axis_ratio",
    "position_angle",
    "sersic_n",
    "r",
    "e1_rot0",
    "e2_rot0",
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def simulation_roots(case: int) -> tuple[Path, Path]:
    block = case // 100 * 100
    return (
        PROJECT_ROOT / f"lsst_sims_fs2_25876_anchorblend_g002_c{block}-{block + 99}",
        PROJECT_ROOT
        / f"lsst_sims_fs2_25876_anchorblend_g002_v2complement_c{block}-{block + 99}",
    )


def catalogue_paths(root: Path, case: int, sign: float) -> dict[str, Path]:
    cat = root / f"case{case}_{sign}" / "real0" / "catalogues"
    return {
        "truth": cat / "input" / "gals_info_tile180.0_-0.5.feather",
        "cross": cat / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather",
        "shape": cat / "Shapes" / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather",
    }


def read_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(path).rename(columns=lambda name: name.replace("_input", ""))
    missing = sorted({*INTRINSIC_COLUMNS, "gamma1", "gamma2"} - set(frame.columns))
    if missing:
        raise KeyError(f"{path} lacks truth fields {missing}")
    if frame["index"].duplicated().any():
        raise ValueError(f"{path} has duplicate truth identities")
    return frame


def verify_scene_pair(plus: pd.DataFrame, minus: pd.DataFrame, anchor_ids: np.ndarray) -> None:
    """The instrument only means what it means if the shear pattern is exact."""
    if len(plus) != len(minus):
        raise RuntimeError("the two truth scenes have different row counts")
    for name in INTRINSIC_COLUMNS:
        if not np.array_equal(plus[name].to_numpy(), minus[name].to_numpy()):
            raise RuntimeError(f"the two truth scenes differ in {name}")
    anchor = plus["index"].isin(anchor_ids).to_numpy()
    if not np.array_equal(anchor, minus["index"].isin(anchor_ids).to_numpy()):
        raise RuntimeError("the two truth scenes have different anchor membership")
    for leg, frame in (("plus", plus), ("minus", minus)):
        if not np.allclose(frame.loc[anchor, ["gamma1", "gamma2"]], 0.0, atol=1e-12):
            raise RuntimeError(f"{leg}-leg anchor primary is not unsheared")
        if not np.allclose(frame.loc[~anchor, "gamma2"], 0.0, atol=1e-12):
            raise RuntimeError(f"{leg}-leg neighbours have nonzero g2")
    if not np.allclose(plus.loc[~anchor, "gamma1"], +H, atol=1e-12):
        raise RuntimeError("plus-leg neighbours are not coherently sheared by +h")
    if not np.allclose(minus.loc[~anchor, "gamma1"], -H, atol=1e-12):
        raise RuntimeError("minus-leg neighbours are not coherently sheared by -h")


def measured_leg(paths: dict[str, Path], active_ids: np.ndarray) -> pd.DataFrame:
    """One sheared leg's selected anchors, with the run's measured cuts applied."""
    cross = pd.read_feather(paths["cross"], columns=["id_detec", "id_input"])
    shape = pd.read_feather(
        paths["shape"],
        columns=["NUMBER", "NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"],
    )
    if cross["id_input"].duplicated().any() or cross["id_detec"].duplicated().any():
        raise ValueError(f"{paths['cross']} is not one-to-one")
    if shape["NUMBER"].duplicated().any():
        raise ValueError(f"{paths['shape']} has duplicate detection identities")
    joined = cross.merge(
        shape,
        left_on="id_detec",
        right_on="NUMBER",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    active = joined.loc[joined["id_input"].isin(active_ids)]
    values = active[["NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"]].to_numpy(float)
    finite_shape = np.isfinite(values[:, :2]).all(axis=1)
    valid_shape = finite_shape & (np.square(values[:, :2]).sum(axis=1) < 1.0)
    selected = (
        valid_shape
        & np.isfinite(values[:, 2:]).all(axis=1)
        & (values[:, 2] < MAG_MAX)
        & (values[:, 3] > RADIUS_MIN_PIXELS)
    )
    output = pd.DataFrame(
        {
            "input_index": active.loc[selected, "id_input"].to_numpy(np.int64),
            "e1": values[selected, 0],
            "e2": values[selected, 1],
        }
    ).sort_values("input_index", kind="stable", ignore_index=True)
    if output["input_index"].duplicated().any():
        raise RuntimeError("selected measured population has duplicate identities")
    return output


def predict_primaries(predictor, boundary, frame: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Per-primary signed and absolute blending response over the whole scene.

    ``R_scene`` is the signed sum the production run used.  ``R_scene_abs`` is
    the sum of per-pair magnitudes: the signed sum reaches zero by cancellation
    rather than by blending being absent, so only the absolute sum orders
    primaries by how much blending they actually carry.
    """
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if not len(ids):
        raise RuntimeError("selected population is empty")
    primary = frame.loc[frame["index"].isin(ids)]
    if len(primary) != len(ids):
        raise RuntimeError(
            f"truth scene misses {len(ids) - len(primary)} selected emulator primaries"
        )
    prediction = predictor.predict_response(primary, frame)
    key = next(
        (n for n in prediction.columns if n.startswith("index") and n.endswith("_p")), None
    )
    if key is None:
        raise RuntimeError("emulator output lacks a primary identity column")
    if not np.isin(prediction[key].to_numpy(np.int64), ids).all():
        raise RuntimeError("emulator returned a non-selected primary identity")
    outside = np.zeros(len(prediction), dtype=bool)
    for index, name in enumerate(FEATURES):
        values = prediction[name].to_numpy(float)
        finite = np.isfinite(values)
        outside |= ~finite | (finite & (values < boundary[index, 0]))
        outside |= finite & (values > boundary[index, 1])
    response = prediction["response"].to_numpy(np.float64)
    grouped = pd.DataFrame(
        {
            "input_index": prediction[key].to_numpy(np.int64),
            "R_scene": response,
            "R_scene_abs": np.abs(response),
            "pairs": 1,
            "pairs_outside_training": outside.astype(np.int64),
        }
    ).groupby("input_index", sort=False).sum()
    filled = grouped.reindex(ids, fill_value=0)
    if not np.isfinite(filled[["R_scene", "R_scene_abs"]].to_numpy()).all():
        raise RuntimeError("emulator produced a nonfinite scene response")
    return filled.reset_index(names="input_index")


def load_predictor(manifest: dict, device: str):
    from sbsi.models import ModelPaths, load_emulator

    metadata = Path(manifest["emulator_metadata"])
    model = Path(manifest["emulator_model"])
    paths = ModelPaths(
        flow_checkpoints=(metadata,),
        emulator_model=model,
        emulator_metadata=metadata,
        emulator_sha256=manifest["static_sha256"][str(model)],
        emulator_metadata_sha256=manifest["static_sha256"][str(metadata)],
    )
    paths.validate()
    return load_emulator(paths, conditions=manifest["conditions"], device=device)


def stratum_weight(case: int, stratum: int) -> float:
    """Inverse-sampling weight carrying each anchor stratum back to the V2 parent."""
    roots = simulation_roots(case)
    complement = pd.read_feather(roots[1] / f"anchors_case{case}.feather")
    for name in ("n_parent_original_domain", "n_parent_complement", "n_parent_v2"):
        if complement[name].nunique() != 1:
            raise RuntimeError(f"case {case} complement has variable {name}")
    n_original = int(complement["n_parent_original_domain"].iloc[0])
    n_complement = int(complement["n_parent_complement"].iloc[0])
    if n_original + n_complement != int(complement["n_parent_v2"].iloc[0]):
        raise RuntimeError("stratum parent counts do not sum to V2 parent")
    if stratum == 1:
        weight = n_complement / len(complement)
        if not np.allclose(complement["inverse_sampling_weight"], weight):
            raise RuntimeError("stored complement weight differs from reconstructed weight")
        return float(weight)
    original = pd.read_feather(roots[0] / f"anchors_case{case}.feather")
    return float(n_original / len(original))


def case_stratum(predictor, boundary, case: int, stratum: int) -> pd.DataFrame:
    """One (case, stratum) block, one row per primary selected in either leg."""
    root = simulation_roots(case)[stratum]
    anchors = pd.read_feather(root / f"anchors_case{case}.feather")
    if anchors["index"].duplicated().any():
        raise ValueError(f"case {case} stratum {stratum} has duplicate anchor identities")
    if not (anchors["case"] == case).all():
        raise RuntimeError("anchor case labels are inconsistent")

    legs = {sign: catalogue_paths(root, case, sign) for sign in (+H, -H)}
    plus_truth = read_truth(legs[+H]["truth"])
    minus_truth = read_truth(legs[-H]["truth"])
    anchor_ids = anchors["index"].to_numpy(np.int64)
    verify_scene_pair(plus_truth, minus_truth, anchor_ids)

    rendered = np.intersect1d(anchor_ids, plus_truth["index"].to_numpy(np.int64))
    plus = measured_leg(legs[+H], rendered)
    minus = measured_leg(legs[-H], rendered)
    scored = np.union1d(plus["input_index"], minus["input_index"])
    if not len(plus) or not len(minus):
        raise RuntimeError(f"case {case} stratum {stratum} has an empty selected leg")

    rows = predict_primaries(predictor, boundary, plus_truth, scored)
    #  The truth is identical between legs, so the primary's own magnitude and
    #  size can be carried along for a magnitude-crossed read later.
    carried = ("r", "Re", "sersic_n")
    truth = plus_truth.set_index("index")[list(carried)].reindex(rows["input_index"])
    if truth.isna().to_numpy().any():
        raise RuntimeError("a scored primary is missing from the truth scene")
    for name in carried:
        rows[f"{name}_input_p"] = truth[name].to_numpy(np.float64)
    for name, leg in (("plus", plus), ("minus", minus)):
        aligned = leg.set_index("input_index").reindex(rows["input_index"])
        rows[f"e1_{name}"] = aligned["e1"].to_numpy(np.float64)
        rows[f"e2_{name}"] = aligned["e2"].to_numpy(np.float64)
    rows["matched"] = np.isfinite(rows["e1_plus"]) & np.isfinite(rows["e1_minus"])
    rows["case"] = np.int32(case)
    rows["stratum"] = np.int8(stratum)
    rows["weight"] = stratum_weight(case, stratum)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="manifest.json of the production anchorblend run")
    parser.add_argument("--case-start", type=int, required=True)
    parser.add_argument("--case-stop", type=int, required=True, help="exclusive")
    parser.add_argument("--output", type=Path, required=True, help="directory for parts")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    for path, expected in manifest["static_sha256"].items():
        found = file_sha256(path)
        if found != expected:
            raise RuntimeError(f"{path} hashes {found}, manifest pins {expected}")
    boundary = np.asarray(manifest["feature_boundary"], dtype=float)
    predictor = load_predictor(manifest, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for case in range(args.case_start, args.case_stop):
        for stratum in (0, 1):
            rows = case_stratum(predictor, boundary, case, stratum)
            destination = args.output / f"case{case}_stratum{stratum}.feather"
            rows.to_feather(destination)
            print(
                f"ANCHORBLEND_PRIMARIES case={case} stratum={stratum} "
                f"rows={len(rows)} matched={int(rows['matched'].sum())} "
                f"meanR={rows['R_scene'].mean():+.6f} "
                f"oob={int(rows['pairs_outside_training'].sum())}",
                flush=True,
            )


if __name__ == "__main__":
    main()
