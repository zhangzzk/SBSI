from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876")
DOMAIN_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/domain"
)
MODEL_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/"
    "response_profile_predictions_c0_19_v2"
)

CASES = tuple(range(20))
PIXEL_SCALE = 0.2
RESPONSE_SHEAR = 0.2
RESPONSE_BINS = 24
RESPONSE_AXES = (
    ("g0_MAG_AUTO", r"$g=0$ measured MAG_AUTO", False),
    ("g0_FLUX_RADIUS_arcsec", r"$g=0$ measured FLUX_RADIUS (arcsec)", True),
    ("true_r_magnitude", r"True $r$ magnitude", False),
    ("true_Re_semimajor_arcsec", r"True intrinsic semi-major $R_e$ (arcsec)", True),
)


def load_raw():
    truth_columns = [
        "index_input",
        "Re_input",
        "r_input",
        "e1_input_rot0",
        "e2_input_rot0",
    ]
    measured_columns = ["NUMBER", "MAG_AUTO", "FLUX_RADIUS", "NGMIX_G1", "NGMIX_G2"]
    frames = []
    audits = []

    for case in CASES:
        catalogue = DATA_ROOT / f"case{case}_0.0/real0/catalogues"
        truth = pd.read_feather(
            catalogue / "input/gals_info_tile180.0_-0.5.feather",
            columns=truth_columns,
        )
        cross = pd.read_feather(
            catalogue / "CrossMatch/tile180.0_-0.5_rot0_matched.feather",
            columns=["id_detec", "id_input"],
        )
        first = pd.read_feather(
            catalogue / "Shapes/shape_catalogue_detect_position_tile180.0_-0.5.feather",
            columns=measured_columns,
        )
        second = pd.read_feather(
            catalogue
            / "Shapes/shape_catalogue_detect_position_secondaries_tile180.0_-0.5.feather",
            columns=measured_columns,
        )

        if not cross["id_detec"].is_unique or not cross["id_input"].is_unique:
            raise ValueError(f"case {case}: crossmatch is not one-to-one")

        shapes = first.merge(
            second,
            on="NUMBER",
            how="outer",
            suffixes=("_first", "_second"),
            validate="one_to_one",
            indicator=True,
        )
        if not (shapes["_merge"] == "both").all():
            raise ValueError(f"case {case}: target-half shape IDs differ")

        def valid_half(suffix):
            values = shapes[
                [
                    f"MAG_AUTO_{suffix}",
                    f"FLUX_RADIUS_{suffix}",
                    f"NGMIX_G1_{suffix}",
                    f"NGMIX_G2_{suffix}",
                ]
            ].to_numpy(float)
            return (
                np.isfinite(values).all(axis=1)
                & (values[:, 1] > 0)
                & ~((values[:, 2] == -1) & (values[:, 3] == -1))
            )

        valid_first = valid_half("first")
        valid_second = valid_half("second")
        if np.any(valid_first & valid_second):
            raise ValueError(f"case {case}: target halves overlap")

        measured = shapes[["NUMBER"]].copy()
        for name in measured_columns[1:]:
            measured[name] = np.where(
                valid_first, shapes[f"{name}_first"], shapes[f"{name}_second"]
            )
        measured["valid"] = valid_first | valid_second

        joined = cross.merge(
            measured,
            left_on="id_detec",
            right_on="NUMBER",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int((joined["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"case {case}: {missing:,} crossmatches lack shapes")
        joined = joined.drop(columns="_merge").merge(
            truth,
            left_on="id_input",
            right_on="index_input",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int((joined["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"case {case}: {missing:,} crossmatches lack truth rows")
        joined = joined.drop(columns="_merge")
        joined = joined.loc[joined["valid"]].copy()
        joined.insert(0, "case", np.int16(case))
        frames.append(joined)
        audits.append(
            {
                "case": case,
                "truth_rows": len(truth),
                "crossmatched_detections": len(cross),
                "truth_without_crossmatched_detection": len(truth) - len(cross),
                "retained": len(joined),
            }
        )
        print(f"raw case {case:02d}: {len(joined):,} rows", flush=True)

    data = pd.concat(frames, ignore_index=True)
    data["measured_radius_arcsec"] = data["FLUX_RADIUS"] * PIXEL_SCALE
    audit = pd.DataFrame(audits)
    print(
        f"raw: retained {len(data):,}; unmatched truth objects "
        f"{audit.truth_without_crossmatched_detection.sum():,}"
    )
    return data, audit


def _keys(case, index):
    return (
        np.asarray(case, dtype=np.uint64) << np.uint64(32)
    ) | np.asarray(index, dtype=np.uint64)


def _match(left_case, left_index, right_case, right_index):
    left = _keys(left_case, left_index)
    right = _keys(right_case, right_index)
    if len(np.unique(left)) != len(left) or len(np.unique(right)) != len(right):
        raise ValueError("response keys are not unique")
    _, li, ri = np.intersect1d(left, right, assume_unique=True, return_indices=True)
    order = np.argsort(left[li], kind="stable")
    return li[order], ri[order]


def _load_anchor():
    path = DOMAIN_ROOT / "response_anchor.feather"
    parts = []
    with ipc.open_file(path) as reader:
        for batch_number in range(reader.num_record_batches):
            batch = reader.get_batch(batch_number)
            frame = pa.Table.from_batches([batch]).select(["case", "input_index"]).to_pandas()
            frame = frame.loc[frame["case"].isin(CASES)]
            if not frame.empty:
                parts.append(frame)
    anchor = pd.concat(parts, ignore_index=True)
    if anchor.duplicated(["case", "input_index"]).any():
        raise ValueError("response anchor keys are duplicated")
    return anchor


def _load_self_response():
    frames = []
    unmatched_zero = 0
    unmatched_sheared = 0
    for case in CASES:
        with np.load(DOMAIN_ROOT / f"flow/g0/case{case:03d}.npz") as zero, np.load(
            DOMAIN_ROOT / f"flow/g05/case{case:03d}.npz"
        ) as sheared:
            i0, ig = _match(
                zero["case"], zero["input_index"], sheared["case"], sheared["input_index"]
            )
            unmatched_zero += len(zero["case"]) - len(i0)
            unmatched_sheared += len(sheared["case"]) - len(ig)
            g = sheared["gamma"][ig].astype(float)
            delta = sheared["target"][ig, :2] - zero["target"][i0, :2]
            h2 = np.einsum("ij,ij->i", g, g)
            keep = np.isfinite(delta).all(axis=1) & np.isfinite(g).all(axis=1) & (h2 > 0)
            i0 = i0[keep]
            g = g[keep]
            delta = delta[keep]
            h2 = h2[keep]
            target0 = zero["target"][i0]
            frame = pd.DataFrame(
                {
                    "case": np.full(len(i0), case, dtype=np.int16),
                    "input_index": zero["input_index"][i0].astype(np.int64),
                    "R_self": np.einsum("ij,ij->i", delta, g) / h2,
                    "R_self_cross": (-delta[:, 0] * g[:, 1] + delta[:, 1] * g[:, 0])
                    / h2,
                    "g0_MAG_AUTO": 30.0 - 2.5 * np.log10(target0[:, 3]),
                    "g0_FLUX_RADIUS_arcsec": target0[:, 2] * PIXEL_SCALE,
                }
            )

        truth = pd.read_feather(
            DATA_ROOT / f"case{case}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather",
            columns=["index_input", "r_input", "Re_input"],
        ).rename(
            columns={
                "index_input": "input_index",
                "r_input": "true_r_magnitude",
                "Re_input": "true_Re_semimajor_arcsec",
            }
        )
        frame = frame.merge(
            truth,
            on="input_index",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int((frame["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"case {case}: {missing:,} self responses lack truth rows")
        frame = frame.drop(columns="_merge")
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    if not ((result["g0_MAG_AUTO"] < 25.8).all() and (result["g0_FLUX_RADIUS_arcsec"] > 0.6).all()):
        raise ValueError("self response escapes the measured g=0 domain")
    print(
        f"R_self: {len(result):,}; unmatched g0={unmatched_zero:,}, "
        f"g05={unmatched_sheared:,}"
    )
    return result


def _load_blend_response(anchor):
    anchor_ids = {
        int(case): group["input_index"].to_numpy(np.int64)
        for case, group in anchor.groupby("case")
    }
    columns = [
        "case",
        "input_index",
        "delta_et1",
        "delta_et2",
        "measured_e1_m",
        "measured_e2_m",
        "measured_e1_p",
        "measured_e2_p",
    ]
    parts = []
    path = DATA_ROOT / "response_catalogue_train.feather"
    with ipc.open_file(path) as reader:
        case_column = reader.schema.get_field_index("case")
        for batch_number in range(reader.num_record_batches):
            batch = reader.get_batch(batch_number)
            batch_cases = batch.column(case_column).to_numpy(zero_copy_only=False)
            if batch_cases.min() > CASES[-1]:
                break
            if batch_cases.max() < CASES[0]:
                continue
            frame = pa.Table.from_batches([batch]).select(columns).to_pandas()
            frame = frame.loc[frame["case"].isin(CASES)].copy()
            if frame.empty:
                continue
            anchored = np.zeros(len(frame), dtype=bool)
            for case in frame["case"].unique():
                local = frame["case"].to_numpy() == case
                anchored[local] = np.isin(
                    frame.loc[local, "input_index"], anchor_ids[int(case)]
                )
            frame = frame.loc[anchored]
            measured = frame[
                ["measured_e1_m", "measured_e2_m", "measured_e1_p", "measured_e2_p"]
            ].to_numpy(float)
            keep = np.isfinite(frame[columns].to_numpy(float)).all(axis=1)
            keep &= np.square(measured[:, :2]).sum(axis=1) < 1
            keep &= np.square(measured[:, 2:]).sum(axis=1) < 1
            frame = frame.loc[keep].copy()
            frame["R_blend"] = frame["delta_et1"] / RESPONSE_SHEAR
            frame["R_blend_cross"] = frame["delta_et2"] / RESPONSE_SHEAR
            parts.append(
                frame.groupby(["case", "input_index"], as_index=False).agg(
                    R_blend=("R_blend", "sum"),
                    R_blend_cross=("R_blend_cross", "sum"),
                )
            )
            if (batch_number + 1) % 100 == 0:
                print(
                    f"response batch {batch_number + 1}/{reader.num_record_batches}",
                    flush=True,
                )

    response = pd.concat(parts).groupby(
        ["case", "input_index"], as_index=False
    ).sum()
    response = anchor.merge(
        response,
        on=["case", "input_index"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_pairs = int((response["_merge"] != "both").sum())
    response = response.loc[response["_merge"] == "both"].drop(columns="_merge")

    baselines = []
    for case in CASES:
        keys = anchor.loc[anchor["case"] == case, ["case", "input_index"]]
        catalogue = DATA_ROOT / f"case{case}_0.0/real0/catalogues"
        cross = pd.read_feather(
            catalogue / "CrossMatch/tile180.0_-0.5_rot0_matched.feather",
            columns=["id_detec", "id_input"],
        )
        shape = pd.read_feather(
            catalogue / "Shapes/shape_catalogue_detect_position_tile180.0_-0.5.feather",
            columns=["NUMBER", "MAG_AUTO", "FLUX_RADIUS"],
        )
        truth = pd.read_feather(
            catalogue / "input/gals_info_tile180.0_-0.5.feather",
            columns=["index_input", "r_input", "Re_input"],
        )
        baseline = keys.merge(
            cross,
            left_on="input_index",
            right_on="id_input",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int((baseline["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"case {case}: {missing:,} blend anchors lack crossmatches")
        baseline = baseline.drop(columns="_merge").merge(
            shape,
            left_on="id_detec",
            right_on="NUMBER",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int((baseline["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"case {case}: {missing:,} blend anchors lack shapes")
        baseline = baseline.drop(columns="_merge").merge(
            truth,
            left_on="input_index",
            right_on="index_input",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int((baseline["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"case {case}: {missing:,} blend anchors lack truth rows")
        baseline = baseline.drop(columns="_merge")
        baselines.append(
            baseline[
                ["case", "input_index", "MAG_AUTO", "FLUX_RADIUS", "r_input", "Re_input"]
            ]
        )

    baseline = pd.concat(baselines, ignore_index=True).rename(
        columns={
            "MAG_AUTO": "g0_MAG_AUTO",
            "r_input": "true_r_magnitude",
            "Re_input": "true_Re_semimajor_arcsec",
        }
    )
    baseline["g0_FLUX_RADIUS_arcsec"] = baseline.pop("FLUX_RADIUS") * PIXEL_SCALE
    result = response.merge(
        baseline, on=["case", "input_index"], validate="one_to_one"
    )
    if not ((result["g0_MAG_AUTO"] < 25.8).all() and (result["g0_FLUX_RADIUS_arcsec"] > 0.6).all()):
        raise ValueError("blend response escapes the measured g=0 domain")
    print(f"R_blend: {len(result):,}; anchors without finite pairs={missing_pairs:,}")
    return result


def load_responses():
    anchor = _load_anchor()
    return _load_self_response(), _load_blend_response(anchor)


def load_models(self_response, blend_response):
    pairs = []
    for measured, filename, column in (
        (self_response, "R_self_model.feather", "R_self_model"),
        (blend_response, "R_blend_model.feather", "R_blend_model"),
    ):
        model = pd.read_feather(MODEL_ROOT / filename)
        if model.duplicated(["case", "input_index"]).any() or not np.isfinite(model[column]).all():
            raise ValueError(f"invalid {filename}")
        joined = measured.merge(
            model,
            on=["case", "input_index"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = int((joined["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"{filename}: {missing:,} measured rows lack predictions")
        pairs.append(joined.drop(columns="_merge"))
    return tuple(pairs)
