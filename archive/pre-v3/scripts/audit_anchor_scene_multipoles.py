"""Test intrinsic neighbour-flux multipoles as proxies for anchor instability.

Cases 200--249 define all property cells and proxy medians.  Cases 250--299
test high-minus-low V2.2 residual contrasts within prediction/magnitude/size
cells.  Pair selection is the exact deployed V2.2 support.  No correction is
fit and no constgold quantity is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import data_utils, nz_utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
PSF_RE = 0.5268
TILE = "tile180.0_-0.5"
KEY = ["case", "input_index"]
PROXIES = [
    "log1p_overlap_flux_sum", "log1p_overlap_dipole_abs",
    "overlap_dipole_fraction", "overlap_quadrupole_fraction",
    "close3_dipole_fraction", "close3_quadrupole_fraction",
]


def input_frame(base: str, case: int, sign: float) -> pd.DataFrame:
    path = os.path.join(
        base, f"case{case}_{str(float(sign))}", "real0", "catalogues", "input",
        f"gals_info_{TILE}.feather",
    )
    frame = pd.read_feather(path)
    return frame.rename(columns={column: column.replace("_input", "") for column in frame})


def complex_fraction(weight: np.ndarray, angle: np.ndarray, order: int) -> tuple[float, float]:
    total = float(weight.sum())
    if total <= 0:
        return 0.0, 0.0
    amplitude = float(np.abs(np.sum(weight * np.exp(1j * order * angle))))
    return amplitude, amplitude / total


def build_features(base: str, observed: pd.DataFrame,
                   predictor: BlendingPredictor, sign: float) -> pd.DataFrame:
    cuts, r_max, k = predictor._select("regression")
    parts = []
    for case, case_observed in observed.groupby("case", sort=True):
        case = int(case)
        ids = case_observed.input_index.to_numpy(np.int64)
        frame = input_frame(base, case, sign)
        primary = frame[frame["index"].isin(ids)]
        raw = nz_utils.make_reg_features(
            primary, frame, r_max=float(r_max) / 3600.0, k=int(k),
        )
        primary_id = next(
            column for column in raw if column.startswith("index") and column.endswith("_p")
        )
        pairs = data_utils.source_select_reg(raw, cuts=cuts).copy()
        pairs["distance_arcsec"] = pairs.distance.to_numpy(float) * 3600.0
        cosdec = np.cos(np.deg2rad(float(frame.DEC.median())))
        dx = (
            pairs.RA_input_s.to_numpy(float) - pairs.RA_input_p.to_numpy(float)
        ) * cosdec * 3600.0
        dy = (
            pairs.DEC_input_s.to_numpy(float) - pairs.DEC_input_p.to_numpy(float)
        ) * 3600.0
        pairs["phi"] = np.arctan2(dy, dx)
        flux_ratio = np.power(
            10.0, -0.4 * (
                pairs.r_input_s.to_numpy(float) - pairs.r_input_p.to_numpy(float)
            ),
        )
        width = np.sqrt(
            np.square(pairs.Re_input_p.to_numpy(float))
            + np.square(pairs.Re_input_s.to_numpy(float)) + PSF_RE**2
        )
        pairs["overlap_weight"] = flux_ratio * np.exp(
            -0.5 * np.square(pairs.distance_arcsec.to_numpy(float) / width)
        )
        pairs["flux_ratio"] = flux_ratio
        rows = []
        for input_id, group in pairs.groupby(primary_id, sort=False):
            angle = group.phi.to_numpy(float)
            overlap = group.overlap_weight.to_numpy(float)
            dipole_abs, dipole_fraction = complex_fraction(overlap, angle, 1)
            _, quadrupole_fraction = complex_fraction(overlap, angle, 2)
            close = group.distance_arcsec.to_numpy(float) < 3.0
            _, close_dipole = complex_fraction(
                group.flux_ratio.to_numpy(float)[close], angle[close], 1,
            )
            _, close_quadrupole = complex_fraction(
                group.flux_ratio.to_numpy(float)[close], angle[close], 2,
            )
            rows.append({
                "input_index": int(input_id),
                "log1p_overlap_flux_sum": float(np.log1p(overlap.sum())),
                "log1p_overlap_dipole_abs": float(np.log1p(dipole_abs)),
                "overlap_dipole_fraction": dipole_fraction,
                "overlap_quadrupole_fraction": quadrupole_fraction,
                "close3_dipole_fraction": close_dipole,
                "close3_quadrupole_fraction": close_quadrupole,
            })
        feature = pd.DataFrame(rows).set_index("input_index")
        anchors = pd.Index(ids, name="input_index")
        feature = feature.reindex(anchors, fill_value=0.0).reset_index()
        feature["case"] = case
        parts.append(feature)
        print(f"case {case}: anchors={len(ids):,} supported_pairs={len(pairs):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def edges(values: np.ndarray, count: int) -> np.ndarray:
    boundary = np.unique(np.quantile(np.asarray(values, float), np.linspace(0, 1, count + 1)))
    if len(boundary) != count + 1:
        raise RuntimeError("quantile edges collapsed")
    boundary[0], boundary[-1] = -np.inf, np.inf
    return boundary


def assign(values: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(boundary, values, side="right") - 1, 0, len(boundary) - 2)


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    test = stats.ttest_1samp(values, 0.0)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)), "t": float(test.statistic),
        "p_two_sided": float(test.pvalue),
    }


def contrast(frame: pd.DataFrame, proxy: str, threshold: pd.Series,
             target: str) -> tuple[np.ndarray, list[int]]:
    work = frame.copy()
    work["threshold"] = work.cell.map(threshold)
    work = work[np.isfinite(work.threshold)].copy()
    work["high"] = work[proxy] > work.threshold
    values, used = [], []
    for _, case in work.groupby("case", sort=True):
        differences, weights = [], []
        for _, cell in case.groupby("cell", sort=False):
            high = cell.loc[cell.high, target].to_numpy(float)
            low = cell.loc[~cell.high, target].to_numpy(float)
            if len(high) < 3 or len(low) < 3:
                continue
            differences.append(float(high.mean() - low.mean()))
            weights.append(float(min(len(high), len(low))))
        if differences:
            values.append(float(np.average(differences, weights=weights)))
            used.append(len(differences))
    return np.asarray(values), used


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    result, running = {}, 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * pvalues[name]))
        result[name] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--sign", type=float, default=0.05)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    observed = pd.read_feather(args.response)
    observed = observed[observed.case.between(200, 299)].copy()
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    feature = build_features(args.base, observed, predictor, args.sign)
    frame = observed.merge(feature, on=KEY, validate="one_to_one")
    frame.to_feather(args.table_output)
    development = frame[frame.case <= args.development_max].copy()
    validation = frame[frame.case > args.development_max].copy()
    boundaries = {
        "prediction": edges(development.R_blend_lsst_r_extnbr_v22, 10),
        "magnitude": edges(development.r_input_p_plus, 4),
        "size": edges(development.Re_input_p_plus, 4),
    }
    for part in (development, validation):
        p = assign(part.R_blend_lsst_r_extnbr_v22, boundaries["prediction"])
        m = assign(part.r_input_p_plus, boundaries["magnitude"])
        s = assign(part.Re_input_p_plus, boundaries["size"])
        part["cell"] = p * 16 + m * 4 + s
    endpoints, raw_p = {}, {}
    for proxy in PROXIES:
        threshold = development.groupby("cell", sort=True)[proxy].median()
        dev_gap, dev_used = contrast(development, proxy, threshold, "gap")
        val_gap, val_used = contrast(validation, proxy, threshold, "gap")
        dev_shift, _ = contrast(development, proxy, threshold, "centroid_shift_arcsec")
        val_shift, _ = contrast(validation, proxy, threshold, "centroid_shift_arcsec")
        endpoints[proxy] = {
            "development_gap": stat(dev_gap), "validation_gap": stat(val_gap),
            "development_centroid_shift": stat(dev_shift),
            "validation_centroid_shift": stat(val_shift),
            "development_median_cells_per_case": float(np.median(dev_used)),
            "validation_median_cells_per_case": float(np.median(val_used)),
        }
        raw_p[proxy] = endpoints[proxy]["validation_gap"]["p_two_sided"]
    adjusted = holm(raw_p)
    for proxy, value in adjusted.items():
        endpoints[proxy]["validation_gap_p_holm"] = float(value)
    payload = {
        "design": "c200--249 thresholds; c250--299 validation; within 10 V2.2 prediction x 4 mag x 4 size cells",
        "overlap_weight": "flux_secondary/flux_primary * exp[-0.5*(distance/sqrt(Re_p^2+Re_s^2+PSF_Re^2))^2]",
        "endpoints": endpoints,
        "n_rows": int(len(frame)),
        "correction_fitted": False, "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_SCENE_MULTIPOLE_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
