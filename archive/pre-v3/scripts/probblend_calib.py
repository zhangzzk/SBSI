"""Close the limiting residual: the classifier's soft census over-predicts the undetected blend
response by 8% (soft/hard=1.084) -> Delta_m ~ -0.28%. Is this a simple monotonic MIScalibration of
p_det (fixable by recalibration against detection truth), and does recalibration drive soft->hard?

Per galaxy (as a potential undetected NEIGHBOUR), we have:
  - p_det  : classifier prediction (predict_detection on the true field)
  - detected: detection truth
  - w      : its response weight = sum of f_reg over the detected primaries it neighbours
             (from probblend_char.feather) — how much it matters for R_blend.
Response-WEIGHTED calibration curve p_det vs detected rate reveals the miscalibration; isotonic
regression (fit on detection truth, the legitimate training signal) recalibrates it; we then
re-measure the response-weighted soft census with the calibrated p_det vs the hard truth.
"""
import pathlib as _pathlib, sys as _sys  # noqa: E402  -- make `sbs_shear` importable
_sys.path.insert(0, str(next(p for p in _pathlib.Path(__file__).resolve().parents
                             if (p / 'sbs_shear').is_dir())))
from sbs_shear.emulator import load_blending_predictor  # noqa: E402
import sys
import numpy as np, pandas as pd, pyarrow.feather as pf
from sklearn.isotonic import IsotonicRegression
from sbs_shear.paths import SIM_BASE as BASE

DET = f"{BASE}/detection_catalogue_train.feather"
TILE = "tile180.0_-0.5"
CASES = [0, 1, 2, 3]
R_TOTAL = 0.462


def main():
    pred = load_blending_predictor(tag="lsst_r_extnbr_ho")

    det = pf.read_table(DET, columns=["case", "input_index", "detected"]).to_pandas()
    det = det[det.case.isin(CASES)].drop_duplicates(["case", "input_index"])

    # per-galaxy p_det (classifier on the true field) and detection truth
    rows = []
    for c in CASES:
        fld = pf.read_table(f"{BASE}/case{c}_0.0/real0/catalogues/input/gals_info_{TILE}.feather").to_pandas()
        fld = fld.rename(columns={col: col.replace("_input", "") for col in fld.columns})
        dfd = pred.predict_detection(fld, warn_extrapolation=False)
        ik = [x for x in dfd.columns if x.startswith("index") and x.endswith("_p")][0]
        rows.append(pd.DataFrame({"case": c, "idx": dfd[ik].to_numpy().astype(np.int64),
                                  "p_det": dfd["detection_prob"].to_numpy(),
                                  "r": dfd["r_input_p"].to_numpy()}))
    G = pd.concat(rows, ignore_index=True)
    G = G.merge(det.rename(columns={"input_index": "idx"}), on=["case", "idx"], how="left")
    G["detected"] = G["detected"].fillna(False).astype(bool)

    # response weight per galaxy = sum of f_reg over detected primaries it neighbours
    P = pf.read_table("/home/z/Zekang.Zhang/SBSI/results/probblend_char.feather",
                      columns=["case", "is_", "resp", "det_p"]).to_pandas()
    P = P[P.det_p.to_numpy()]
    w = P.groupby(["case", "is_"]).resp.sum().reset_index().rename(columns={"is_": "idx", "resp": "w"})
    G = G.merge(w, on=["case", "idx"], how="left")
    G["w"] = G["w"].fillna(0.0)
    G = G[G.case.isin(P.case.unique())]           # only cases present in char file

    # --- response-weighted calibration curve ---
    print("response-weighted calibration (p_det bin -> mean p_det vs actual detected rate):")
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (G.p_det >= lo) & (G.p_det < hi)
        ww = G.w[m]
        if ww.abs().sum() < 1:
            continue
        pred_rate = np.average(G.p_det[m], weights=ww) if ww.sum() > 0 else np.nan
        act_rate = np.average(G.detected[m].astype(float), weights=ww) if ww.sum() > 0 else np.nan
        print(f"  p_det {lo:.1f}-{hi:.1f}: <p_det>={pred_rate:.3f}  actual_det={act_rate:.3f}  "
              f"w_frac={ww.sum()/G.w.sum():+.1%}")

    # soft (raw) vs hard census, response weighted
    soft_raw = np.sum((1 - G.p_det) * G.w)
    hard = np.sum((1 - G.detected.astype(float)) * G.w)
    print(f"\n  raw soft/hard = {soft_raw/hard:.3f}")

    # --- isotonic recalibration: p_det -> P(detected|p_det), fit on truth (weighted) ---
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(G.p_det.to_numpy(), G.detected.to_numpy().astype(float), sample_weight=G.w.to_numpy())
    p_cal = iso.predict(G.p_det.to_numpy())
    soft_cal = np.sum((1 - p_cal) * G.w)
    print(f"  isotonic-calibrated soft/hard = {soft_cal/hard:.3f}")

    # The undetected term, if perfectly modelled, is unbiased; the classifier miscalibration adds
    # (soft/hard - 1) * <R_undet>/R_total.  <R_undet_hard> ~ 0.0152 => scale 3.29%.
    DM_SCALE = 3.29  # percent per unit (soft/hard - 1)
    print(f"\n  classifier-calibration Delta_m contribution:")
    print(f"    raw       soft/hard={soft_raw/hard:.3f}  ->  Delta_m ~ {(soft_raw/hard-1)*DM_SCALE:+.2f}%")
    print(f"    isotonic  soft/hard={soft_cal/hard:.3f}  ->  Delta_m ~ {(soft_cal/hard-1)*DM_SCALE:+.2f}%")
    print("PROBBLEND_CALIB_DONE")


if __name__ == "__main__":
    main()
