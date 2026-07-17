"""Fix the per-magnitude tilt: is it removed by a MAGNITUDE-CONDITIONAL classifier calibration?

Diagnosis (probblend_forward tilt run): the undetected census soft/hard flips sign with primary
magnitude — classifier over-predicts undetectedness at bright primaries (their neighbours) and
UNDER-predicts by 35% at the faintest (r_p 27-28). Because faint primaries have faint neighbours,
this is a magnitude-dependent miscalibration of the neighbour p_det. A single global isotonic curve
cannot fix both signs.

Test: calibrate p_det -> P(detected) with isotonic regression fit WITHIN neighbour-magnitude bins
(2D calibration in p_det x r), then re-measure the response-weighted undetected census soft/hard
PER PRIMARY-MAGNITUDE bin, comparing raw / global-isotonic / magnitude-conditional. If the 2D
calibration flattens soft/hard ~1 across all primary-mag bins, the tilt is a calibration artefact
and this is the fix.
"""
import sys
import numpy as np, pandas as pd, pyarrow.feather as pf
from sklearn.isotonic import IsotonicRegression

BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
DET = f"{BASE}/detection_catalogue_train.feather"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"
CASES = [0, 1, 2, 3, 4, 5, 6, 7]
MAG_BINS = np.array([13, 24, 25, 25.5, 26, 26.5, 27, 27.5, 28, 29.5])   # neighbour-mag calibration bins


def main():
    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred = BlendingPredictor.load(BLEND_MODELS, tag="lsst_r_extnbr_ho", conditions=COND, device="cpu")

    det = pf.read_table(DET, columns=["case", "input_index", "detected"]).to_pandas()
    det = det[det.case.isin(CASES)].drop_duplicates(["case", "input_index"])

    # per-galaxy p_det + detection truth + own magnitude
    rows = []
    for c in CASES:
        fld = pf.read_table(f"{BASE}/case{c}_0.0/real0/catalogues/input/gals_info_{TILE}.feather").to_pandas()
        fld = fld.rename(columns={col: col.replace("_input", "") for col in fld.columns})
        dfd = pred.predict_detection(fld, warn_extrapolation=False)
        ik = [x for x in dfd.columns if x.startswith("index") and x.endswith("_p")][0]
        rows.append(pd.DataFrame({"case": c, "idx": dfd[ik].to_numpy().astype(np.int64),
                                  "p_det": dfd["detection_prob"].to_numpy(),
                                  "r": dfd["r_input_p"].to_numpy()}))
    G = pd.concat(rows, ignore_index=True).merge(
        det.rename(columns={"input_index": "idx"}), on=["case", "idx"], how="left")
    G["detected"] = G["detected"].fillna(False).astype(bool)

    # response weight per galaxy (as undetected neighbour)
    P = pf.read_table("/home/z/Zekang.Zhang/SBSI/results/probblend_char.feather",
                      columns=["case", "ip", "is_", "rp", "resp", "det_p", "det_s"]).to_pandas()
    P = P[P.det_p.to_numpy()]
    w = P.groupby(["case", "is_"]).resp.sum().reset_index().rename(columns={"is_": "idx", "resp": "w"})
    G = G.merge(w, on=["case", "idx"], how="left"); G["w"] = G["w"].fillna(0.0)
    G = G[G.case.isin(P.case.unique())].copy()
    G["mbin"] = np.clip(np.digitize(G.r, MAG_BINS) - 1, 0, len(MAG_BINS) - 2)
    Gpos = G[G.w > 0].copy()                       # isotonic needs non-negative weights

    # --- calibrators ---
    giso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    giso.fit(Gpos.p_det, Gpos.detected.astype(float), sample_weight=Gpos.w)
    G["p_giso"] = giso.predict(G.p_det.to_numpy())

    # magnitude-conditional: isotonic within neighbour-magnitude bins
    G["p_2d"] = G["p_det"]
    for b, gg in Gpos.groupby("mbin"):
        if len(gg) < 500 or gg.detected.nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(gg.p_det, gg.detected.astype(float), sample_weight=gg.w)
        m = G.mbin == b
        G.loc[m, "p_2d"] = iso.predict(G.loc[m, "p_det"].to_numpy())

    # --- map calibrated p_det back onto the char pairs (secondary = neighbour) ---
    pm = G.set_index(["case", "idx"])[["p_det", "p_giso", "p_2d"]]
    Pp = P.merge(pm, left_on=["case", "is_"], right_index=True, how="left").dropna(subset=["p_det"])

    print("undetected census soft/hard PER PRIMARY-magnitude bin:")
    print(f"{'r_p bin':>10} {'N pairs':>10} {'hard':>9} {'raw':>9} {'g-iso':>9} {'2D':>9}   "
          f"{'raw/h':>6} {'giso/h':>6} {'2D/h':>6}")
    edges = [(18, 24), (24, 25), (25, 26), (26, 27), (27, 28.1)]
    R_TOTAL = 0.462
    agg = {}
    for lo, hi in edges:
        s = Pp[(Pp.rp >= lo) & (Pp.rp < hi)]
        if len(s) < 1000:
            continue
        hard = np.sum((1 - s.det_s.astype(float)) * s.resp)
        raw = np.sum((1 - s.p_det) * s.resp)
        gis = np.sum((1 - s.p_giso) * s.resp)
        two = np.sum((1 - s.p_2d) * s.resp)
        n = s.rp.count()
        agg[(lo, hi)] = (hard, raw, gis, two, len(s))
        print(f"  {lo:.0f}-{hi:4.1f} {len(s):>10,} {hard:9.1f} {raw:9.1f} {gis:9.1f} {two:9.1f}   "
              f"{raw/hard:6.2f} {gis/hard:6.2f} {two/hard:6.2f}")
    # global
    tot = {k: sum(a[i] for a in agg.values()) for i, k in enumerate(["hard", "raw", "gis", "two"])}
    print(f"  {'GLOBAL':>8} {'':>10} {tot['hard']:9.1f} {tot['raw']:9.1f} {tot['gis']:9.1f} {tot['two']:9.1f}   "
          f"{tot['raw']/tot['hard']:6.2f} {tot['gis']/tot['hard']:6.2f} {tot['two']/tot['hard']:6.2f}")
    # per-bin tilt magnitude: max|soft/hard-1| over bins, as a Delta_m proxy on the undetected term
    def tilt(idx):
        return max(abs(a[idx] / a[0] - 1) for a in agg.values())
    print(f"\n  worst per-bin |soft/hard-1|:  raw={tilt(1):.1%}  g-iso={tilt(2):.1%}  2D={tilt(3):.1%}")
    print("PROBBLEND_CALIB2D_DONE")


if __name__ == "__main__":
    main()
