"""Validate the response-aware selection classifier: does its induced selection response
b_model now match the sim's b_true, GLOBALLY and PER (flux x size x blend) cell?

On a sheared catalogue (parent = detected + undetected):
  s_par      = S_gamma(e_intrinsic) . ghat
  b_true(cell)  = [<s_par>_det,cell  - <s_par>_cell] / g          (sim; real detected label)
  b_model(cell) = [ (sum w*s_par / sum w)_cell - <s_par>_cell ] / g
                   w = P(s=1 | sheared-scene features)            (classifier)

Run on the OLD vs NEW classifier for a before/after.  PASS = b_model tracks b_true (sign +
magnitude), globally near the sim value and per-cell tracking the -4..-6% structure.
"""
import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.selection_model import load_selection_model  # noqa
from sbs_shear.preprocessing import raw_columns_for_selection_features, rescale  # noqa
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa
from sbs_shear.sim_stream import stream_reservoir  # noqa

_PARENT_EXTRA = {"detected", "gamma1_input_p", "gamma2_input_p", "e1_input_rot0_p",
                 "e2_input_rot0_p", "r_input_p", "Re_input_p", "distance", "neighbored"}


def load_parent(cat, feats, max_rows, seed):
    cols = lambda avail: raw_columns_for_selection_features(feats, available_columns=avail) | _PARENT_EXTRA  # noqa: E731
    return stream_reservoir(cat, cols, max_rows, seed=seed, detected=None, shear_threshold=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-model", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--target-shear", type=float, default=0.05,
                    help="shear the response target was built at; g==this => IN-SAMPLE, else HELD-OUT")
    ap.add_argument("--n-flux", type=int, default=4)
    ap.add_argument("--n-size", type=int, default=2)
    ap.add_argument("--n-dist", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pixel-rms", type=float, default=0.312)
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--zero-mag", type=float, default=30.0)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    args = ap.parse_args()

    import torch  # noqa
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sel = load_selection_model(args.selection_model, device=device)
    feats = sel.preprocessor.feature_names
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    g = args.nominal_g

    df = load_parent(args.catalogue, feats, args.max_rows, args.seed)
    g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2); gh1, gh2 = g1 / gm, g2 / gm
    sc1, sc2 = apply_shear_to_ellipticity(df["e1_input_rot0_p"].to_numpy(float),
                                          df["e2_input_rot0_p"].to_numpy(float), g1, g2)
    s_par = sc1 * gh1 + sc2 * gh2
    det = df["detected"].astype(bool).to_numpy()
    frame = df.copy(); frame["e1_input_rot0_p"] = sc1; frame["e2_input_rot0_p"] = sc2
    for c in ("gamma1_input_p", "gamma2_input_p"):
        if c in frame.columns:
            frame[c] = 0.0          # shear folded into rot0; match the trainer's shifted_ctx
    frame = rescale(frame, **rk)
    w = sel.predict_proba(frame, batch_size=65536)

    flux = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    nbf = df["neighbored"].astype(bool).to_numpy(); dist = df["distance"].to_numpy(float)
    ef = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-6; ef[-1] += 1e-6
    es = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-6; es[-1] += 1e-6
    db = dist[nbf & np.isfinite(dist)]
    ed = np.quantile(db, np.linspace(0, 1, args.n_dist + 1)); ed[0] -= 1e-6; ed[-1] += 1e-6
    fi = np.clip(np.digitize(flux, ef) - 1, 0, args.n_flux - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, args.n_size - 1)
    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, args.n_dist - 1), 0)
    nbl = args.n_dist + 1

    def bcell(mask):
        par = float(np.mean(s_par[mask]))
        bt = (float(np.mean(s_par[mask & det])) - par) / g
        ww = w[mask]
        bm = (float(np.sum(ww * s_par[mask]) / np.sum(ww)) - par) / g
        return bt, bm

    tag = "IN-SAMPLE (target shear)" if abs(g - args.target_shear) < 1e-3 else "HELD-OUT (genuine test)"
    print(f"model={os.path.basename(args.selection_model)}  g={g}  [{tag}]  N_parent={len(df):,}  det_frac={det.mean():.4f}")
    gt, gm_ = bcell(np.ones(len(df), bool))
    print(f"GLOBAL: b_true/g={gt*100:+.3f}%   b_model/g={gm_*100:+.3f}%")
    print("\nper blend bin (count-weighted over flux x size):")
    print(f"{'blend':>10} {'b_true/g%':>10} {'b_model/g%':>11} {'N':>10}")
    for c in range(nbl):
        m = di == c
        if m.sum() < 1000:
            continue
        bt, bm = bcell(m)
        tag = "ISOLATED" if c == 0 else f"blend d{c}"
        print(f"{tag:>10} {bt*100:>+10.2f} {bm*100:>+11.2f} {int(m.sum()):>10,}")
    errs = []
    for a in range(args.n_flux):
        for b in range(args.n_size):
            for c in range(nbl):
                m = (fi == a) & (si == b) & (di == c)
                if m.sum() > 2000 and det[m].sum() > 200:
                    bt, bm = bcell(m)
                    errs.append(bm - bt)
    errs = np.array(errs)
    print(f"\nper-cell |b_model - b_true|/g: mean={np.mean(np.abs(errs))*100:.2f}%  "
          f"median={np.median(np.abs(errs))*100:.2f}%  (baseline classifier was ~3.7% global)")


if __name__ == "__main__":
    main()
