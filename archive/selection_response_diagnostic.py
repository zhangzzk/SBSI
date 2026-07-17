"""Selection-response gating diagnostic (companion to response_ratio_diagnostic.py).

The catalogue density is p_cat = p_meas(xhat|x) * P(s=1|x), so the estimator's shear
response is d log p_meas/dg + d log P(s=1)/dg.  The measurement diagnostic checks the
first term; this one checks the SECOND -- whether the learned selection classifier's
induced shear response matches the simulation's actual (shape-dependent) selection.

isolate_selection_effect already showed detection fraction is shear-independent when
binned by true (size x mag): selection is NOT in the magnification channel.  The channel
that can still bias m is SHAPE/ORIENTATION-dependent selection: detection depends on the
rendered scene shape projected on the shear axis, so shear shifts the detected sample's
mean shape.  We quantify that as the selection-induced shift in mean true-scene shape
along the shear direction:

  s_par   = S_gamma(intrinsic) . ghat        (true scene shape projected on shear axis)
  b_true(g) = <s_par>_detected - <s_par>_parent          (sim: uses the real label)
  b_model(g)= (sum w*s_par)/(sum w) - <s_par>_parent     (classifier: w = P(s=1|scene))

If b_model ~= b_true the classifier's selection response is calibrated; if not it is
mis-specified -- which is exactly why turning on p_cat moved m the WRONG way (+0.038 vs
+0.030).  b/g is the additive selection contribution per unit shear (compare to m).
A response-aware selection loss would supervise b_model -> b_true.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.selection_model import load_selection_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402


def load_parent_sample(catalogue, sel_bundle, max_rows, shear_threshold, seed, max_read_batches=None):
    """Stream a sheared catalogue, apply ONLY the source (parent) selection -- keep BOTH
    detected and undetected rows, so detection fraction is measurable."""
    rng = np.random.default_rng(seed)
    cond_features = sel_bundle.preprocessor.feature_names

    with ipc.open_file(catalogue) as reader:
        available = set(reader.schema.names)
        needed = set()
        needed |= raw_columns_for_selection_features(cond_features, available_columns=available)
        needed |= {"detected", "gamma1_input_p", "gamma2_input_p"}
        needed |= {"e1_input_rot0_p", "e2_input_rot0_p"}
        needed |= {"r_input_p", "Re_input_p", "distance", "neighbored"}
        read_columns = sorted(c for c in needed if c in available)

        reservoir = None
        raw_rows = 0
        for bi in range(reader.num_record_batches):
            if max_read_batches is not None and bi >= max_read_batches:
                break
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(read_columns).to_pandas()
            raw_rows += len(batch)
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
            if len(batch) == 0:
                continue
            gmag = np.hypot(batch["gamma1_input_p"].to_numpy(float), batch["gamma2_input_p"].to_numpy(float))
            batch = batch[gmag > shear_threshold].reset_index(drop=True)
            if len(batch) == 0:
                continue
            batch = batch.copy()
            batch["__key"] = rng.random(len(batch))
            reservoir = batch if reservoir is None else pd.concat([reservoir, batch], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)

    if reservoir is None:
        raise SystemExit(f"No parent rows selected from {catalogue}")
    if len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    reservoir = reservoir.drop(columns="__key").reset_index(drop=True)
    print(f"  raw scanned={raw_rows:,}  parent kept={len(reservoir):,}  "
          f"det_frac={reservoir['detected'].astype(bool).mean():.4f}")
    return reservoir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection-model", default=os.path.join(SBSI_ROOT, "models/selection_mlp_g0_shearfree_v1.pt"))
    ap.add_argument("--catalogue-005", required=True)
    ap.add_argument("--catalogue-020", required=True)
    ap.add_argument("--max-rows", type=int, default=800_000)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--shear-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=os.path.join(SBSI_ROOT, "results/response_ratio"))
    ap.add_argument("--pixel-rms", type=float, default=0.312)
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--zero-mag", type=float, default=30.0)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    sel = load_selection_model(args.selection_model, device=device)
    tag = os.path.splitext(os.path.basename(args.selection_model))[0]
    print(f"Selection model: {tag}  features={len(sel.preprocessor.feature_names)}")

    rescale_kwargs = dict(
        pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
        psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta,
    )

    rows = []
    for g, cat in ((0.05, args.catalogue_005), (0.20, args.catalogue_020)):
        print(f"\n########## nominal g={g}  catalogue={os.path.basename(cat)} ##########", flush=True)
        base = load_parent_sample(cat, sel, args.max_rows, args.shear_threshold, args.seed)

        g1 = base["gamma1_input_p"].to_numpy(float)
        g2 = base["gamma2_input_p"].to_numpy(float)
        gmag = np.hypot(g1, g2)
        ghat1, ghat2 = g1 / gmag, g2 / gmag
        det = base["detected"].astype(bool).to_numpy()

        # true SCENE shape (intrinsic sheared by the actual applied shear) projected on ghat
        e1i = base["e1_input_rot0_p"].to_numpy(float)
        e2i = base["e2_input_rot0_p"].to_numpy(float)
        sc1, sc2 = apply_shear_to_ellipticity(e1i, e2i, g1, g2)
        s_par = sc1 * ghat1 + sc2 * ghat2
        mean_parent = float(np.mean(s_par))

        # b_true: selection-induced shift of mean scene-shape-along-shear (sim label)
        b_true = float(np.mean(s_par[det])) - mean_parent

        # b_model: classifier reweighting.  Feed it the scene shape (what it would see for
        # the sheared galaxy) by overwriting the shape columns then re-deriving features.
        frame = base.copy()
        frame["e1_input_rot0_p"] = sc1
        frame["e2_input_rot0_p"] = sc2
        frame = rescale(frame, **rescale_kwargs)
        t0 = time.time()
        w = sel.predict_proba(frame, batch_size=args.batch_size)
        b_model = float(np.sum(w * s_par) / np.sum(w)) - mean_parent

        detf = float(det.mean())
        wbar = float(np.mean(w))
        print(f"  det_frac(sim)={detf:.4f}  mean_Psel(model)={wbar:.4f}  "
              f"<s_par>_parent={mean_parent:+.5f}")
        print(f"  b_true ={b_true:+.6f}   b_true/g ={b_true/g:+.5f}")
        print(f"  b_model={b_model:+.6f}   b_model/g={b_model/g:+.5f}  "
              f"(eval {time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(g=g, det_frac=detf, mean_Psel=wbar, mean_parent=mean_parent,
                         b_true=b_true, b_model=b_model, b_true_per_g=b_true / g,
                         b_model_per_g=b_model / g, n=len(base)))

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out = os.path.join(args.output_dir, f"selection_response_{tag}.csv")
    df.to_csv(out, index=False)

    print("\n================ SUMMARY ================")
    print(df.to_string(index=False))
    if len(df) == 2:
        bt = df["b_true_per_g"].to_numpy()
        bm = df["b_model_per_g"].to_numpy()
        print(f"\nb_true/g : 0.05 -> {bt[0]:+.5f}   0.2 -> {bt[1]:+.5f}   "
              f"(sim selection response { 'NEGLIGIBLE' if max(abs(bt))<0.003 else 'SIGNIFICANT' })")
        print(f"b_model/g: 0.05 -> {bm[0]:+.5f}   0.2 -> {bm[1]:+.5f}")
        mis = np.max(np.abs(bm - bt))
        print(f"max|b_model-b_true|/g = {mis:.5f}   "
              f"(classifier selection response { 'CALIBRATED' if mis<0.003 else 'MIS-SPECIFIED -> needs response-aware selection loss' })")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
