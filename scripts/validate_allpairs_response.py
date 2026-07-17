"""First-moment multiplicative bias m on the INCOHERENT (half-render) all-pairs validation set,
STREAMING so it uses the full catalogue (memory-bounded), with correct per-(case,target)
weighting and neighbour-averaging inference.

This is the goal-relevant estimator: the response-aware flow calibrates the FIRST-MOMENT response
R_model(bin)->R_sim(bin). We measure, weighted so each (case,galaxy) counts once and a target's
per-neighbour predictions are averaged:

  R_sim   = <e_ngmix . ghat> / g                              (weighted)
  R_model = <(mu(S_{+g}) - mu(S_{-g})) . ghat> / (2g)         (flow induced first moment)
  m       = R_sim / R_model - 1   +/- shot-noise error        (global + per neighbour-distance bin)

Per-(case,target) weights are computed WITHIN each streamed batch (a target's ~few pairs are
contiguous within a case, so boundary error is negligible at 65k-row batches).
"""
import argparse, os, sys
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model, add_measurement_target_features  # noqa
from sbs_shear.preprocessing import (  # noqa
    DEFAULT_SELECTION_CUTS, source_select_selection, rescale, raw_columns_for_selection_features)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa
from scripts.response_ratio_diagnostic import _shape_target_indices  # noqa

NB = 5  # bins: 0=isolated, 1..4 = neighbour-distance quartiles (fixed edges below)


def per_row_proj(bundle, base, s, gh1, gh2, intr, rk, n_samples, batch_size):
    frame = base.copy()
    e1p, e2p = apply_shear_to_ellipticity(intr[0], intr[1], s * gh1, s * gh2)
    frame["e1_input_rot0_p"] = e1p; frame["e2_input_rot0_p"] = e2p
    frame = rescale(frame, **rk)
    draws = bundle.sample(frame, n_samples=n_samples, batch_size=batch_size)
    mean = draws.mean(axis=1)
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    return mean[:, i1] * gh1 + mean[:, i2] * gh2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--dist-edges", type=float, nargs="+", default=[0.5, 2.0, 3.0, 4.5, 7.01])
    ap.add_argument("--max-rows", type=int, default=0)  # 0 = all
    ap.add_argument("--max-case", type=int, default=None,
                    help="keep only case <= this (match the blend-lookup's case coverage).")
    ap.add_argument("--snr-min", type=float, default=None,
                    help="MEASURED S/N cut (FLUX_AUTO/FLUXERR_AUTO) -- WARNING: shear-DEPENDENT "
                         "selection (drops shear-elongated gals); use --true-mag-max instead.")
    ap.add_argument("--true-mag-max", type=float, default=None,
                    help="TRUE-magnitude cut: keep r_input_p < this (brighter). Shear-INDEPENDENT "
                         "-> removes the low-S/N noise-bias tail without a shear-dependent selection.")
    ap.add_argument("--snc-lookup", default=None,
                    help="g=0 per-(case,target) ngmix feather (build_g0_lookup.py). Enables SHAPE-NOISE "
                         "CANCELLATION: R_sim uses [e(g)-e(0)].ghat (intrinsic cancels). Keeps only "
                         "targets detected+converged at BOTH shears (mutual-detection subset).")
    ap.add_argument("--blend-lookup", default=None,
                    help="per-(case,input_index) emulator R_blend feather. Adds a SECOND breakdown "
                         "binned by R_blend so the R_blend~0 bin (self-response only) isolates flow "
                         "calibration from blend on the incoherent set.")
    ap.add_argument("--rb-edges", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.25],
                    help="R_blend bin edges; bin 0 = R_blend<first edge (truly isolated).")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pixel-rms", type=float, default=0.312)
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--zero-mag", type=float, default=30.0)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    g = args.nominal_g
    ded = np.asarray(args.dist_edges, float)

    snc_keys = snc_e1 = snc_e2 = None
    if args.snc_lookup:
        import pyarrow.feather as _paf
        lk = _paf.read_table(args.snc_lookup).to_pandas()
        k = lk["case"].to_numpy(np.int64) * 1_000_003 + lk["input_index"].to_numpy(np.int64)
        o = np.argsort(k)
        snc_keys = k[o]; snc_e1 = lk["ngmix0_g1"].to_numpy(float)[o]; snc_e2 = lk["ngmix0_g2"].to_numpy(float)[o]
        print(f"SNC ON: {len(snc_keys):,} g=0 counterparts loaded (mutual-detection pairing)")

    blk_keys = blk_rb = None
    if args.blend_lookup:
        import pyarrow.feather as _pafb
        bl = _pafb.read_table(args.blend_lookup).to_pandas()
        bk = bl["case"].to_numpy(np.int64) * 1_000_003 + bl["input_index"].to_numpy(np.int64)
        bo = np.argsort(bk)
        blk_keys = bk[bo]; blk_rb = bl["R_blend"].to_numpy(float)[bo]
        print(f"BLEND-BIN ON: {len(blk_keys):,} R_blend entries loaded; edges={args.rb_edges}")
    rbe = np.asarray(args.rb_edges, float)
    nrb = len(rbe) + 1

    need = {"measured_ngmix_g1", "measured_ngmix_g2", "gamma1_input_p", "gamma2_input_p",
            "e1_input_rot0_p", "e2_input_rot0_p", "r_input_p", "Re_input_p", "detected",
            "distance", "neighbored", "input_index", "case",
            "measured_flux_auto", "measured_fluxerr_auto"}
    need |= set(raw_columns_for_selection_features(bundle.condition_preprocessor.feature_names))

    # streaming accumulators per bin: sum_w, sum_w*rs, sum_w*rm, sum_w*rs^2, sum_w^2
    nb = NB
    Sw = np.zeros(nb); Swrs = np.zeros(nb); Swrm = np.zeros(nb)
    Swrs2 = np.zeros(nb); Sw2 = np.zeros(nb)
    # parallel accumulators binned by emulator R_blend (bin 0 = R_blend<rbe[0])
    RBw = np.zeros(nrb); RBwrs = np.zeros(nrb); RBwrm = np.zeros(nrb); RBwrs2 = np.zeros(nrb)
    nread = 0; npair = 0
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names); cols = sorted(c for c in need if c in avail)
        for bi in range(r.num_record_batches):
            if args.max_rows and nread >= args.max_rows:
                break
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas(); nread += len(b)
            if args.max_case is not None and "case" in b.columns:
                b = b[b["case"] <= args.max_case]
                if len(b) == 0:
                    continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            g1 = b["gamma1_input_p"].to_numpy(float); g2 = b["gamma2_input_p"].to_numpy(float)
            gm = np.hypot(g1, g2); keep = gm > 1e-6
            if keep.sum() == 0:
                continue
            b = b[keep].reset_index(drop=True); g1, g2, gm = g1[keep], g2[keep], gm[keep]
            if args.true_mag_max is not None and "r_input_p" in b.columns:  # shear-INDEPENDENT cut
                sm = b["r_input_p"].to_numpy(float) < args.true_mag_max
                if sm.sum() == 0:
                    continue
                b = b[sm].reset_index(drop=True); g1, g2, gm = g1[sm], g2[sm], gm[sm]
            if args.snr_min is not None and "measured_flux_auto" in b.columns:  # shear-DEPENDENT (ref only)
                snr = b["measured_flux_auto"].to_numpy(float) / b["measured_fluxerr_auto"].to_numpy(float)
                sm = np.isfinite(snr) & (snr >= args.snr_min)
                if sm.sum() == 0:
                    continue
                b = b[sm].reset_index(drop=True); g1, g2, gm = g1[sm], g2[sm], gm[sm]
            gh1, gh2 = g1 / gm, g2 / gm
            b = add_measurement_target_features(b)
            # per-(case,target) weight within batch
            ii = b["input_index"].to_numpy(np.int64)
            key = (b["case"].to_numpy(np.int64) * 1_000_003 + ii) if "case" in b.columns else ii
            _, inv, npc = np.unique(key, return_inverse=True, return_counts=True)
            w = (1.0 / npc[inv]).astype(float)
            e1 = b["measured_ngmix_g1"].to_numpy(float); e2 = b["measured_ngmix_g2"].to_numpy(float)
            if snc_keys is not None:      # SHAPE-NOISE CANCELLATION: subtract the g=0 counterpart
                pos = np.clip(np.searchsorted(snc_keys, key), 0, len(snc_keys) - 1)
                match = snc_keys[pos] == key
                e1 = e1 - np.where(match, snc_e1[pos], np.nan)   # NaN (no g0 match) -> dropped by fin mask
                e2 = e2 - np.where(match, snc_e2[pos], np.nan)
            r_sim = (e1 * gh1 + e2 * gh2) / g
            intr = (b["e1_input_rot0_p"].to_numpy(float).copy(), b["e2_input_rot0_p"].to_numpy(float).copy())
            pp = per_row_proj(bundle, b, +g, gh1, gh2, intr, rk, args.n_samples, args.batch_size)
            pm = per_row_proj(bundle, b, -g, gh1, gh2, intr, rk, args.n_samples, args.batch_size)
            r_model = (pp - pm) / (2 * g)
            fin = np.isfinite(r_sim) & np.isfinite(r_model)
            nbf = b["neighbored"].astype(bool).to_numpy(); dist = b["distance"].to_numpy(float)
            binid = np.where(nbf, 1 + np.clip(np.digitize(dist, ded) - 1, 0, nb - 2), 0)
            for arr, mask in ((None, fin),):
                pass
            m = fin
            bid = binid[m]; ww = w[m]; rs = r_sim[m]; rm = r_model[m]
            np.add.at(Sw, bid, ww); np.add.at(Swrs, bid, ww * rs); np.add.at(Swrm, bid, ww * rm)
            np.add.at(Swrs2, bid, ww * rs * rs); np.add.at(Sw2, bid, ww * ww)
            if blk_keys is not None:      # bin by emulator R_blend (0 where target not scored -> isolated)
                posb = np.clip(np.searchsorted(blk_keys, key), 0, len(blk_keys) - 1)
                rb_row = np.where(blk_keys[posb] == key, blk_rb[posb], 0.0)
                rbb = np.digitize(rb_row, rbe)[m]
                np.add.at(RBw, rbb, ww); np.add.at(RBwrs, rbb, ww * rs); np.add.at(RBwrm, rbb, ww * rm)
                np.add.at(RBwrs2, rbb, ww * rs * rs)
            npair += int(m.sum())

    def summ(idx):
        sw = Sw[idx].sum(); swrs = Swrs[idx].sum(); swrm = Swrm[idx].sum()
        swrs2 = Swrs2[idx].sum()
        if sw <= 0:
            return (np.nan,) * 5
        Rs = swrs / sw; Rm = swrm / sw
        # Each (case,target) is ONE independent r_sim (its pairs share the same measured shape),
        # weighted to sum to 1 -> effective independent N = Sw = #unique (case,target).
        neff = sw
        var_rs = max(swrs2 / sw - Rs ** 2, 0.0)
        rs_se = np.sqrt(var_rs / max(neff, 1.0))
        return Rs, Rm, rs_se, neff, sw

    allb = np.arange(nb)
    Rs, Rm, rs_se, neff, sw = summ(allb)
    print(f"model={os.path.basename(args.measurement_model)}  |g|={g}  N_pairs={npair:,}  N_eff={sw:.0f}  neff(var)={neff:.0f}")
    print(f"\nGLOBAL (incoherent, weighted):  R_sim={Rs:.4f}+/-{rs_se:.4f}  R_model={Rm:.4f}  "
          f"m = R_sim/R_model-1 = {Rs/Rm-1:+.2%} +/- {rs_se/abs(Rm):.2%}")
    print(f"\n{'bin':>10} {'R_sim':>8} {'R_model':>8} {'m':>9} {'+/-':>7} {'N_eff':>10}")
    for c in range(nb):
        rs, rm, se, ne, w_ = summ([c])
        if not np.isfinite(rs) or w_ < 100:
            continue
        tag = "ISOLATED" if c == 0 else f"nbdist b{c}"
        print(f"{tag:>10} {rs:>8.4f} {rm:>8.4f} {rs/rm-1:>+9.2%} {se/abs(rm):>7.2%} {w_:>10.0f}")
    # blended-only global (exclude isolated bin 0, which the 7-arcsec flow extrapolates badly)
    Rs2, Rm2, se2, ne2, sw2 = summ(np.arange(1, nb))
    print(f"\nBLENDED-ONLY (excl. isolated):  R_sim={Rs2:.4f}  R_model={Rm2:.4f}  "
          f"m = {Rs2/Rm2-1:+.2%} +/- {se2/abs(Rm2):.2%}  (N_eff={sw2:.0f})")

    if blk_keys is not None:
        def summ_rb(idx):
            sw = RBw[idx].sum(); swrs = RBwrs[idx].sum(); swrm = RBwrm[idx].sum(); swrs2 = RBwrs2[idx].sum()
            if sw <= 0:
                return (np.nan,) * 4
            Rs = swrs / sw; Rm = swrm / sw
            var = max(swrs2 / sw - Rs ** 2, 0.0)
            return Rs, Rm, np.sqrt(var / max(sw, 1.0)), sw
        labels = ["ISO(Rbl~0)"] + [f"[{rbe[i-1]:.2f},{rbe[i]:.2f})" for i in range(1, len(rbe))] + [f">={rbe[-1]:.2f}"]
        print(f"\n--- INCOHERENT binned by emulator R_blend (self-response only; R_sim SHOULD ~= R_model "
              f"in every bin if the flow is calibrated for that population) ---")
        print(f"{'Rblend-bin':>12} {'R_sim':>8} {'R_model':>8} {'m':>9} {'+/-':>7} {'N_eff':>10}")
        for c in range(nrb):
            rs, rm, se, w_ = summ_rb([c])
            if not np.isfinite(rs) or w_ < 100:
                continue
            print(f"{labels[c]:>12} {rs:>8.4f} {rm:>8.4f} {rs/rm-1:>+9.2%} {se/abs(rm):>7.2%} {w_:>10.0f}")


if __name__ == "__main__":
    main()
