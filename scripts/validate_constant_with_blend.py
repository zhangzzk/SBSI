"""Constant-shear (COHERENT) gold validation WITH the blend-emulator correction.

The flow models the incoherent SELF-response (primary's own shear + neighbours' flux, NOT
neighbours' shear). The constant render is coherent, so its response is higher. Fix (user):
add the BlendEMU blending-response emulator linearly, summed over neighbours:

  R_total(gal) = R_flow(self, flow)  +  R_blend(gal, emulator summed over neighbours)
  m = R_sim / R_total - 1        (should collapse the bare-flow +55% toward sub-percent)

R_flow and R_sim reuse validate_constant_response's machinery; R_blend comes from
BlendingPredictor.predict_response on the constant INPUT galaxies (keyed by input_index).
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch


def _seed_flow(seed):
    """Reseed the global torch RNG so the flow's latent draws are reproducible.
    Called with the SAME seed before the +g and -g legs -> Common Random Numbers:
    both legs draw identical z, so the flow-sampling noise cancels in the
    (mp - mm) response difference instead of being amplified 1/(2g)=25x. Also
    removes serial-position dependence (R_flow no longer depends on how many
    seeds were harvested before it in the same process)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model  # noqa
from sbs_shear.preprocessing import (  # noqa
    DEFAULT_SELECTION_CUTS,
    apply_structure_measurement_noise,
    source_select_selection,
)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa
from scripts.response_ratio_diagnostic import model_mean_proj  # noqa
from sbs_shear.paths import CONST_SIM_BASE as CBASE

BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)


def r_blend_lookup(input_feather):
    """R_blend per input_index: emulator delta_et/gamma summed over neighbours (7in,k=30)."""
    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred = BlendingPredictor.load(BLEND_MODELS, tag="lsst_r", conditions=COND, device="cpu")
    t = pf.read_table(input_feather).to_pandas()
    t = t.rename(columns={c: c.replace("_input", "") for c in t.columns})
    reg = pred.predict_response(t, t)                      # all gals are both primaries and neighbour pool
    pk = [c for c in reg.columns if c.startswith("index")][0]
    rb = reg.groupby(pk)["response"].sum()
    print(f"  R_blend: {len(rb):,} gals with neighbours, mean={rb.mean():.4f}")
    return rb.to_dict()                                     # {index_input: R_blend}


def _case_filtered_table(path, columns, min_case=None, max_case=None, max_rows=0):
    """Read selected IPC columns while retaining only a bounded case range.

    Constgold is stored in case order.  Filtering each Arrow batch before conversion to pandas
    keeps case-sharded evaluations below the host-memory limit of the CIP GPU slices.  With no
    case bounds this is equivalent to the former whole-file reader.  ``max_rows=0`` means all
    matching rows; a positive limit is applied *after* the case filter.
    """
    parts = []
    n = 0
    with ipc.open_file(pa.memory_map(path)) as reader:
        available = set(reader.schema.names)
        use = [c for c in columns if c in available]
        if (min_case is not None or max_case is not None) and "case" not in use:
            raise ValueError(f"case filtering requested but {path} has no 'case' column")
        for bi in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(bi)]).select(use)
            if min_case is not None:
                table = table.filter(pc.greater_equal(table["case"], pa.scalar(min_case)))
            if max_case is not None:
                table = table.filter(pc.less(table["case"], pa.scalar(max_case)))
            if not table.num_rows:
                continue
            if max_rows > 0:
                remaining = max_rows - n
                if remaining <= 0:
                    break
                table = table.slice(0, remaining)
            parts.append(table)
            n += table.num_rows
            if max_rows > 0 and n >= max_rows:
                break
    if not parts:
        return pa.table({c: pa.array([]) for c in use})
    return pa.concat_tables(parts)


def _ipc_schema_names(path):
    """Inspect an Arrow/Feather schema without materialising the file."""
    with ipc.open_file(pa.memory_map(path)) as reader:
        return set(reader.schema.names)


def load(cat, max_rows, min_case=None, max_case=None):
    need = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "neighbored", "distance", "input_index", "case", "polarization_angle",
            "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
            "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
            "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s"]
    df = _case_filtered_table(
        cat, need, min_case=min_case, max_case=max_case, max_rows=max_rows
    ).to_pandas()
    if not len(df):
        raise ValueError(f"no catalogue rows in requested case range [{min_case}, {max_case})")
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                 df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i; df["e2_input_rot0_p"] = e2i
    df["gamma1_input_p"] = 0.0; df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    return df


def boot_m_err(rs, rb, Rf, cases, n_boot, seed=0):
    """Per-case bootstrap std of m = <r_sim>/(R_flow + <rb>) - 1 (R_flow held deterministic)."""
    uc = np.unique(cases)
    per = {c: (float(rs[cases == c].sum()), float(rb[cases == c].sum()), int((cases == c).sum())) for c in uc}
    rng = np.random.default_rng(seed); mb = []
    for _ in range(n_boot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        srs = sum(per[c][0] for c in pick); srb = sum(per[c][1] for c in pick); sn = sum(per[c][2] for c in pick)
        if sn:
            mb.append((srs / sn) / (Rf + srb / sn) - 1)
    return float(np.std(mb)) if mb else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", required=True)
    ap.add_argument("--catalogue", default=CBASE + "constant_response_catalogue_train.feather")
    ap.add_argument("--input-feather", default=CBASE + "case0_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather")
    ap.add_argument("--blend-lookup", default="results/blend_lookup_const_c0-39.feather",
                    help="prebuilt per-(case,input_index) R_blend feather; falls back to on-the-fly one-tile if missing")
    ap.add_argument("--crowd-flux-lookup", default=None,
                    help="per-(case,input_index) shell-flux feather; merges nbr_flux_near/far so a crowd_flux flow can condition on them")
    ap.add_argument("--meas-prim-lookup", default=None,
                    help="per-(case,input_index) MEASURED primary observables (build_meas_prim_lookup.py); "
                         "merges measured_mag_auto/flux_radius/class_star so a g0_meas_* realistic flow can condition on them")
    ap.add_argument("--ood-lookup", default=None,
                    help="per-(case,input_index) OUT-OF-DOMAIN neighbour flux (build_ood_lookup.py); enables --max-ood")
    ap.add_argument("--max-ood", type=float, default=None,
                    help="keep only galaxies with ood_flux < this (all neighbours in the emulator domain). Isolates the r<28/Re-cut effect (Option 1).")
    ap.add_argument("--ood-col", default="ood_flux",
                    help="which OOD-flux column to cut on (ood_flux | ood_flux_bright | ood_flux_faint from build_ood_split_lookup.py)")
    ap.add_argument("--nn-lookup", default=None,
                    help="per-(case,input_index) TRUTH nearest-neighbour distance (build_nn_distance_lookup.py); "
                         "enables the truth-isolation self-response table (blend-free by construction, no emulator).")
    ap.add_argument("--nn-radii", type=float, nargs="+", default=[0.0, 3.0, 5.0, 7.0, 10.0],
                    help="isolation radii (arcsec): rows report m for galaxies with nearest neighbour > R.")
    ap.add_argument("--nn-col", default="nn_dist_any", choices=["nn_dist_any", "nn_dist_bright"],
                    help="which truth-NN distance to isolate on (any-magnitude neighbour, or nearest brighter one).")
    ap.add_argument("--n-boot", type=int, default=200, help="per-case bootstrap resamples for the m error bar")
    ap.add_argument("--noise-photoz", type=float, default=0.0,
                    help="photo-z scatter sigma=this*(1+z) on redshift_input_p; MUST match the trained model")
    ap.add_argument("--noise-sersic-frac", type=float, default=0.0,
                    help="fractional scatter sigma=this*|n| on sersic_n_input_p; MUST match the trained model")
    ap.add_argument("--min-case", type=int, default=None, help="keep only case >= this (held-out validation split)")
    ap.add_argument("--max-case", type=int, default=None, help="keep only case < this (fit split; pairs with --min-case)")
    ap.add_argument("--no-blend", action="store_true", help="R_total = R_flow only (flow carries the full coherent response)")
    ap.add_argument("--global-only", action="store_true",
                    help="print only the GLOBAL R_flow / R_sim / R_blend / m and return, skipping all per-bin "
                         "diagnostic tables. For seed-ensemble certification: only R_flow varies across seeds "
                         "(R_sim, R_blend are seed-independent), so the per-bin tables are redundant per seed.")
    ap.add_argument("--n-dist", type=int, default=3)
    ap.add_argument("--n-blend", type=int, default=4, help="quantile bins of R_blend (above eps)")
    ap.add_argument("--blend-eps", type=float, default=0.02, help="R_blend below this = truly-isolated bin")
    ap.add_argument("--rblend-edges-npz", default=None,
                    help="optional response-target npz; if present, also bin constant-gold by its edges_crowd")
    ap.add_argument("--max-rows", type=int, default=6_000_000)
    ap.add_argument("--dump", default=None,
                    help="if set, write per-object (case,input_index,r_input_p,r_sim,R_flow,R_blend,"
                         "neighbored,distance) feather for offline binning by any covariate")
    ap.add_argument("--mult-lookup", default=None,
                    help="optional per-(case,input_index) multiplicity feather (build_blend_multiplicity.py: "
                         "n_pairs,rb_max,rb_top2); enables the multiplicity discriminator table")
    ap.add_argument("--fit-rblend-corr", default=None,
                    help="FIT a smooth deficit(R_blend) recalibration on THIS run and write it to this npz "
                         "(centers,deficit). Run on cases 0-39, then --apply on held-out 40-79 to validate.")
    ap.add_argument("--apply-rblend-corr", default=None,
                    help="APPLY a previously-fit deficit(R_blend) npz: adds interp(deficit, R_blend) to the "
                         "additive R_blend term before m. Out-of-sample use is the real test of the fix.")
    ap.add_argument("--corr-nbin", type=int, default=24, help="number of R_blend quantile bins for the fit")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--flow-seed", type=int, default=12345,
                    help="fixed torch seed used (Common Random Numbers) for BOTH the +g and -g "
                         "R_flow legs so their flow-sampling noise cancels in the response difference.")
    ap.add_argument("--device", default=None)
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)

    df = load(args.catalogue, args.max_rows, min_case=args.min_case, max_case=args.max_case)
    if args.noise_photoz or args.noise_sersic_frac:
        # feed the flow the SAME noisy structure channel it was trained on (seed fixed for reproducibility)
        apply_structure_measurement_noise(
            df, photoz_sigma=args.noise_photoz, sersic_frac=args.noise_sersic_frac, seed=12345)
        print(f"structure noise: photoz_sigma={args.noise_photoz} sersic_frac={args.noise_sersic_frac}")
    if args.min_case is not None or args.max_case is not None:
        print(f"case range [{args.min_case}, {args.max_case}) -> N={len(df):,}")
    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    gh1 = df["applied_g1"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"])
    gh2 = df["applied_g2"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"])
    e1p, e2p = df["measured_e1_plus"].to_numpy(float), df["measured_e2_plus"].to_numpy(float)
    e1m, e2m = df["measured_e1_minus"].to_numpy(float), df["measured_e2_minus"].to_numpy(float)
    r_sim_i = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / (2 * g)
    # per-galaxy R_blend matched by (case, input_index); 0 where absent (r>26 / isolated / not scored)
    if os.path.exists(args.blend_lookup):
        rbdf = _case_filtered_table(
            args.blend_lookup, ["case", "input_index", "R_blend"],
            min_case=args.min_case, max_case=args.max_case,
        ).to_pandas()
        merged = df[["case", "input_index"]].merge(rbdf, on=["case", "input_index"], how="left")
        rb_i = merged["R_blend"].fillna(0.0).to_numpy(float)
        print(f"R_blend lookup: {args.blend_lookup}  matched {np.mean(merged['R_blend'].notna()):.1%} of rows, "
              f"{rbdf['case'].nunique()} cases")
    else:
        print(f"WARNING blend-lookup {args.blend_lookup} missing -> on-the-fly one-tile (input_index only)")
        rb = r_blend_lookup(args.input_feather)
        rb_i = df["input_index"].map(lambda k: rb.get(int(k), 0.0)).to_numpy(float)

    # expose crowding columns for crowd_* flows (self-response conditioning; NOT the additive R_blend term)
    df["r_blend"] = rb_i
    if args.crowd_flux_lookup and os.path.exists(args.crowd_flux_lookup):
        fcols = ["nbr_flux_near", "nbr_flux_far"]
        if "nbr_flux_max" in _ipc_schema_names(args.crowd_flux_lookup):
            fcols = fcols + ["nbr_flux_max"]   # flux-concentration feature for the conc flow
        cf = _case_filtered_table(
            args.crowd_flux_lookup, ["case", "input_index", *fcols],
            min_case=args.min_case, max_case=args.max_case,
        ).to_pandas()
        cm = df[["case", "input_index"]].merge(cf, on=["case", "input_index"], how="left")
        for fc in fcols:
            df[fc] = cm[fc].fillna(0.0).to_numpy(float)
        print(f"crowd-flux lookup: matched {np.mean(cm['nbr_flux_near'].notna()):.1%} of rows; cols={fcols}")

    # MEASURED PRIMARY observables for realistic g0_meas_* flows (WORKLOG cont.19). Do NOT fillna(0.0):
    # 0.0 is a valid-looking garbage magnitude; leave NaN so the preprocessor applies the TRAINED
    # fill_values + __is_missing indicator (consistent with training). Averaged over +/- signs already.
    if args.meas_prim_lookup and os.path.exists(args.meas_prim_lookup):
        mpf = pf.read_table(args.meas_prim_lookup).to_pandas()
        mcols = [c for c in mpf.columns if c.startswith("measured_")]
        mm2 = df[["case", "input_index"]].merge(mpf[["case", "input_index", *mcols]],
                                                on=["case", "input_index"], how="left")
        for mc in mcols:
            df[mc] = mm2[mc].to_numpy(float)          # keep NaN where unmatched (missing-indicator)
        match = float(np.mean(mm2["measured_mag_auto"].notna())) if "measured_mag_auto" in mcols else 0.0
        print(f"meas-prim lookup: matched {match:.1%} of rows; cols={mcols}")
        if match < 0.95:
            print(f"  !! WARNING meas-prim match {match:.1%} < 95% -> many __is_missing rows (OOD); m may be biased")

    # TRUTH nearest-neighbour distance (emulator-independent isolation); NaN where unmatched -> excluded
    if args.nn_lookup and os.path.exists(args.nn_lookup):
        nnl = pf.read_table(args.nn_lookup).to_pandas()[["case", "input_index", "nn_dist_any", "nn_dist_bright"]]
        nm = df[["case", "input_index"]].merge(nnl, on=["case", "input_index"], how="left")
        df["nn_dist_any"] = nm["nn_dist_any"].to_numpy(float)
        df["nn_dist_bright"] = nm["nn_dist_bright"].to_numpy(float)
        print(f"nn lookup: matched {np.mean(nm['nn_dist_any'].notna()):.1%} of rows")

    # OOD neighbour flux (bright/faint split) for the mechanism-B binning diagnostic below; NaN->0
    if args.ood_lookup and os.path.exists(args.ood_lookup):
        odf = pf.read_table(args.ood_lookup).to_pandas()
        ocols = [c for c in ("ood_flux", "ood_flux_bright", "ood_flux_faint") if c in odf.columns]
        om = df[["case", "input_index"]].merge(odf[["case", "input_index"] + ocols], on=["case", "input_index"], how="left")
        for c in ocols:
            df[c] = om[c].fillna(0.0).to_numpy(float)
        print(f"ood lookup: cols={ocols}  matched {np.mean(om[ocols[0]].notna()):.1%} of rows")

    # Option 1: keep only galaxies whose neighbours are ALL in the emulator domain (ood_flux < thresh)
    if args.ood_lookup and args.max_ood is not None and args.ood_col in df.columns:
        ood = df[args.ood_col].to_numpy(float)
        keep = ood < args.max_ood
        print(f"OOD cut [{args.ood_col}]: max_ood={args.max_ood} keeps {keep.mean():.1%} ({int(keep.sum()):,}) of {len(df):,} "
              f"(<ood>={np.mean(ood):.3f})")
        df = df[keep].reset_index(drop=True)
        gh1, gh2 = gh1[keep], gh2[keep]
        r_sim_i = r_sim_i[keep]; rb_i = rb_i[keep]

    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(), df["e2_input_rot0_p"].to_numpy(float).copy())
    # Common Random Numbers: reseed to the SAME value before each leg so both draw
    # identical flow latents -> the sampling noise cancels in (mp - mm) rather than
    # being amplified 1/(2g)~25x. Makes R_flow deterministic + position-independent.
    _seed_flow(args.flow_seed); mp, _, projp = model_mean_proj(bundle, df, +g, gh1, gh2, intr, rk, args.n_samples, args.batch_size, return_proj=True)
    _seed_flow(args.flow_seed); mm, _, projm = model_mean_proj(bundle, df, -g, gh1, gh2, intr, rk, args.n_samples, args.batch_size, return_proj=True)
    R_flow = (mp - mm) / (2 * g)                  # global scalar (unchanged; identically == mean of R_flow_perobj)
    R_flow_perobj = (projp - projm) / (2 * g)     # per-object CRN self-response (for the --dump; global is its mean)
    rb_add = np.zeros_like(rb_i) if args.no_blend else rb_i   # additive blend term (0 if flow carries full response)
    if args.apply_rblend_corr:                                # OUT-OF-SAMPLE test: add the fit deficit(R_blend)
        cz = np.load(args.apply_rblend_corr)
        corr = np.interp(rb_i, cz["centers"], cz["deficit"])  # clamped at ends by np.interp
        rb_add = rb_add + corr
        print(f"applied R_blend correction {args.apply_rblend_corr}: <corr>=+{corr.mean():.4f} "
              f"(range {corr.min():+.4f}..{corr.max():+.4f}, {len(cz['centers'])} pts)", flush=True)
    if args.dump:
        pd.DataFrame(dict(case=df["case"].to_numpy(np.int64),
                          input_index=df["input_index"].to_numpy(np.int64),
                          r_input_p=df["r_input_p"].to_numpy(float),
                          r_sim=r_sim_i, R_flow=R_flow_perobj, R_blend=rb_add,
                          neighbored=df["neighbored"].astype(bool).to_numpy(),
                          distance=df["distance"].to_numpy(float))).to_feather(args.dump)
        print(f"per-object dump -> {args.dump} ({len(df):,} rows)", flush=True)
    R_sim = float(np.mean(r_sim_i)); R_blend = float(np.mean(rb_add))
    # per-case bootstrap error on m (resample the constant cases; R_flow held at its global value -- it is
    # model-deterministic, so the sampling error is dominated by measured R_sim and the emulator R_blend)
    case_arr = df["case"].to_numpy(np.int64); ucases = np.unique(case_arr)
    per = {c: (float(r_sim_i[case_arr == c].sum()), float(rb_add[case_arr == c].sum()), int((case_arr == c).sum()))
           for c in ucases}
    rng = np.random.default_rng(0); mb = []
    for _ in range(args.n_boot):
        pick = rng.choice(ucases, size=len(ucases), replace=True)
        srs = sum(per[c][0] for c in pick); srb = sum(per[c][1] for c in pick); sn = sum(per[c][2] for c in pick)
        mb.append((srs / sn) / (R_flow + srb / sn) - 1)
    m_err = float(np.std(mb))
    print(f"\nmodel={os.path.basename(args.measurement_model)}  |g|={g:.4f}  N={len(df):,}  n_cases={len(ucases)}  blended={df['neighbored'].astype(bool).mean():.3f}")
    print(f"GLOBAL:  R_sim={R_sim:.4f}   R_flow(self)={R_flow:.4f}   R_blend(emulator)={R_blend:.4f}   R_total={R_flow+R_blend:.4f}")
    print(f"  bare-flow  m = R_sim/R_flow - 1              = {R_sim/R_flow-1:+.2%}")
    print(f"  WITH blend m = R_sim/(R_flow+R_blend) - 1    = {R_sim/(R_flow+R_blend)-1:+.2%} +/- {m_err:.2%}")

    if args.global_only:
        print("GLOBAL_ONLY: skipping per-bin diagnostic tables.", flush=True)
        return

    # --- FIT a smooth deficit(R_blend) recalibration on THIS sample: per fine R_blend quantile bin,
    #     proper per-bin R_flow (model_mean_proj) and deficit = <R_sim> - R_flow - <R_blend>. Save the
    #     curve; apply to held-out cases 40-79 to test whether the residual is a transferable function
    #     of blend strength (=> reachable within framework) or covariate-shift noise (=> not). ---
    if args.fit_rblend_corr:
        nb = args.corr_nbin
        qe2 = np.quantile(rb_i, np.linspace(0, 1, nb + 1)); qe2[0] -= 1e-9; qe2[-1] += 1e-9
        fb = np.clip(np.digitize(rb_i, qe2) - 1, 0, nb - 1)
        centers, deficit = [], []
        print(f"\n--- FIT deficit(R_blend) recalibration ({nb} quantile bins) ---")
        print(f"  {'<R_bl>':>8} {'<R_sim>':>8} {'R_flow':>8} {'deficit':>8} {'N':>9}")
        for c in range(nb):
            m = fb == c
            if m.sum() < 2000:
                continue
            sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
            mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_i[m]))
            centers.append(Rb); deficit.append(Rs - Rf - Rb)
            print(f"  {Rb:>8.4f} {Rs:>8.4f} {Rf:>8.4f} {Rs-Rf-Rb:>+8.4f} {int(m.sum()):>9,}")
        centers = np.asarray(centers); deficit = np.asarray(deficit)
        np.savez(args.fit_rblend_corr, centers=centers, deficit=deficit)
        corr = np.interp(rb_i, centers, deficit)
        print(f"saved -> {args.fit_rblend_corr} ({len(centers)} pts); "
              f"in-sample corrected m = {R_sim/(R_flow+float(np.mean(rb_add+corr)))-1:+.2%} (trivially ~0 by construction)")
        print("FIT_ONLY: skipping diagnostic tables.", flush=True)
        return

    # --- MAGNITUDE-binned self-response calibration: proper R_flow via model_mean_proj per target-mag
    #     subset. ISOLATED (R_blend<eps) is blend-free -> pure flow test; does the flow reproduce the
    #     steep faint-end drop of the true self-response? ALL shows the full R_flow+R_blend closure. ---
    rp = df["r_input_p"].to_numpy(float)
    mag_edges = np.array([18, 24, 24.5, 25, 25.5, 26, 26.5, 27, 28.1])
    iso_mask = rb_i < args.blend_eps
    for scope, smask in [("ISOLATED (blend-free -> pure flow)", iso_mask), ("ALL", np.ones(len(df), bool))]:
        print(f"\n--- by target mag r_p [{scope}] ---")
        print(f"{'r_p bin':>11} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'m_bare':>8} {'m_blend':>8} {'N':>9}")
        for i in range(len(mag_edges) - 1):
            m = smask & (rp >= mag_edges[i]) & (rp < mag_edges[i + 1])
            if m.sum() < 5000:
                continue
            sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
            mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
            print(f"{mag_edges[i]:.1f}-{mag_edges[i+1]:.1f}".rjust(11) +
                  f" {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {Rs/Rf-1:>+8.1%} {Rs/(Rf+Rb)-1:>+8.1%} {int(m.sum()):>9,}")

    nbf = df["neighbored"].astype(bool).to_numpy(); dist = df["distance"].to_numpy(float)
    db = dist[nbf & np.isfinite(dist)]; ed = np.quantile(db, np.linspace(0, 1, args.n_dist + 1)); ed[0] -= 1e-6; ed[-1] += 1e-6
    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, args.n_dist - 1), 0)
    print(f"\n{'blend':>10} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'m_bare':>8} {'m_blend':>8} {'N':>9}")
    for c in range(args.n_dist + 1):
        m = di == c
        if m.sum() < 5000:
            continue
        sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
        mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
        mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
        Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
        tag = "ISOLATED" if c == 0 else f"blend d{c}"
        print(f"{tag:>10} {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {Rs/Rf-1:>+8.1%} {Rs/(Rf+Rb)-1:>+8.1%} {int(m.sum()):>9,}")

    # --- THE CLEAN DIAGNOSTIC: bin by emulator R_blend, NOT the 3" neighbored flag.
    #     Bin 0 = truly-isolated (R_blend<eps): there R_sim should EQUAL R_flow (pure
    #     self-response, zero coherent blend). If m_bare~0 there -> flow is fine and the
    #     residual lives entirely in the blend/emulator; if R_sim>R_flow -> the flow's
    #     self-response is the culprit. ---
    eps = args.blend_eps
    hi = rb_i >= eps
    print(f"\n--- binned by emulator R_blend (eps={eps}; truly-isolated frac={np.mean(~hi):.3f}) ---")
    print(f"{'Rblend-bin':>12} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'m_bare':>8} {'m_blend':>8} {'+/-':>6} {'N':>9}")
    if hi.sum():
        qe = np.quantile(rb_i[hi], np.linspace(0, 1, args.n_blend + 1)); qe[0] -= 1e-9; qe[-1] += 1e-9
        bi = np.where(~hi, 0, 1 + np.clip(np.digitize(rb_i, qe) - 1, 0, args.n_blend - 1))
    else:
        bi = np.zeros(len(rb_i), int)
    for c in range(args.n_blend + 1):
        m = bi == c
        if m.sum() < (500 if c == 0 else 5000):
            continue
        sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
        mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
        mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
        Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
        tag = "ISO(Rbl~0)" if c == 0 else f"Rbl q{c}"
        err = boot_m_err(r_sim_i[m], rb_add[m], Rf, case_arr[m], args.n_boot)
        print(f"{tag:>12} {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {Rs/Rf-1:>+8.1%} {Rs/(Rf+Rb)-1:>+8.1%} {err:>6.1%} {int(m.sum()):>9,}")

    # --- CASE-SPLIT replication: does the moderate-blend residual reproduce across independent case
    #     subsets, or is it a few-case fluke? Split cases into two halves (low idx vs high idx) and print
    #     m_blend per R_blend quantile for each half. Concordant halves -> real & sample-stable. ---
    uc_sorted = np.sort(ucases); half = uc_sorted[len(uc_sorted) // 2]
    for hlabel, hmask in [(f"cases<{half}", case_arr < half), (f"cases>={half}", case_arr >= half)]:
        print(f"\n--- R_blend quantiles [{hlabel}, N={int(hmask.sum()):,}] ---")
        print(f"{'Rblend-bin':>12} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'m_blend':>8} {'+/-':>6} {'N':>9}")
        for c in range(args.n_blend + 1):
            m = (bi == c) & hmask
            if m.sum() < (500 if c == 0 else 5000):
                continue
            sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
            mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
            err = boot_m_err(r_sim_i[m], rb_add[m], Rf, case_arr[m], args.n_boot)
            tag = "ISO(Rbl~0)" if c == 0 else f"Rbl q{c}"
            print(f"{tag:>12} {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {Rs/(Rf+Rb)-1:>+8.1%} {err:>6.1%} {int(m.sum()):>9,}")

    # --- MULTIPLICITY DISCRIMINATOR: does the moderate-blend deficit come from the per-pair SUM
    #     under-counting many-neighbour objects (R_blend too low) or from the flow (R_flow too low)?
    #     Within each R_blend quantile bin, split by neighbour count n_pairs (low/high) and by whether
    #     one neighbour dominates (rb_max/R_blend). Proper per-cell R_flow via model_mean_proj.
    #       deficit GROWS with n_pairs at fixed R_blend -> per-pair sum under-counts (super-additive) -> fix R_blend.
    #       deficit FLAT in n_pairs, present for one-dominant too   -> flow crowd-conditioning -> fix R_flow. ---
    if args.mult_lookup and os.path.exists(args.mult_lookup):
        ml = pf.read_table(args.mult_lookup).to_pandas()[["case", "input_index", "n_pairs", "rb_max"]]
        mm = df[["case", "input_index"]].merge(ml, on=["case", "input_index"], how="left")
        npair = mm["n_pairs"].fillna(0).to_numpy(float)
        rbmax = mm["rb_max"].fillna(0.0).to_numpy(float)
        domfrac = np.where(rb_i > 1e-9, np.abs(rbmax) / np.maximum(np.abs(rb_i), 1e-9), 0.0)
        print(f"\n--- MULTIPLICITY discriminator (n_pairs & dominance within R_blend quantiles) ---")
        print(f"  matched n_pairs for {np.mean(mm['n_pairs'].notna()):.1%} of rows; "
              f"blended <n_pairs>={npair[hi].mean():.2f}")

        def _cell(m, tag):
            if m.sum() < 5000:
                return
            sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
            mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
            err = boot_m_err(r_sim_i[m], rb_add[m], Rf, case_arr[m], args.n_boot)
            print(f"{tag:>22} {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {Rs/Rf-1:>+8.1%} {Rs/(Rf+Rb)-1:>+8.1%} "
                  f"{err:>6.1%} {npair[m].mean():>5.1f} {int(m.sum()):>9,}")

        print(f"{'cell':>22} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'m_bare':>8} {'m_blend':>8} {'+/-':>6} {'<np>':>5} {'N':>9}")
        for c in range(1, args.n_blend + 1):                       # skip ISO bin; multiplicity only meaningful when blended
            cb = bi == c
            if cb.sum() < 10000:
                continue
            nmed = np.median(npair[cb])
            _cell(cb & (npair <= nmed), f"q{c} n<={nmed:.0f}")
            _cell(cb & (npair > nmed), f"q{c} n>{nmed:.0f}")
            # dominance split: one neighbour carries >70% of |R_blend| vs spread over many
            _cell(cb & (domfrac >= 0.7), f"q{c} 1-dominant")
            _cell(cb & (domfrac < 0.7), f"q{c} many-small")

        # --- ORTHOGONALITY CONFIRM (does a flow retrain with n_pairs actually help?):
        #     The flow's ONLY crowding inputs are nbr_flux_near/far. Bin the blended set into TIGHT
        #     nbr_flux bins (fix what the flow sees), then within each bin split by n_pairs. If the
        #     flow deficit (= R_sim - R_flow - R_blend) rises with n_pairs AT MATCHED <flux>, that is
        #     information the flow structurally cannot access -> adding n_pairs as a conditioning
        #     feature is justified. If deficit is flat in n_pairs at fixed flux -> the flow already has
        #     the info (nbr_flux is sufficient) and a same-feature retrain, not a new feature, is the fix.
        if "nbr_flux_near" in df.columns:
            flux_tot = (df["nbr_flux_near"] + df["nbr_flux_far"]).to_numpy(float)
            NQF = 4
            print(f"\n--- ORTHOGONALITY confirm: flow deficit by nbr_flux(flow input) x n_pairs(candidate), blended set ---")
            print(f"{'cell':>22} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'deficit':>8} {'+/-':>7} {'<flux>':>7} {'<np>':>5} {'N':>9}")

            def _cell_def(m, tag):
                if m.sum() < 5000:
                    return None
                sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
                mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
                mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
                Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
                dfc = Rs - Rf - Rb
                err = boot_m_err(r_sim_i[m], rb_add[m], Rf, case_arr[m], args.n_boot) * (Rf + Rb)
                print(f"{tag:>22} {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {dfc:>+8.4f} {err:>7.4f} "
                      f"{flux_tot[m].mean():>7.2f} {npair[m].mean():>5.1f} {int(m.sum()):>9,}")
                return dfc

            fe = np.quantile(flux_tot[hi], np.linspace(0, 1, NQF + 1)); fe[0] -= 1e-9; fe[-1] += 1e-9
            for t in range(NQF):
                ft = hi & (flux_tot >= fe[t]) & (flux_tot < fe[t + 1])
                if ft.sum() < 12000:
                    continue
                nmed = np.median(npair[ft])
                dlo = _cell_def(ft & (npair <= nmed), f"fluxQ{t+1} n<={nmed:.0f}")
                dhi = _cell_def(ft & (npair > nmed), f"fluxQ{t+1} n>{nmed:.0f}")
                if dlo is not None and dhi is not None:
                    print(f"{'  -> d(deficit) nhi-nlo':>22} {dhi - dlo:>+8.4f}   (>0 => n_pairs adds info beyond nbr_flux)")

    # Same diagnostic, but with the exact crowding edges used by the response target.
    # This makes constant-gold residuals directly comparable to the trained loss bins.
    if args.rblend_edges_npz and os.path.exists(args.rblend_edges_npz):
        td = np.load(args.rblend_edges_npz)
        if "edges_crowd" in td.files:
            edges = np.asarray(td["edges_crowd"], dtype=float)
            tb = np.clip(np.digitize(rb_i, edges) - 1, 0, len(edges) - 2)
            print(f"\n--- binned by training R_blend edges from {os.path.basename(args.rblend_edges_npz)} ---")
            print(f"{'train-bin':>12} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'m_bare':>8} {'m_blend':>8} {'+/-':>6} {'N':>9}")
            for c in range(len(edges) - 1):
                m = tb == c
                if m.sum() < 5000:
                    continue
                sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
                mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
                mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
                Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
                err = boot_m_err(r_sim_i[m], rb_add[m], Rf, case_arr[m], args.n_boot)
                tag = f"edge {c}"
                print(f"{tag:>12} {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {Rs/Rf-1:>+8.1%} {Rs/(Rf+Rb)-1:>+8.1%} {err:>6.1%} {int(m.sum()):>9,}")

    # --- MECHANISM B: bin by OUT-OF-DOMAIN neighbour flux (bright vs faint). The emulator scores
    #     OOD neighbours as 0. If m_blend RISES with ood_flux_BRIGHT, bright uncounted neighbours are a
    #     REAL blend under-count (positive m). If m_blend is FLAT vs ood_flux_FAINT, faint neighbours
    #     carry ~0 blend and any cut-based m shift on them is a crowding-SELECTION artifact, not blend. ---
    for col in ("ood_flux_bright", "ood_flux_faint"):
        if col not in df.columns:
            continue
        ov = df[col].to_numpy(float); pos = ov > 1e-6
        print(f"\n--- binned by {col} (has-{col.split('_')[-1]}-ood frac={np.mean(pos):.3f}) ---")
        print(f"{'bin':>12} {'R_sim':>7} {'R_flow':>7} {'R_bl':>6} {'m_bare':>8} {'m_blend':>8} {'+/-':>6} {'N':>9}")
        if pos.sum() > 5000:
            qe = np.quantile(ov[pos], np.linspace(0, 1, 4)); qe[0] -= 1e-9; qe[-1] += 1e-9
            ob = np.where(~pos, 0, 1 + np.clip(np.digitize(ov, qe) - 1, 0, 2))
        else:
            ob = np.zeros(len(ov), int)
        for c in range(4):
            m = ob == c
            if m.sum() < (500 if c == 0 else 3000):
                continue
            sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
            mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m])); Rb = float(np.mean(rb_add[m]))
            tag = "no-ood" if c == 0 else f"ood q{c}"
            err = boot_m_err(r_sim_i[m], rb_add[m], Rf, case_arr[m], args.n_boot)
            print(f"{tag:>12} {Rs:>7.4f} {Rf:>7.4f} {Rb:>6.3f} {Rs/Rf-1:>+8.1%} {Rs/(Rf+Rb)-1:>+8.1%} {err:>6.1%} {int(m.sum()):>9,}")

    # --- TRUTH-isolation self-response (emulator-independent): keep galaxies whose nearest input
    #     neighbour is > R arcsec. As R grows these approach blend-free BY TRUTH (no coherent neighbour
    #     light in the aperture), so m_bare = R_sim/R_flow - 1 approaches the pure self-response residual
    #     of the flow on the COHERENT render. Two cuts: any-magnitude neighbour, and nearest BRIGHTER
    #     neighbour (fainter neighbours blend little, so this keeps far more galaxies at large R). ---
    nn_cols = [c for c in ("nn_dist_any", "nn_dist_bright")
               if c in df.columns and np.isfinite(df[c].to_numpy(float)).any()]
    for col in nn_cols:
        nn = df[col].to_numpy(float); finite = np.isfinite(nn)
        print(f"\n--- TRUTH isolation by {col} (no emulator): self-response residual vs radius "
              f"[median={np.median(nn[finite]):.2f}\", matched={finite.mean():.1%}] ---")
        print(f"{'cut':>12} {'R_sim':>7} {'R_flow':>7} {'m_bare':>8} {'+/-':>6} {'N':>9} {'frac':>6}")
        for R in args.nn_radii:
            m = finite & (nn > R)
            lab = f'{col.split("_")[-1]}>{R:g}"'
            if m.sum() < 500:
                print(f"{lab:>12} {'--':>7} {'--':>7} {'(N<500)':>8}")
                continue
            sub = df[m].reset_index(drop=True); intrc = (intr[0][m], intr[1][m])
            mpc, _ = model_mean_proj(bundle, sub, +g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            mmc, _ = model_mean_proj(bundle, sub, -g, gh1[m], gh2[m], intrc, rk, args.n_samples, args.batch_size)
            Rf = (mpc - mmc) / (2 * g); Rs = float(np.mean(r_sim_i[m]))
            err = boot_m_err(r_sim_i[m], np.zeros(int(m.sum())), Rf, case_arr[m], args.n_boot)
            print(f"{lab:>12} {Rs:>7.4f} {Rf:>7.4f} {Rs/Rf-1:>+8.1%} {err:>6.1%} {int(m.sum()):>9,} {m.mean():>6.3f}")


if __name__ == "__main__":
    main()
