"""[STATUS — see SBSI/STATE_OF_PLAY.md] The DEFAULT (forward, g=0.05) mode is CANONICAL: it builds the
isolated-separated snc self-response target used by the _prod winner (cont.60). The `--antithetic` mode
(added cont.63; reads the constgold +/-g antithetic render, i.e. the acceptance-harness truth) is
RETRACTED as train-on-validation circularity (cont.67) — do NOT use it to build an R_flow target; use
compute_deltaet_target.py (constgold never read) instead.

Blend-aware property-resolved response target R_sim(flux x size x blend) for Phase-2 training.

Extends compute_response_target.py with a THIRD binning axis: blend severity.
  blend bin 0          = ISOLATED (neighbored == False)
  blend bins 1..n_dist = BLENDED, split by neighbour `distance` quantiles (closer = severer)

Within each (true flux x true size x blend) cell we measure R_sim = <e_meas.ghat>/g from the
g=0.05 sample.  train_measurement_model.py consumes the 3D Rsim + edges to supervise the flow's
induced response per cell -- so it must reproduce the blended-vs-isolated (and severity)
response, which probe 2/3 showed it currently gets with the WRONG sign.

Memory-safe: streams only the needed columns (the feather is ~17GB).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    add_measurement_target_features, raw_columns_for_measurement_targets)
from sbs_shear import domain as sbs_domain  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402


def _selection_cuts(args):
    """DEFAULT_SELECTION_CUTS with the PRIMARY domain optionally narrowed (see the trainer's twin).

    `source_select_detection` reads only cuts[1] (primary true mag) and cuts[3] (primary true Re),
    so this restricts the primary and leaves neighbours full-population. Returns the unmodified
    defaults when neither flag is given, so the historical targets stay reproducible.
    """
    cuts = [list(c) for c in DEFAULT_SELECTION_CUTS]
    if getattr(args, "primary_mag_max", None) is not None:
        cuts[1][1] = float(args.primary_mag_max)
    if getattr(args, "primary_re_min", None) is not None:
        cuts[3][0] = float(args.primary_re_min)
    return cuts


def load_snc_lookup(path, cols):
    if not path:
        return None
    lookup = pf.read_table(path, columns=["case", "input_index", *cols]).to_pandas()
    key = lookup["case"].to_numpy(np.int64) * 1_000_003 + lookup["input_index"].to_numpy(np.int64)
    order = np.argsort(key)
    print(f"SNC lookup: {path}  rows={len(lookup):,}  cases={lookup['case'].nunique():,}")
    return {
        "key": key[order],
        "e1": lookup[cols[0]].to_numpy(float)[order],
        "e2": lookup[cols[1]].to_numpy(float)[order],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--n-flux", type=int, default=6)
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--size-edges", default=None,
                    help="comma-separated explicit size-bin edges (overrides --n-size quantile edges). "
                         "Use to add resolution where the response gradient is steep (e.g. large sizes), "
                         "which pure quantile spacing under-resolves. A-priori physics choice, not |m|-tuning.")
    ap.add_argument("--flux-edges", default=None,
                    help="comma-separated explicit flux(true-mag)-bin edges (overrides --n-flux quantile "
                         "edges). Same rationale as --size-edges: equal-COUNT spacing equalises the "
                         "absolute response step per cell, but `m` is a RATIO, and the relative step "
                         "d(log R) grows toward faint. Placing edges on equal d(log R) instead puts "
                         "resolution where the response changes fastest in the units the metric uses. "
                         "A-priori choice made on the half-shear ruler, NOT |m|-tuning.")
    ap.add_argument("--n-dist", type=int, default=3, help="distance bins for BLENDED gals (isolated is a separate bin 0)")
    ap.add_argument("--crowd-col", default=None,
                    help="if set (e.g. r_blend / nbr_flux_near), use QUANTILE bins of this crowding "
                         "column as the 3rd axis instead of neighbour distance. Equal-count bins keep "
                         "every cell well-sampled; no separate isolated bin (isolated = low crowding).")
    ap.add_argument("--n-crowd", type=int, default=5, help="number of crowding quantile bins")
    ap.add_argument("--target-cols", nargs=2, default=["measured_e1_image", "measured_e2_image"],
                    help="the two measured-shape columns whose response to compute "
                         "(e.g. measured_ngmix_g1 measured_ngmix_g2).")
    ap.add_argument("--min-count", type=float, default=200,
                    help="cells with fewer EFFECTIVE (unique-target, sum-of-weights) galaxies fall "
                         "back to the global R. Raise for noisy estimators like ngmix. NOTE: for "
                         "all-pairs catalogues this is the effective (per-target) count, ~n_pairs "
                         "smaller than the raw pair-row count.")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Read at most this many raw rows (0=all). For huge all-pairs catalogues "
                         "the target is computed on a memory-fitting case subset (still ample stats).")
    ap.add_argument("--min-case", type=int, default=None,
                    help="Keep only case >= min-case. With --max-case this isolates a case WINDOW, "
                         "which is how the target's population (half-shear cases 0-99) is tested "
                         "against constgold's (cases 40-139): rebuild on the 40-99 overlap and see "
                         "whether the target moves.")
    ap.add_argument("--max-case", type=int, default=None,
                    help="Keep only case <= max-case. Useful when an SNC lookup covers a case subset.")
    ap.add_argument("--snc-lookup", default=None,
                    help="Optional g=0 per-(case,input_index) lookup. If set, target uses "
                         "[e(g)-e(0)].ghat/g instead of raw e(g).ghat/g.")
    ap.add_argument("--snc-cols", nargs=2, default=["ngmix0_g1", "ngmix0_g2"],
                    help="Two shape columns in --snc-lookup.")
    ap.add_argument("--antithetic", action="store_true",
                    help="METRIC-CONSISTENT mode: read a constant-shear ANTITHETIC (+g/-g) render "
                         "catalogue and compute R = <(e_+g - e_-g).ghat>/(2g) -- the SAME sample and "
                         "central-diff scheme as the acceptance metric's r_sim (infer_posterior_shape). "
                         "Uses applied_g1/g2 for ghat; ignores --target-cols and --snc-lookup; "
                         "g = --nominal-g (set to the render |g|, e.g. 0.02).")
    ap.add_argument("--anti-cols", nargs=4,
                    default=["measured_e1_plus", "measured_e2_plus",
                             "measured_e1_minus", "measured_e2_minus"],
                    help="Four +/-g shape columns for --antithetic: e1_plus e2_plus e1_minus e2_minus.")
    ap.add_argument("--primary-mag-max", type=float, default=None,
                    help="build the target on primaries with true mag below this (deliverable "
                         "domain: 26.0). MUST match the trainer's --primary-mag-max, otherwise the "
                         "pin supervises the kept rows with a target measured on rows the trainer "
                         "never sees -- see WORKLOG 2026-07-27d.")
    ap.add_argument("--primary-re-min", type=float, default=None,
                    help="build the target on primaries with true Re above this (deliverable "
                         "domain: 0.3). MUST match the trainer's --primary-re-min.")
    ap.add_argument("--v21-domain", action="store_true",
                    help="build the target on the V2.1 domain (primary true Re > 0.5\" AND true "
                         "S/N > 10), taking the thresholds from sbs_shear.domain. Pass this "
                         "whenever the flow is trained with --v21-domain: a target built on a "
                         "different population pins the kept rows to a mean that includes rows the "
                         "trainer never sees, which cost 5%% of R_flow in WORKLOG 2026-07-27d.")
    # WHY A COMPLEMENT. 2026-08-05j/k showed the fiducial `m` is a CANCELLATION across the V2.1
    # resolution cut (+1.64% well-resolved, -2.21% on the rest) and that the emulator is unbiased on
    # BOTH halves, so the split is the flow's. The flow is trained against THIS target, so the next
    # question is whether the target carries the same sign flip. Answering it needs the target built
    # on the complement as well as on V2.1, under otherwise identical settings.
    ap.add_argument("--v21-complement", action="store_true",
                    help="build the target on the COMPLEMENT of the V2.1 domain (within whatever "
                         "--primary-* cuts are given). Mutually exclusive with --v21-domain.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if getattr(args, "v21_domain", False) and getattr(args, "v21_complement", False):
        raise SystemExit("--v21-domain and --v21-complement are mutually exclusive")
    snc = None if args.antithetic else load_snc_lookup(args.snc_lookup, args.snc_cols)

    if args.antithetic:
        need = set(args.anti_cols) | {"applied_g1", "applied_g2"}
    else:
        need = set(raw_columns_for_measurement_targets(args.target_cols))
        need |= {"gamma1_input_p", "gamma2_input_p", "detected"}
    need |= {"r_input_p", "Re_input_p", "distance", "neighbored", "input_index", "case"}
    if args.crowd_col:
        need.add(args.crowd_col)
    parts = []
    nread = 0
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names); cols = sorted(c for c in need if c in avail)
        for bi in range(r.num_record_batches):
            if args.max_rows and nread >= args.max_rows:
                break                       # memory cap: target on a case subset is plenty
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            nread += len(b)
            if "case" in b.columns:
                if args.max_case is not None:
                    b = b[b["case"] <= args.max_case]
                if args.min_case is not None:
                    b = b[b["case"] >= args.min_case]
                if len(b) == 0:
                    continue
            b = source_select_selection(b, cuts=_selection_cuts(args))
            if getattr(args, "v21_domain", False):
                b = sbs_domain.select_frame(b)
            elif getattr(args, "v21_complement", False):
                b = b.drop(index=sbs_domain.select_frame(b).index)
            if len(b) == 0:
                continue
            if "detected" in b.columns:                       # constgold antithetic renders have no detected col
                b = b[b["detected"].astype(bool)]
            b = b.reset_index(drop=True)
            parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    if args.antithetic:
        g1 = df["applied_g1"].to_numpy(float); g2 = df["applied_g2"].to_numpy(float)
    else:
        g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2); keep = gmag > 1e-6
    df = df[keep].reset_index(drop=True); g1, g2, gmag = g1[keep], g2[keep], gmag[keep]
    gh1, gh2 = g1 / gmag, g2 / gmag
    # Per-(case,target) weighting: with ALL-PAIRS each target galaxy appears n_pairs times WITHIN
    # ONE CASE (same measured shape, different neighbour). Weight each row 1/n_pairs so each
    # (case, target) occurrence contributes equally -- one independent measurement / noise
    # realisation, NOT proportional to its neighbour count, and WITHOUT collapsing the 200 cases
    # (noise realisations) of a galaxy into one. Nearest-pair -> n_pairs=1 -> weights all 1.
    if "input_index" in df.columns:
        _ii = df["input_index"].to_numpy(np.int64)
        if "case" in df.columns:
            _key = df["case"].to_numpy(np.int64) * 1_000_003 + _ii   # unique per (case, target)
        else:
            _key = _ii
        _, _inv, _npairs = np.unique(_key, return_inverse=True, return_counts=True)
        w = (1.0 / _npairs[_inv]).astype(float)
    else:
        w = np.ones(len(df), dtype=float)
    if args.antithetic:
        e1p = df[args.anti_cols[0]].to_numpy(float); e2p = df[args.anti_cols[1]].to_numpy(float)
        e1m = df[args.anti_cols[2]].to_numpy(float); e2m = df[args.anti_cols[3]].to_numpy(float)
        # /2 here + /g below => R = <(e_+g - e_-g).ghat> / (2g)  (antithetic cancels intrinsic + noise)
        proj = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / 2.0
        print(f"ANTITHETIC central-diff response: |g|~{np.median(gmag):.4f} (nominal-g={args.nominal_g})")
    else:
        meas = add_measurement_target_features(df.copy())
        e1 = meas[args.target_cols[0]].to_numpy(float)
        e2 = meas[args.target_cols[1]].to_numpy(float)
        if snc is not None:
            if not {"case", "input_index"}.issubset(df.columns):
                raise KeyError("--snc-lookup requires case and input_index in the sheared catalogue")
            key = df["case"].to_numpy(np.int64) * 1_000_003 + df["input_index"].to_numpy(np.int64)
            pos = np.clip(np.searchsorted(snc["key"], key), 0, len(snc["key"]) - 1)
            match = snc["key"][pos] == key
            e1 = e1 - np.where(match, snc["e1"][pos], np.nan)
            e2 = e2 - np.where(match, snc["e2"][pos], np.nan)
            print(f"SNC match after selection: {match.mean():.2%} ({int(match.sum()):,}/{len(match):,})")
        proj = e1 * gh1 + e2 * gh2
    g = args.nominal_g

    flux = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    nbflag = df["neighbored"].astype(bool).to_numpy()
    dist = df["distance"].to_numpy(float)
    crowd = df[args.crowd_col].to_numpy(float) if args.crowd_col else None
    fin = np.isfinite(flux) & np.isfinite(size) & np.isfinite(proj)
    if crowd is not None:
        fin &= np.isfinite(crowd)
    flux, size, proj, nbflag, dist, w = flux[fin], size[fin], proj[fin], nbflag[fin], dist[fin], w[fin]
    if crowd is not None:
        crowd = crowd[fin]

    if args.flux_edges:
        ef = np.array([float(x) for x in args.flux_edges.split(",")], dtype=float)
        ef = np.sort(ef); ef[0] -= 1e-6; ef[-1] += 1e-6
        args.n_flux = len(ef) - 1
        print(f"custom flux edges ({args.n_flux} bins): {np.round(ef, 4).tolist()}")
    else:
        ef = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-6; ef[-1] += 1e-6
    if args.size_edges:
        es = np.array([float(x) for x in args.size_edges.split(",")], dtype=float)
        es = np.sort(es); es[0] -= 1e-6; es[-1] += 1e-6
        args.n_size = len(es) - 1
        print(f"custom size edges ({args.n_size} bins): {np.round(es, 4).tolist()}")
    else:
        es = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-6; es[-1] += 1e-6
    fi = np.clip(np.digitize(flux, ef) - 1, 0, args.n_flux - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, args.n_size - 1)
    if crowd is not None:                              # 3rd axis = crowding QUANTILE bins (equal-count)
        ed = np.quantile(crowd, np.linspace(0, 1, args.n_crowd + 1)); ed[0] -= 1e-9; ed[-1] += 1e-9
        di = np.clip(np.digitize(crowd, ed) - 1, 0, args.n_crowd - 1)
        nblend = args.n_crowd
    else:                                              # 3rd axis = isolated (bin 0) + distance quantiles
        db = dist[nbflag & np.isfinite(dist)]
        ed = np.quantile(db, np.linspace(0, 1, args.n_dist + 1)); ed[0] -= 1e-6; ed[-1] += 1e-6
        di = np.where(nbflag, 1 + np.clip(np.digitize(dist, ed) - 1, 0, args.n_dist - 1), 0)
        nblend = args.n_dist + 1  # bin 0 = isolated

    Rsim = np.full((args.n_flux, args.n_size, nblend), np.nan)
    cnt = np.zeros((args.n_flux, args.n_size, nblend), dtype=np.float64)   # EFFECTIVE (sum of weights = unique-target count)
    rawcnt = np.zeros((args.n_flux, args.n_size, nblend), dtype=np.int64)  # raw pair-row count
    gm = float(np.average(proj, weights=w)) / g
    for a in range(args.n_flux):
        for b in range(args.n_size):
            for c in range(nblend):
                m = (fi == a) & (si == b) & (di == c)
                rawcnt[a, b, c] = int(m.sum())
                cnt[a, b, c] = float(w[m].sum())
                if cnt[a, b, c] > args.min_count:                          # min_count = EFFECTIVE count
                    Rsim[a, b, c] = float(np.average(proj[m], weights=w[m])) / g
    Rsim = np.where(np.isfinite(Rsim), Rsim, gm)

    np.savez(args.output, edges_flux=ef, edges_size=es, edges_dist=ed, Rsim=Rsim,
             counts=cnt, raw_counts=rawcnt, nominal_g=g, global_R=gm, n_dist=args.n_dist,
             crowd_col=(args.crowd_col or ""), edges_crowd=ed,
             response_estimator=("snc" if snc is not None else "raw"),
             snc_lookup=(args.snc_lookup or ""), max_case=(-1 if args.max_case is None else args.max_case),
             # Which POPULATION this target was measured on. Stamped because the flow and its
             # target must agree, and a mismatch is silent -- it shows up only as a wrong R_flow.
             domain=("COMPLEMENT of " + sbs_domain.describe()
                     if getattr(args, "v21_complement", False)
                     else sbs_domain.describe() if getattr(args, "v21_domain", False)
                     else "default"))
    axis = f"crowd[{args.crowd_col}]" if args.crowd_col else "blend"
    print(f"R_sim(flux x size x {axis}) target, g={g}, N_pairs={len(proj):,}, N_eff(unique-target)={w.sum():.0f}, global R={gm:.4f}")
    print(f"grid {args.n_flux}x{args.n_size}x{nblend}" + ("" if args.crowd_col else " (blend bin 0=isolated, by distance)"))
    for c in range(nblend):
        tag = (f"{args.crowd_col} q{c}" if args.crowd_col else ("ISOLATED" if c == 0 else f"blend d-bin {c}"))
        mincell = int(np.nanmin(cnt[:, :, c])) if np.isfinite(cnt[:, :, c]).any() else 0
        print(f"  {tag}: R_sim {np.nanmin(Rsim[:, :, c]):.3f}..{np.nanmax(Rsim[:, :, c]):.3f}  "
              f"(N_eff={cnt[:, :, c].sum():.0f}, min-cell N_eff={mincell}, N_pairs={rawcnt[:, :, c].sum():,})")
    print(f"R_sim spans {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f} (x{np.nanmax(Rsim)/np.nanmin(Rsim):.2f})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
