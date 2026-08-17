"""Thin SBSI-side builder for g-specific detection+measured catalogues.

The standalone blendemu ``build_detection_catalogue.py`` CLI was removed when the
blendemu pipeline was rewritten around ``run_pipeline.py``.  Its engine,
``blendemu.response.retrieve_detection``, still exists and is what the pipeline
calls internally.  The pipeline, however, only builds a single detection
catalogue at the *sheared* self-response case (``shear=0.05`` for the live
``lsst_sims_fs2_25876`` run) and without SExtractor measured columns.

The refined SBSI framework (SBI_shear.md §2/§4) needs:

  * a ``g=0`` detection catalogue to train the shear-free forward model
    ``P(s=1|x,n)`` and ``p_meas(x_hat|x,n)``, and
  * ``g=0.05`` / ``g=0.2`` detection catalogues as held-out-shear validation.

All of these renderings already exist on disk (``case{c}_{g}/real0/...``), so this
script simply re-derives the catalogue for a requested shear with
``include_measured=True`` and the nearest-pair (``k=2``) neighbour annotation,
mirroring the pipeline's ``step_response`` detection loop.  It does NOT
re-simulate.  Catalogue-building logic stays in blendemu; this is only a loop /
I/O wrapper, kept on the SBSI side per the user's instruction.
"""

from __future__ import annotations

import pathlib as _pathlib, sys as _sys  # noqa: E402  -- make `sbs_shear` importable
_sys.path.insert(0, str(next(p for p in _pathlib.Path(__file__).resolve().parents
                             if (p / 'sbs_shear').is_dir())))
from sbs_shear import paths  # noqa: E402

import argparse
import os
import sys
import time

import pandas as pd

# This is the one SBSI script that requires blendemu at import time: it is the bridge
# that turns an existing blendemu rendering into an SBSI input catalogue. Everything
# downstream consumes the feather it writes and needs no blendemu at all. Point at your
# checkout with BLENDEMU_ROOT (see sbs_shear/paths.py).
sys.path.insert(0, str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                            if (p / "sbs_shear").is_dir())))
from sbs_shear.emulator import import_blendemu  # noqa: E402

import_blendemu()
from blendemu import response  # noqa: E402
from blendemu import catalog as bcatalog  # noqa: E402


def parse_cases(spec: str) -> list[int]:
    """Parse a case spec like '0-199' or '0,3,5' or '0-9,20,25-27'."""
    cases: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cases.extend(range(int(lo), int(hi) + 1))
        else:
            cases.append(int(part))
    return sorted(set(cases))


def case_dir_exists(data_path: str, case: int, shear_label: str, real: str) -> bool:
    return os.path.isdir(os.path.join(data_path, f"case{case}_{shear_label}", real))


def build_one_case(case, shear_label, args):
    """Return (case, dataframe-or-None, error-or-None)."""
    try:
        df = response.retrieve_detection(
            case=case,
            shear=shear_label,
            real=args.real,
            r_max=args.r_max,
            r_min=args.r_min,
            k=args.k,
            data_path=args.data_path,
            tile_name=args.tile_name,
            include_measured=args.include_measured,
            measured_columns=None if args.measured_columns is None else args.measured_columns,
            attach_nearest_neighbor=args.attach_nearest_neighbor,
            include_shapes=args.include_shapes,
            shape_columns=None if args.shape_columns is None else args.shape_columns,
        )
        if df is not None and "case" not in df.columns:
            df["case"] = case          # needed for per-(case,target) weighting of all-pairs rows
        if getattr(args, "flow_only", False) and df is not None:
            import numpy as _np
            if "measured_ngmix_g1" in df.columns and "detected" in df.columns:
                keep = (df["detected"].astype(bool).to_numpy()
                        & _np.isfinite(df["measured_ngmix_g1"].to_numpy(float)))
                df = df[keep].reset_index(drop=True)
        return case, df, None
    except Exception as exc:  # noqa: BLE001 - we want to skip and report bad cases
        return case, None, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=f"{paths.SIM_DIR}",
                        help="blendemu simulation output directory (holds case{c}_{g}/ and gals{c}_{g}.feather)")
    parser.add_argument("--shear", required=True,
                        help="Shear value, e.g. 0.0 / 0.05 / 0.2. Matched to the case dir label.")
    parser.add_argument("--cases", default="0-199", help="Case spec, e.g. '0-199' or '0,5,10'.")
    parser.add_argument("--output", required=True, help="Output merged feather path.")
    parser.add_argument("--real", default="real0")
    parser.add_argument("--tile-name", default="tile180.0_-0.5")
    parser.add_argument("--r-max", type=float, default=3.0, help="Neighbour search radius [arcsec].")
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=2, help="KDTree neighbours (k=2 -> nearest pair).")
    parser.add_argument("--no-measured", dest="include_measured", action="store_false",
                        help="Disable the SExtractor measured-column join (on by default).")
    parser.add_argument("--measured-columns", default=None,
                        help="Comma-separated SExtractor columns, or 'all'. Default: blendemu compact set.")
    parser.add_argument("--include-shapes", action="store_true",
                        help="Also join PSF-corrected shape estimators (NGMIX_G1/G2, GALSIM_G1/G2) "
                             "from the aligned Shapes/ catalogue -> measured_ngmix_g1/g2 etc.")
    parser.add_argument("--shape-columns", default=None,
                        help="Comma-separated Shapes columns to join. Default: NGMIX/GALSIM G1/G2.")
    parser.add_argument("--attach-nearest-neighbor", action="store_true",
                        help="Fill non-neighbour rows with the nearest catalogue neighbour (neighbored stays False). "
                             "Off by default to match the live pipeline detection catalogue.")
    parser.add_argument("--flow-only", action="store_true",
                        help="Keep only flow-usable rows (detected & finite ngmix = measured sheared "
                             "secondaries); drops undetected / primary-target / ngmix-fail rows. "
                             "~1/5 the size for all-pairs builds. Requires --include-shapes.")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Cases per intermediate feather (memory control).")
    parser.set_defaults(include_measured=True)
    args = parser.parse_args()

    if args.measured_columns is not None and args.measured_columns != "all":
        args.measured_columns = [c.strip() for c in args.measured_columns.split(",") if c.strip()]
    if args.shape_columns is not None and args.shape_columns != "all":
        args.shape_columns = [c.strip() for c in args.shape_columns.split(",") if c.strip()]

    shear_label = bcatalog.shear_label(float(args.shear))
    requested = parse_cases(args.cases)
    present = [c for c in requested if case_dir_exists(args.data_path, c, shear_label, args.real)]
    missing_dirs = [c for c in requested if c not in present]

    print(f"SBSI detection+measured catalogue build")
    print(f"  data_path        : {args.data_path}")
    print(f"  shear            : {args.shear} -> dir label '{shear_label}'")
    print(f"  cases requested  : {len(requested)} ({args.cases})")
    print(f"  cases present    : {len(present)}")
    if missing_dirs:
        print(f"  cases MISSING dir: {len(missing_dirs)} -> {missing_dirs[:20]}{' ...' if len(missing_dirs) > 20 else ''}")
    print(f"  include_measured : {args.include_measured}")
    print(f"  attach_nearest   : {args.attach_nearest_neighbor}")
    print(f"  r_max/r_min/k    : {args.r_max}/{args.r_min}/{args.k}")
    print(f"  output           : {args.output}")
    print(f"  n_jobs/batch     : {args.n_jobs}/{args.batch_size}", flush=True)

    if not present:
        raise SystemExit("No present case directories for the requested shear; nothing to build.")

    from joblib import Parallel, delayed

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp_dir = os.path.abspath(args.output) + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)

    t0 = time.time()
    parts: list[str] = []
    failures: list[tuple[int, str]] = []
    total_rows = 0

    # Process in batches of cases to bound peak memory.
    batches = [present[i:i + args.batch_size] for i in range(0, len(present), args.batch_size)]
    for bi, batch_cases in enumerate(batches):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(build_one_case)(case, shear_label, args) for case in batch_cases
        )
        frames = []
        for case, df, err in results:
            if err is not None:
                failures.append((case, err))
                continue
            frames.append(df)
        if not frames:
            print(f"  batch {bi}: no successful cases", flush=True)
            continue
        batch_df = pd.concat(frames, ignore_index=True)
        total_rows += len(batch_df)
        part_path = os.path.join(tmp_dir, f"part_{bi:04d}.feather")
        batch_df.to_feather(part_path)
        parts.append(part_path)
        del batch_df, frames
        print(f"  batch {bi+1}/{len(batches)} cases={batch_cases[0]}..{batch_cases[-1]} "
              f"rows_so_far={total_rows:,} elapsed={time.time()-t0:.0f}s", flush=True)

    if not parts:
        raise SystemExit("No catalogue parts were produced.")

    # Stream-merge the parts into one Arrow IPC (feather) file so peak memory is
    # bounded by a single part, not the full (tens-of-GB) catalogue.
    import pyarrow.feather as pafeather
    import pyarrow.ipc as ipc

    print(f"\nStream-merging {len(parts)} parts -> {args.output}", flush=True)
    writer = None
    merged_rows = 0
    det_sum = det_n = nbr_sum = nbr_n = 0
    measured_cols = []
    try:
        for p in parts:
            table = pafeather.read_table(p)
            if writer is None:
                writer = ipc.new_file(args.output, table.schema)
                measured_cols = [c for c in table.schema.names if c.startswith("measured_")]
            writer.write_table(table)
            merged_rows += table.num_rows
            if "detected" in table.schema.names:
                col = table.column("detected").to_numpy(zero_copy_only=False)
                det_sum += int(col.sum()); det_n += len(col)
            if "neighbored" in table.schema.names:
                col = table.column("neighbored").to_numpy(zero_copy_only=False)
                nbr_sum += int(col.sum()); nbr_n += len(col)
            del table
    finally:
        if writer is not None:
            writer.close()
    print(f"  merged rows: {merged_rows:,}")
    print(f"  measured_* columns ({len(measured_cols)}): {measured_cols}")
    if det_n:
        print(f"  detected rate: {det_sum / det_n:.4f}")
    if nbr_n:
        print(f"  neighbored rate: {nbr_sum / nbr_n:.4f}")

    # Clean up intermediate parts.
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if failures:
        print(f"\n  WARNING: {len(failures)} case(s) failed and were skipped:")
        for case, err in failures[:30]:
            print(f"    case {case}: {err}")
        if len(failures) > 30:
            print(f"    ... and {len(failures) - 30} more")

    print(f"\nDone in {time.time() - t0:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
