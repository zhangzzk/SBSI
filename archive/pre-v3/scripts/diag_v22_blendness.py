"""Localise V2.2 constgold closure versus two distinct blendness definitions.

This is evaluation-only.  It reports the flow residual

    R_flow / (R_sim - R_blend) - 1

with seed and case-blocked simulation errors.  ``R_blend`` is the emulator-defined total blendness
used by the deployed model.  ``neighbored``/``distance`` come from constgold's nearest-neighbour,
3-arcsec-capped response catalogue and therefore diagnose close-neighbour status only; they are not
a complete scene neighbour list and must not be called true isolation.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from scripts.diag_rflow_v21 import Table, cell_table  # noqa: E402
from scripts.eval_v2_indomain_m import catalogue_true_props  # noqa: E402


def assigned_table(title, labels, assignments, blocks):
    ids = np.full(len(assignments[0]), -1, np.int64)
    for i, mask in enumerate(assignments):
        if np.any((ids >= 0) & mask):
            raise RuntimeError(f"overlapping masks in {title}")
        ids[mask] = i
    return Table(title, "blendness", labels, ids, len(labels), *blocks)


def report_total_m(table):
    """Report the actual per-bin model bias, with the same two error sources as Table.report."""
    a = np.stack(table.per_seed)  # seed, bin, (R_sim, R_flow, R_blend)
    nseed = a.shape[0]
    print("  direct total-model bias: m = R_sim/(R_flow+R_blend)-1")
    print(f"  {table.header:>22}{'m %':>10}{'+-seed':>10}{'+-sim':>10}{'+-total':>10}{'N':>12}")
    for i, label in enumerate(table.labels):
        if table.n[i] < 2000:
            continue
        total = a[:, i, 1] + a[:, i, 2]
        m_seed = 100.0 * (a[:, i, 0] / total - 1.0)
        e_seed = m_seed.std(ddof=1) / np.sqrt(nseed)
        # As in the established evaluator, only finite-simulation R_sim sampling contributes this
        # term; the model prediction is treated as fixed on the evaluated population.
        e_sim = abs(100.0 / total.mean()) * table.sim_sem[i]
        print(f"  {label:>22}{m_seed.mean():>+10.2f}{e_seed:>10.2f}{e_sim:>10.2f}"
              f"{np.hypot(e_seed, e_sim):>10.2f}{table.n[i]:>12,}")


def report_paired_total_m_change(candidate, reference):
    """Same-row/same-seed curve change; simulation sampling cancels exactly."""
    ca = np.stack(candidate.per_seed)
    ra = np.stack(reference.per_seed)
    if ca.shape != ra.shape or not np.array_equal(ca[:, :, 0], ra[:, :, 0]) \
            or not np.array_equal(ca[:, :, 2], ra[:, :, 2]):
        raise RuntimeError("candidate/reference R_sim or R_blend cell means differ")
    cm = 100.0 * (ca[:, :, 0] / (ca[:, :, 1] + ca[:, :, 2]) - 1.0)
    rm = 100.0 * (ra[:, :, 0] / (ra[:, :, 1] + ra[:, :, 2]) - 1.0)
    dm = cm - rm
    print("  paired total-model curve change (candidate - reference):")
    print(f"  {candidate.header:>22}{'reference':>12}{'candidate':>12}{'delta':>12}{'+-paired':>12}")
    for i, label in enumerate(candidate.labels):
        if candidate.n[i] < 2000:
            continue
        e = dm[:, i].std(ddof=1) / np.sqrt(dm.shape[0])
        print(f"  {label:>22}{rm[:, i].mean():>+12.3f}{cm[:, i].mean():>+12.3f}"
              f"{dm[:, i].mean():>+12.3f}{e:>12.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--reference-dump-glob", default=None,
                    help="optional same-seed reference dumps; reports paired per-bin changes")
    ap.add_argument("--expect-seeds", type=int, default=16)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--blend-eps", type=float, default=0.02)
    ap.add_argument(
        "--rblend-scale", type=float, default=1.0,
        help="explicit frozen multiplicative calibration applied after raw-R_blend bin assignment",
    )
    ap.add_argument(
        "--rblend-offset", type=float, default=0.0,
        help="explicit frozen additive response applied after raw-R_blend bin assignment",
    )
    args = ap.parse_args()
    t0 = time.time()

    dumps = sorted(glob.glob(args.dump_glob))
    if len(dumps) != args.expect_seeds:
        raise SystemExit(f"expected {args.expect_seeds} candidate dumps, found {len(dumps)}")
    seed_re = re.compile(r"_s(\d+)\.feather$")

    def seed_map(paths):
        out = {}
        for path in paths:
            match = seed_re.search(os.path.basename(path))
            if not match:
                raise RuntimeError(f"cannot parse seed from {path}")
            seed = int(match.group(1))
            if seed in out:
                raise RuntimeError(f"duplicate seed {seed} in dump glob")
            out[seed] = path
        return out

    candidate_by_seed = seed_map(dumps)
    seeds = sorted(candidate_by_seed)
    dumps = [candidate_by_seed[s] for s in seeds]
    reference_dumps = None
    if args.reference_dump_glob:
        reference_by_seed = seed_map(sorted(glob.glob(args.reference_dump_glob)))
        missing = [seed for seed in seeds if seed not in reference_by_seed]
        if missing:
            raise SystemExit(f"reference dumps are missing candidate seeds {missing}")
        reference_dumps = [reference_by_seed[s] for s in seeds]
    tp = catalogue_true_props(args.catalogue, args.min_case, t0)
    mag = tp["r_input_p"].to_numpy(float)
    re_ = tp["Re_input_p"].to_numpy(float)
    domain = (mag < args.mag_max) & (re_ > args.re_min)

    first = pf.read_table(dumps[0], columns=["case", "input_index", "R_blend", "neighbored", "distance"])
    if len(first) != len(tp):
        raise RuntimeError("dump/catalogue length mismatch")
    if not np.array_equal(first["case"].to_numpy(), tp["case"].to_numpy(np.int64)) \
            or not np.array_equal(first["input_index"].to_numpy(), tp["input_index"].to_numpy(np.int64)):
        raise RuntimeError("dump/catalogue key alignment failure")
    rb0 = first["R_blend"].to_numpy(zero_copy_only=False).astype(float)
    ngh = first["neighbored"].to_numpy(zero_copy_only=False).astype(bool)
    dist = first["distance"].to_numpy(zero_copy_only=False).astype(float)

    cases = tp["case"].to_numpy(np.int64)
    uniq = np.unique(cases)
    blocks = (np.searchsorted(uniq, cases), len(uniq))

    # Model-defined blendness: one low-response class plus equal-count quantiles above the fixed
    # epsilon already used by validate_constant_with_blend.py.  Edges are frozen before responses
    # are inspected, and R_blend is asserted identical across every seed dump below.
    positive = domain & (rb0 >= args.blend_eps)
    q = np.quantile(rb0[positive], [0.0, 0.25, 0.5, 0.75, 1.0])
    q[-1] = np.nextafter(q[-1], np.inf)
    rb_masks = [domain & (rb0 < args.blend_eps)]
    rb_labels = [f"Rbl<{args.blend_eps:.2f}"]
    for i in range(4):
        rb_masks.append(domain & (rb0 >= q[i]) & (rb0 < q[i + 1]))
        rb_labels.append(f"q{i+1} [{q[i]:.3f},{q[i+1]:.3f})")
    rb_table = assigned_table("by EMULATOR TOTAL R_blend", rb_labels, rb_masks, blocks)

    # Catalogue-defined close-neighbour status.  Its false class means no ANNOTATED neighbour in
    # this nearest-neighbour/3-inch build, not absence of all physical or emulator-list neighbours.
    flag_table = cell_table(
        "by CONSTGOLD 3-inch close-neighbour flag", "close-neighbour",
        [("not flagged (<3in)", domain & ~ngh), ("flagged (<3in)", domain & ngh)], blocks=blocks,
    )
    distance_table = cell_table(
        "by CONSTGOLD nearest annotated distance", "distance",
        [("not flagged", domain & ~ngh),
         ("[0,1) arcsec", domain & ngh & (dist >= 0) & (dist < 1)),
         ("[1,2) arcsec", domain & ngh & (dist >= 1) & (dist < 2)),
         ("[2,3.01) arcsec", domain & ngh & (dist >= 2) & (dist < 3.01))],
        blocks=blocks,
    )
    tables = [rb_table, flag_table, distance_table]
    reference_tables = None
    if reference_dumps is not None:
        reference_tables = [
            assigned_table("reference by EMULATOR TOTAL R_blend", rb_labels, rb_masks, blocks),
            cell_table(
                "reference by CONSTGOLD 3-inch close-neighbour flag", "close-neighbour",
                [("not flagged (<3in)", domain & ~ngh), ("flagged (<3in)", domain & ngh)],
                blocks=blocks,
            ),
            cell_table(
                "reference by CONSTGOLD nearest annotated distance", "distance",
                [("not flagged", domain & ~ngh),
                 ("[0,1) arcsec", domain & ngh & (dist >= 0) & (dist < 1)),
                 ("[1,2) arcsec", domain & ngh & (dist >= 1) & (dist < 2)),
                 ("[2,3.01) arcsec", domain & ngh & (dist >= 2) & (dist < 3.01))],
                blocks=blocks,
            ),
        ]

    rs0 = None
    for path in dumps:
        t = pf.read_table(path, columns=["r_sim", "R_flow", "R_blend"])
        rb = t["R_blend"].to_numpy(zero_copy_only=False).astype(float)
        if not np.array_equal(rb, rb0):
            raise RuntimeError(f"R_blend differs across seeds: {path}")
        rs = t["r_sim"].to_numpy(zero_copy_only=False).astype(float)
        if rs0 is None:
            rs0 = rs
        elif not np.array_equal(rs, rs0):
            raise RuntimeError(f"R_sim differs across candidate seeds: {path}")
        cols = [rs,
                t["R_flow"].to_numpy(zero_copy_only=False).astype(float),
                args.rblend_scale * rb + args.rblend_offset]
        for table in tables:
            table.add(*cols)

    if reference_dumps is not None:
        for path in reference_dumps:
            t = pf.read_table(path, columns=["r_sim", "R_flow", "R_blend"])
            rs = t["r_sim"].to_numpy(zero_copy_only=False).astype(float)
            rb = t["R_blend"].to_numpy(zero_copy_only=False).astype(float)
            if not np.array_equal(rs, rs0) or not np.array_equal(rb, rb0):
                raise RuntimeError(f"reference R_sim/R_blend differs from candidate: {path}")
            cols = [rs, t["R_flow"].to_numpy(zero_copy_only=False).astype(float), rb]
            for table in reference_tables:
                table.add(*cols)

    print(f"\nV2.2 domain N={int(domain.sum()):,}; cases={len(uniq)}; dumps={len(dumps)}")
    print(f"Applied frozen R_blend scale: {args.rblend_scale:.9f}")
    print(f"Applied frozen R_blend offset: {args.rblend_offset:+.9f}")
    print("Residual is R_flow/(R_sim-R_blend)-1: negative means missing model response.")
    print("Errors: seed SEM and case-blocked R_sim sampling SEM, shown separately.")
    print(f"Emulator blend quantile edges above {args.blend_eps}: {q.tolist()}")
    print(f"Constgold close-neighbour flag fraction: {ngh[domain].mean():.4%}")
    for table in tables:
        table.report()
        report_total_m(table)
    if reference_tables is not None:
        for candidate, reference in zip(tables, reference_tables):
            print(f"\n=== PAIRED CURVE: {candidate.title} ===")
            report_paired_total_m_change(candidate, reference)
    print("\nDIAG_V22_BLENDNESS_DONE", flush=True)


if __name__ == "__main__":
    main()
