"""Evaluate the completed V2.1 ensemble with the independent flow-#2 blend term.

This is an evaluation-only, parameter-free comparison on constgold:

    m = R_sim / (R_flow + R_blend) - 1

The baseline is the R_blend stored in each V2.1 per-object dump.  The candidate replaces only
R_blend with the existing 16-seed g=0.2 flow-#2 ensemble, which was trained and selected entirely
on half-shear pairs.  Constgold therefore remains evaluation-only; no value from this script is
used to fit, tune, rescale, or select either model.

Both stochastic response terms are paired by their conventional seed labels.  The pairing is
arbitrary because the flow-#1 and flow-#2 trainings are independent, but forming m inside each
paired seed and taking the spread is conservative and follows AGENTS.md's e-response rule.  The
paired delta additionally cancels R_sim exactly and much of the flow-#1 seed variation.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SEEDS = [501, 502, 503, 505, 506, 507, 508, 509,
         510, 511, 512, 513, 514, 515, 516, 517]


def key64(case, index):
    case = np.asarray(case, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)
    if index.size and (index.min() < 0 or index.max() >= (1 << 40)):
        raise RuntimeError("input_index does not fit the guarded 40-bit packed key")
    return (case << 40) | index


def seed_from_path(path):
    stem = os.path.basename(path).rsplit(".feather", 1)[0]
    try:
        return int(stem.rsplit("_s", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"cannot parse seed from {path}") from exc


def load_domain_module():
    """Load the canonical domain.py without importing sbs_shear/__init__.py (and torch)."""
    path = os.path.join(SBSI_ROOT, "sbs_shear", "domain.py")
    spec = importlib.util.spec_from_file_location("_v21_domain_standalone", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical V2.1 domain from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def true_props_for_dump(catalogue, min_case, ref_key):
    """Join the canonical true properties to dump keys, preserving dump row order.

    This is stricter and lighter than importing the historical catalogue-replay helper: the dump
    already fixes the selected population, so this only asks the source catalogue for true mag/Re
    by the exact (case, input_index) identity.  Coverage and key uniqueness are asserted.
    """
    cols = ["case", "input_index", "r_input_p", "Re_input_p"]
    t = pf.read_table(catalogue, columns=cols, memory_map=True)
    case = t["case"].to_numpy(zero_copy_only=False).astype(np.int64)
    use = case >= min_case
    key = key64(case[use], t["input_index"].to_numpy(zero_copy_only=False)[use])
    order = np.argsort(key)
    skey = key[order]
    if skey.size > 1 and np.any(skey[1:] == skey[:-1]):
        raise SystemExit("REFUSING: constgold catalogue has duplicate (case, input_index) keys")
    pos = np.searchsorted(skey, ref_key)
    clip = np.clip(pos, 0, len(skey) - 1)
    hit = (pos < len(skey)) & (skey[clip] == ref_key)
    if not hit.all():
        raise SystemExit(f"REFUSING: true-property catalogue covers only {hit.mean():.4%} of dump rows")
    mag_all = t["r_input_p"].to_numpy(zero_copy_only=False).astype(float)[use][order]
    re_all = t["Re_input_p"].to_numpy(zero_copy_only=False).astype(float)[use][order]
    return mag_all[clip], re_all[clip]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-glob", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--flow2-lookup", required=True)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--min-match", type=float, default=0.999)
    args = ap.parse_args()

    dumps = sorted(glob.glob(args.dump_glob), key=seed_from_path)
    seeds = [seed_from_path(p) for p in dumps]
    if seeds != SEEDS:
        raise SystemExit(f"REFUSING: expected V2.1 seeds {SEEDS}, found {seeds}")
    print(f"V2.1 dumps: {len(dumps)} seeds {seeds}", flush=True)

    ref = pf.read_table(dumps[0], columns=["case", "input_index"]).to_pandas()
    ref_key = key64(ref["case"].to_numpy(), ref["input_index"].to_numpy())
    mag, re = true_props_for_dump(args.catalogue, args.min_case, ref_key)
    sbs_domain = load_domain_module()
    v21 = sbs_domain.in_domain(mag, re)
    print(f"V2.1 population: {int(v21.sum()):,} of {len(v21):,} rows", flush=True)

    flow_cols = [f"R_blend_flow_s{s}" for s in SEEDS]
    wanted = ["case", "input_index", "in_domain", "R_blend_flow", *flow_cols]
    print(f"reading flow-#2 lookup: {args.flow2_lookup}", flush=True)
    lk = pf.read_table(args.flow2_lookup, columns=wanted, memory_map=True)
    lk_case = lk["case"].to_numpy(zero_copy_only=False).astype(np.int64)
    lk_idx = lk["input_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    lk_key = key64(lk_case, lk_idx)
    order = np.argsort(lk_key)
    skey = lk_key[order]
    if skey.size > 1 and np.any(skey[1:] == skey[:-1]):
        raise SystemExit("REFUSING: flow-#2 lookup has duplicate (case, input_index) keys")

    pos = np.searchsorted(skey, ref_key)
    pos_clip = np.clip(pos, 0, len(skey) - 1)
    hit = (pos < len(skey)) & (skey[pos_clip] == ref_key)
    indom_lk = lk["in_domain"].to_numpy(zero_copy_only=False).astype(bool)[order]
    in_flow2_domain = np.zeros(len(ref), bool)
    in_flow2_domain[hit] = indom_lk[pos_clip[hit]]
    keep = v21 & hit & in_flow2_domain
    frac = keep.sum() / max(v21.sum(), 1)
    print(f"flow-#2 coverage on V2.1: {int(keep.sum()):,}/{int(v21.sum()):,} "
          f"({frac:.4%}); unmatched/out-of-domain rows are DROPPED", flush=True)
    if frac < args.min_match:
        raise SystemExit(f"REFUSING: flow-#2 covers only {frac:.2%} of V2.1 "
                         f"(required {args.min_match:.2%})")

    # Pull only the kept rows after the key search.  This avoids materialising a dense
    # (n_dump_rows x n_seed) matrix for the 80% of constgold outside V2.1.
    take = pos_clip[keep]
    rb2 = np.column_stack([
        lk[c].to_numpy(zero_copy_only=False).astype(float)[order][take] for c in flow_cols
    ])
    rb2_mean_col = lk["R_blend_flow"].to_numpy(zero_copy_only=False).astype(float)[order][take]
    mean_delta = np.max(np.abs(rb2.mean(axis=1) - rb2_mean_col))
    print(f"flow-#2 ensemble-column identity max|mean(seed cols)-stored mean|={mean_delta:.3e}")
    if mean_delta > 1e-6:
        raise RuntimeError("stored flow-#2 ensemble column is not the mean of its seed columns")

    base_m, cand_m = [], []
    components = []
    for j, (seed, path) in enumerate(zip(SEEDS, dumps)):
        t = pf.read_table(path, columns=["case", "input_index", "r_sim", "R_flow", "R_blend"])
        if t.num_rows != len(ref):
            raise RuntimeError(f"{path}: row count changed")
        if not np.array_equal(t["case"].to_numpy(), ref["case"].to_numpy()) \
                or not np.array_equal(t["input_index"].to_numpy(), ref["input_index"].to_numpy()):
            raise RuntimeError(f"{path}: row order changed")
        rs = t["r_sim"].to_numpy(zero_copy_only=False).astype(float)[keep]
        rf = t["R_flow"].to_numpy(zero_copy_only=False).astype(float)[keep]
        rb0 = t["R_blend"].to_numpy(zero_copy_only=False).astype(float)[keep]
        a, b, c0, c2 = rs.mean(), rf.mean(), rb0.mean(), rb2[:, j].mean()
        m0 = 100.0 * (a / (b + c0) - 1.0)
        m2 = 100.0 * (a / (b + c2) - 1.0)
        base_m.append(m0)
        cand_m.append(m2)
        components.append((a, b, c0, c2))
        print(f"seed {seed}: baseline {m0:+.3f}%  flow#2 {m2:+.3f}%  "
              f"delta {m2-m0:+.3f} pt", flush=True)

    base_m = np.asarray(base_m)
    cand_m = np.asarray(cand_m)
    delta = cand_m - base_m
    comp = np.asarray(components).mean(axis=0)

    def summary(name, values):
        sd = values.std(ddof=1)
        print(f"  {name:<24} {values.mean():+8.3f} +- {sd/np.sqrt(len(values)):.3f}% "
              f"(seed sd {sd:.3f}%, N={len(values)})")

    print("\n" + "=" * 92)
    print("V2.1 CONSTGOLD RESULT -- ratio formed inside each paired seed")
    print("=" * 92)
    summary("current V2.1", base_m)
    summary("V2.1 + flow #2", cand_m)
    summary("paired change", delta)
    print(f"\n  seed-mean components: R_sim={comp[0]:.6f} R_flow={comp[1]:.6f} "
          f"R_blend_current={comp[2]:.6f} R_blend_flow2={comp[3]:.6f}")
    print("  Flow #2 was selected on its half-shear ruler before this constgold evaluation; this")
    print("  result is an acceptance test, not permission to choose a model by whichever m is smaller.")
    print("\nEVAL_V21_BLENDFLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
