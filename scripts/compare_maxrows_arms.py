"""Seed-paired comparison of two training arms that differ ONLY in `--max-rows`.

WHY PAIRED.  Each seed carries its own offset, and that offset does NOT cancel in an absolute number
(AGENTS.md "Ensemble Seed Convention").  It DOES cancel in a model-vs-model difference, so this
script compares the two arms WITHIN each seed and then averages the differences.  With 3 seeds that
is the only defensible read; the raw per-arm means are printed for orientation only.

WHAT IT MAY AND MAY NOT SUPPORT.  These are TRAINING-SIDE diagnostics (val NLL, <R_model> on the
validation split, the per-bin response and orientation-coupling residuals).  They say whether the
optimisation landed in a different place.  They are NOT `m`.  Per AGENTS.md any e-response/`m` number
needs 16 seeds and the constgold path with the blend emulator -- **do not quote `m` from this.**

TWO CONFOUNDS, both deliberate and both stated in the output:
  * at fixed --epochs, the all-rows arm also takes ~2.9x more GRADIENT STEPS, so this measures the
    recipe run on all rows, not data volume alone;
  * <R_model> here is on each arm's OWN validation split, and those splits are different populations
    (0.6M drawn from 4M vs 1.75M drawn from 11.68M).  The response/coupling residuals are pinned
    against the same fixed targets, so those are comparable; <R_model> is weaker evidence.

Reported over the SWA window (the last `swa_last_k` epochs actually averaged into the shipped
checkpoint), not the single final epoch, because epoch-to-epoch scatter is large.

Usage:
    python -u scripts/compare_maxrows_arms.py [--seeds 501 502 503]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

CACHE = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation"
STEM = "measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6"
ARMS = [("ctl4m", "4M control"), ("allrows", "all rows")]
METRICS = [
    ("val_nll", "val NLL", "lower better"),
    ("train_nll", "train NLL", "lower better"),
    ("val_resp", "per-bin response resid", "lower better"),
    ("val_theta", "orientation-coupling resid", "lower better"),
    ("val_R", "<R_model> (val split)", "vs target"),
]


def arm_paths(arm, seed):
    return (
        os.path.join(CACHE, f"{STEM}_{arm}_s{seed}_train_curve.npz"),
        os.path.join(CACHE, f"{STEM}_{arm}_s{seed}_swaavg.pt"),
    )


def load_arm(arm, seed):
    curve_path, ckpt_path = arm_paths(arm, seed)
    for p in (curve_path, ckpt_path):
        if not os.path.exists(p):
            return None
    z = np.load(curve_path)
    k = int(z["swa_last_k"])
    out = {name: float(np.mean(z[name][-k:])) for name, _, _ in METRICS}
    out["swa_last_k"] = k
    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)["metadata"]
    out["train_rows"] = int(meta["train_rows"])
    out["val_rows"] = int(meta["validation_rows"])
    out["max_rows"] = meta["max_rows"]
    out["best_val_nll"] = float(meta["best_val_nll"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    args = parser.parse_args()

    loaded, missing = {}, []
    for arm, _ in ARMS:
        for seed in args.seeds:
            got = load_arm(arm, seed)
            if got is None:
                missing.append(f"{arm}/s{seed}")
            else:
                loaded[(arm, seed)] = got
    if missing:
        print(f"MISSING (not finished yet?): {', '.join(missing)}\n")
    seeds = [s for s in args.seeds if all((a, s) in loaded for a, _ in ARMS)]
    if not seeds:
        raise SystemExit("No seed has BOTH arms present -- nothing can be compared yet.")
    print(f"Seeds with both arms: {seeds}\n")

    ref = loaded[(ARMS[0][0], seeds[0])]
    alt = loaded[(ARMS[1][0], seeds[0])]
    print(f"rows:  {ARMS[0][1]}  train={ref['train_rows']:,} val={ref['val_rows']:,} "
          f"(max_rows={ref['max_rows']})")
    print(f"rows:  {ARMS[1][1]}  train={alt['train_rows']:,} val={alt['val_rows']:,} "
          f"(max_rows={alt['max_rows']})")
    ratio = alt["train_rows"] / ref["train_rows"]
    print(f"       -> {ratio:.2f}x the data, and at fixed --epochs also {ratio:.2f}x the gradient "
          f"steps (confound, see docstring)\n")

    print(f"{'metric':<28}{ARMS[0][1]:>12}{ARMS[1][1]:>12}{'paired d':>12}{'+- sem':>10}   note")
    print("-" * 86)
    for name, label, note in METRICS:
        a = np.array([loaded[(ARMS[0][0], s)][name] for s in seeds])
        b = np.array([loaded[(ARMS[1][0], s)][name] for s in seeds])
        d = b - a
        sem = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        print(f"{label:<28}{a.mean():>12.5f}{b.mean():>12.5f}{d.mean():>+12.5f}{sem:>10.5f}   {note}")

    print("\nper-seed paired differences (all rows - 4M control):")
    print(f"  {'seed':<8}" + "".join(f"{lab:>28}" for _, lab, _ in METRICS))
    for s in seeds:
        row = "".join(
            f"{loaded[(ARMS[1][0], s)][n] - loaded[(ARMS[0][0], s)][n]:>+28.5f}"
            for n, _, _ in METRICS
        )
        print(f"  {s:<8}{row}")

    print("\nNOTE: training-side only. Not `m`. Any m/e-response claim needs 16 seeds and the "
          "constgold+emulator path (AGENTS.md).")


if __name__ == "__main__":
    main()
