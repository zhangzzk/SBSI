#!/usr/bin/env python -B
"""Measure the SEED contribution to realistic-cut m, to decide if more seeds would help.

Reads N single-seed selrobust rows.json (each computed with the SAME, common-mode R_blend so
the R_blend level cancels in the seed-to-seed scatter) plus the K-seed ENSEMBLE rows.json.
Per selection label it reports:

  m_ens          : the K-seed ensemble m (what we ship)
  sigma_seed     : std across single-seed m's (ddof=1) -- the 1-seed training/init scatter
  se_ens(K)      : sigma_seed / sqrt(K)   -- seed error already removed by the K-seed ensemble
  merr           : object-sampling (Monte-Carlo) error at fixed R_flow (from the ensemble eval)
  verdict        : BIAS-dom  if |m_ens| >> se_ens (more seeds can't move it)
                   NOISE-dom if |m_ens| comparable to sigma_seed (more seeds shrink it ~1/sqrt(N))

Bottom line for "would more seeds help": a residual only shrinks with more seeds to the extent
sigma_seed is a large fraction of |m_ens|. If |m_ens| is many sigma_seed away from 0 (large z on
the ensemble AND small sigma_seed), it is a systematic bias and seeds are futile.
"""
import argparse, json, math
import numpy as np


def load(path):
    rows = json.load(open(path))
    return {r["label"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-rows", nargs="+", required=True, help="single-seed rows.json files")
    ap.add_argument("--ens-rows", required=True, help="K-seed ensemble rows.json")
    ap.add_argument("--acceptance-only", action="store_true",
                    help="restrict to the converged acceptance family (mag_cum/mag_win/size_cum + "
                         "mag_x_size with size>=thresh); drop small-size-only diagnostics")
    args = ap.parse_args()

    seeds = [load(p) for p in args.seed_rows]
    ens = load(args.ens_rows)
    K = len(seeds)

    # Converged acceptance family: true mag cuts/windows + resolution LOWER-cuts (size_gt / large
    # size windows) + mag x (size>thresh). Small-size-only keeps are diagnostics, not acceptance.
    def is_acceptance(lab, kind):
        if kind in ("global", "mag_cum", "size_cum"):
            return True
        if kind == "mag_win":
            return True  # magnitude windows are on true brightness -> acceptance
        if kind == "env":
            return True
        if kind == "size_win":
            # keep only the resolution-safe (large) windows; drop near-PSF small keeps
            return ("1.0-1.5" in lab) or ("0.75" in lab) or ("_gt" in lab)
        if kind == "mag_x_size":
            return "sizeGt" in lab or "sizegt" in lab  # only size>thresh combos
        if kind == "mag_x_blend":
            return True
        return False

    labels = [l for l in ens if all(l in s for s in seeds)]
    print(f"# seeds={K}  labels_common={len(labels)}")
    hdr = f"{'label':28s} {'kind':12s} {'m_ens%':>8s} {'z_ens':>6s} {'sig_seed%':>9s} " \
          f"{'se_ens%':>8s} {'merr%':>7s} {'|m|/sig':>7s}  verdict"
    print(hdr); print("-" * len(hdr))

    rows_out = []
    for lab in labels:
        e = ens[lab]
        kind = e.get("kind", "?")
        if args.acceptance_only and not is_acceptance(lab, kind):
            continue
        ms = np.array([s[lab]["m"] for s in seeds], float)
        sig = float(ms.std(ddof=1)) if K > 1 else float("nan")
        se_ens = sig / math.sqrt(K)
        m_ens = float(e["m"]); merr = float(e["merr"]); z_ens = float(e["z"])
        ratio = abs(m_ens) / sig if sig > 0 else float("inf")
        # verdict: seeds only help if sigma_seed is a meaningful fraction of |m_ens|.
        if ratio >= 4:
            verdict = "BIAS-dom (seeds futile)"
        elif ratio >= 2:
            verdict = "mostly-bias"
        else:
            verdict = "NOISE-dom (seeds help)"
        rows_out.append((abs(m_ens), lab, kind, m_ens, z_ens, sig, se_ens, merr, ratio, verdict))

    for _, lab, kind, m_ens, z_ens, sig, se_ens, merr, ratio, verdict in sorted(rows_out, reverse=True):
        print(f"{lab:28s} {kind:12s} {m_ens*100:+8.3f} {z_ens:6.1f} {sig*100:9.3f} "
              f"{se_ens*100:8.3f} {merr*100:7.3f} {ratio:7.1f}  {verdict}")

    # Global projection: to what does the worst acceptance residual fall with more seeds?
    accept = [r for r in rows_out if r[9] != "BIAS-dom (seeds futile)"]
    print("\n# --- projection: floor set by BIAS is untouchable by seeds ---")
    if rows_out:
        worst_bias = max(rows_out, key=lambda r: (r[9].startswith("BIAS")) * r[0])
        wb_absm = worst_bias[0]
        print(f"# worst BIAS-dominated acceptance residual : {worst_bias[1]} "
              f"|m|={wb_absm*100:.2f}%  -> this is the seed-independent floor")
    for r in sorted(accept, reverse=True)[:6]:
        absm, lab, kind, m_ens, z_ens, sig, se_ens, merr, ratio, verdict = r
        # naive N to make seed SE < 0.1% (if noise-dominated)
        needN = (sig / 0.001) ** 2 if sig > 0 else 0
        print(f"# {lab:24s} |m_ens|={absm*100:.2f}% sig_seed={sig*100:.2f}% "
              f"-> seeds to push seed-SE<0.10%%: N~{needN:.0f}")


if __name__ == "__main__":
    main()
