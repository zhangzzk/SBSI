"""WHERE does the blend emulator go wrong? A fine error map off the g=0.2 ruler.

2026-08-05b established the deficit (-6.15%, 3.6 sigma) but not its shape. A retrain aimed at
"make it better" is a blind sweep; a retrain aimed at a named failure mode is an experiment. This
maps the per-pair error across every axis the emulator conditions on, so the next training change
has a prediction attached to it.

WHY THE g=0.2 RULER. Paired precision is +-1.73% there against +-4.26% at g=0.05, and the two legs
agree on truth to 0.6 sigma (linearity checked, 2026-08-05b), so the finer slices below are actually
resolvable rather than noise.

THE ERROR IS A PAIRED DIFFERENCE. truth and pred sit on the same rows, so every cell reports
mean(pred - truth) with the sem of that DIFFERENCE. Using truth's own sem -- what the ruler's table
prints -- would overstate the error on a comparison that shares its rows.

WEIGHTED, NOT JUST RELATIVE. A cell can be 50% wrong and irrelevant if its response is tiny. Each
table therefore also reports the cell's SHARE of the total absolute deficit, so the retrain targets
what actually moves R_blend rather than what has the worst percentage.

FIREWALL: reads ruler outputs (half-shear legs only). No constgold, no fitting, no promotion.
"""
from __future__ import annotations

import argparse

import numpy as np


def cells(truth, pred, key, edges, name, label):
    diff = pred - truth
    tot = diff.sum()
    print(f"\n[{label}]")
    print(f"  {name:>22}{'truth':>9}{'emu':>9}{'diff':>9}{'rel %':>9}{'+-':>8}{'sig':>7}"
          f"{'share of deficit':>18}{'N':>12}")
    for i in range(len(edges) - 1):
        m = (key >= edges[i]) & (key < edges[i + 1])
        n = int(m.sum())
        if n < 500:
            continue
        d = diff[m]
        sem = d.std(ddof=1) / np.sqrt(n)
        rel = 100 * (pred[m].mean() / truth[m].mean() - 1) if truth[m].mean() else np.nan
        share = 100 * d.sum() / tot if tot else np.nan
        print(f"  [{edges[i]:>9.3f},{edges[i+1]:>8.3f}){truth[m].mean():>9.4f}{pred[m].mean():>9.4f}"
              f"{d.mean():>+9.4f}{rel:>+9.2f}{sem:>8.4f}{abs(d.mean())/sem:>6.1f}s"
              f"{share:>17.1f}%{n:>12,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    z = np.load(args.npz)
    t, p = z["truth"], z["pred"]
    good = np.isfinite(t) & np.isfinite(p)
    t, p = t[good], p[good]
    d, s, mg = z["distance"][good], z["size"][good], z["mag"][good]

    diff = p - t
    sem = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"\n{'='*104}\nEMULATOR ERROR MAP  {args.label}\n{'='*104}")
    print(f"N={len(t):,}   <truth>={t.mean():.4f}   <emu>={p.mean():.4f}   "
          f"diff={diff.mean():+.5f} +- {sem:.5f} ({abs(diff.mean())/sem:.1f} sigma)   "
          f"rel={100*(p.mean()/t.mean()-1):+.2f}%")
    print("'share of deficit' = this cell's contribution to the TOTAL absolute error. A cell with a")
    print("huge rel % but a tiny share is not what to fix; chase the shares.")

    cells(t, p, d, [0, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7, 10],
          "distance (arcsec)", "by PAIR SEPARATION")
    cells(t, p, s, [.5, .6, .7, .8, .9, 1.0, 1.2, 1.5], "Re_input_p", "by PRIMARY TRUE SIZE")
    cells(t, p, mg, [18, 21, 22, 23, 23.5, 24, 24.5, 25, 25.4, 25.72],
          "r_input_p", "by PRIMARY TRUE MAG")

    # The emulator's own close-pair regime, resolved: below 1" is where 2026-08-05a found -30%, and
    # it is also where deblending physically breaks down. Split it finely enough to see the shape.
    m = d < 1.5
    cells(t[m], p[m], d[m], [0, .2, .35, .5, .65, .8, .95, 1.1, 1.25, 1.5],
          "distance (arcsec)", "CLOSE PAIRS ONLY (under 1.5 arcsec), finely resolved")

    print(f"\n{'='*104}\nJOINT: separation x primary magnitude (the two axes that carry the deficit)"
          f"\n{'='*104}")
    dedges = [0, 1, 2, 3, 10]
    medges = [18, 23, 24, 25, 25.72]
    print(f"  {'':>16}" + "".join(f"{f'mag {medges[j]}-{medges[j+1]}':>18}"
                                  for j in range(len(medges) - 1)))
    for i in range(len(dedges) - 1):
        lab = 'd {}-{}"'.format(dedges[i], dedges[i + 1])
        row = f"  {lab:>16}"
        for j in range(len(medges) - 1):
            m = ((d >= dedges[i]) & (d < dedges[i + 1])
                 & (mg >= medges[j]) & (mg < medges[j + 1]))
            if m.sum() < 500:
                row += f"{'--':>18}"
                continue
            dd = diff[m]
            se = dd.std(ddof=1) / np.sqrt(m.sum())
            row += f"{f'{dd.mean():+.4f}({abs(dd.mean())/se:.1f}s)':>18}"
        print(row)
    print("\nRBLEND_ERROR_MAP_DONE", flush=True)


if __name__ == "__main__":
    main()
