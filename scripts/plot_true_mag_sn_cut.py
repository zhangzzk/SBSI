"""Plot the true-magnitude population before and after the intrinsic S/N cut.

The comparison isolates exactly one change.  Both histograms use input
primaries satisfying the ordinary magnitude/size support and the V2.1 size
cut; the second additionally requires ``sn_true > 10``.  No detection flag or
measured property is read.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from sbs_shear import domain as D
from sbs_shear.paths import catalogue


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--catalogue",
        default=catalogue("det_meas_crowd_conc_g0.0_train_full.feather"),
    )
    ap.add_argument("--output", default="results/true_mag_before_after_sn10.png")
    ap.add_argument("--pdf", default="results/true_mag_before_after_sn10.pdf")
    ap.add_argument("--summary", default="results/true_mag_before_after_sn10.json")
    ap.add_argument("--bin-width", type=float, default=0.05)
    args = ap.parse_args()

    for path in (args.output, args.pdf, args.summary):
        if os.path.exists(path):
            raise SystemExit(f"REFUSING to overwrite existing output: {path}")

    mag_min, mag_max = 18.0, 28.0
    re_min, re_max = D.V21_RE_MIN, 1.5
    edges = np.arange(mag_min, mag_max + args.bin_width * 0.5, args.bin_width)
    before = np.zeros(len(edges) - 1, dtype=np.int64)
    after = np.zeros_like(before)
    n_rows = n_before = n_after = 0

    with ipc.open_file(args.catalogue) as reader:
        for bi in range(reader.num_record_batches):
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(
                [D.MAG_COLUMN, D.RE_COLUMN]
            )
            mag = batch[D.MAG_COLUMN].to_numpy(zero_copy_only=False).astype(float)
            re = batch[D.RE_COLUMN].to_numpy(zero_copy_only=False).astype(float)
            n_rows += len(mag)
            base = (
                np.isfinite(mag)
                & np.isfinite(re)
                & (mag > mag_min)
                & (mag < mag_max)
                & (re > re_min)
                & (re < re_max)
            )
            keep = base & (D.sn_true(mag, re) > D.V21_SN_MIN)
            before += np.histogram(mag[base], bins=edges)[0]
            after += np.histogram(mag[keep], bins=edges)[0]
            n_before += int(base.sum())
            n_after += int(keep.sum())
            if (bi + 1) % 50 == 0 or bi + 1 == reader.num_record_batches:
                print(
                    f"batches {bi + 1}/{reader.num_record_batches}: "
                    f"rows={n_rows:,}, before={n_before:,}, after={n_after:,}",
                    flush=True,
                )

    retention = np.divide(
        after, before, out=np.full_like(after, np.nan, dtype=float), where=before > 0
    )
    limits = {
        str(re): float(D.sn_limiting_mag(re)) for re in (0.5, 0.7, 1.0, 1.5)
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.0, 5.3), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.08},
    )
    grey, blue = "#777777", "#0072B2"
    ax.stairs(
        before / 1e6, edges, fill=True, color=grey, alpha=0.28,
        linewidth=1.2, label=f'Before intrinsic S/N cut  (N={n_before/1e6:.2f}M)',
    )
    ax.stairs(
        after / 1e6, edges, fill=True, color=blue, alpha=0.50,
        linewidth=1.4,
        label=(f'After intrinsic proxy S/N > {D.V21_SN_MIN:g}  '
               f'(N={n_after/1e6:.2f}M)'),
    )
    ax.set_ylabel(f"Input galaxies per {args.bin_width:g} mag (millions)")
    ax.set_title("True-magnitude distribution before and after the V2.1 intrinsic S/N cut")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.02, 0.71,
        f'Fixed in both: {mag_min:g} < true r < {mag_max:g}, '
        f'{re_min:g}\" < true $R_e$ < {re_max:g}\"\n'
        "No detection, measured S/N, or blending quantity is used",
        transform=ax.transAxes, va="top", color="#333333",
    )

    centres = 0.5 * (edges[:-1] + edges[1:])
    axr.plot(centres, retention, color=blue, linewidth=1.8)
    axr.fill_between(centres, 0, retention, color=blue, alpha=0.16)
    axr.set_ylim(0, 1.04)
    axr.set_yticks([0, 0.5, 1.0])
    axr.set_ylabel("Fraction kept")
    axr.set_xlabel("True input LSST r magnitude  (fainter →)")

    for re, maglim in limits.items():
        for axis in (ax, axr):
            axis.axvline(maglim, color="#333333", linestyle="--", linewidth=0.8, alpha=0.5)
        axr.text(
            maglim - 0.015, 0.04, f'$R_e$={float(re):g}\"', rotation=90,
            ha="right", va="bottom", fontsize=7.5, color="#333333",
        )

    for axis in (ax, axr):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_xlim(mag_min, mag_max)
    fig.tight_layout()

    for path in (args.output, args.pdf, args.summary):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "catalogue": args.catalogue,
        "definition": {
            "before": f"{mag_min}<r_input_p<{mag_max}, {re_min}<Re_input_p<{re_max}",
            "after": f"before AND sn_true>{D.V21_SN_MIN}",
            "sn_proxy": D.metadata(),
            "uses_detection_or_measurement": False,
        },
        "rows_read": n_rows,
        "n_before": n_before,
        "n_after": n_after,
        "fraction_after": n_after / n_before,
        "sn10_limiting_magnitude_by_re_arcsec": limits,
        "bin_edges": edges.tolist(),
        "hist_before": before.tolist(),
        "hist_after": after.tolist(),
        "retention": retention.tolist(),
    }
    with open(args.summary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.pdf}")
    print(f"wrote {args.summary}")
    print("TRUE_MAG_SN_HIST_DONE", flush=True)


if __name__ == "__main__":
    main()
