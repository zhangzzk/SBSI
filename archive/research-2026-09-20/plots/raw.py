import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _data import OUT, load_raw


BINS = 32
HEATMAP_BINS = 90
METRICS = (
    ("radius", "Re_input", "measured_radius_arcsec", "Radius", r"True intrinsic $R_e$ (arcsec)", r"Measured FLUX_RADIUS (arcsec)", True, False),
    ("magnitude", "r_input", "MAG_AUTO", "Flux (magnitude)", r"True $r$ magnitude", "Measured MAG_AUTO", False, True),
    ("e1", "e1_input_rot0", "NGMIX_G1", r"Ellipticity $e_1$", r"True intrinsic $e_1$", "Measured NGMIX_G1", False, True),
    ("e2", "e2_input_rot0", "NGMIX_G2", r"Ellipticity $e_2$", r"True intrinsic $e_2$", "Measured NGMIX_G2", False, True),
)


def profile(x, y, case):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, BINS + 1)))
    groups = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for index in range(len(edges) - 1):
        take = groups == index
        xb, yb, cb = x[take], y[take], case[take]
        case_medians = np.array([np.median(yb[cb == value]) for value in np.unique(cb)])
        q025, q16, median, q84, q975 = np.quantile(yb, [0.025, 0.16, 0.5, 0.84, 0.975])
        rows.append(
            {
                "bin": index,
                "x": np.median(xb),
                "n": len(yb),
                "q025": q025,
                "q16": q16,
                "median": median,
                "q84": q84,
                "q975": q975,
                "se": case_medians.std(ddof=1) / np.sqrt(len(case_medians)),
            }
        )
    return pd.DataFrame(rows)


def heatmap(x, y, log_axes):
    xlow, xhigh = np.quantile(x, [0.001, 0.999])
    ylow, yhigh = np.quantile(y, [0.001, 0.999])
    if log_axes:
        xedges = np.geomspace(xlow, xhigh, HEATMAP_BINS + 1)
        yedges = np.geomspace(ylow, yhigh, HEATMAP_BINS + 1)
    else:
        xedges = np.linspace(xlow, xhigh, HEATMAP_BINS + 1)
        yedges = np.linspace(ylow, yhigh, HEATMAP_BINS + 1)
    hist, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
    shown = ((x >= xlow) & (x <= xhigh) & (y >= ylow) & (y <= yhigh)).mean()
    return hist, xedges, yedges, shown


data, audit = load_raw()
cases = data["case"].to_numpy()
profiles = {}
heatmaps = {}
for key, truth, measured, *_rest, log_axes, _identity in METRICS:
    x = data[truth].to_numpy(float)
    y = data[measured].to_numpy(float)
    profiles[key] = profile(x, y, cases)
    heatmaps[key] = heatmap(x, y, log_axes)

mpl.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)
orange = "#D55E00"
blue = "#56B4E9"
norm = LogNorm(vmin=1, vmax=max(values[0].max() for values in heatmaps.values()))
fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2), layout="constrained")

for label, ax, metric in zip("ABCD", axes.flat, METRICS):
    key, _truth, _measured, title, xlabel, ylabel, log_axes, identity = metric
    hist, xedges, yedges, shown = heatmaps[key]
    table = profiles[key]
    mesh = ax.pcolormesh(
        xedges, yedges, hist.T, cmap="cividis", norm=norm, shading="auto", rasterized=True
    )
    ax.fill_between(table.x, table.q025, table.q975, color=orange, alpha=0.13)
    ax.fill_between(table.x, table.q16, table.q84, color=orange, alpha=0.28)
    ax.errorbar(table.x, table["median"], yerr=table.se, fmt="o-", color=orange, markersize=2.5, capsize=1.5)
    if identity:
        low, high = max(xedges[0], yedges[0]), min(xedges[-1], yedges[-1])
        ax.plot([low, high], [low, high], "--", color="0.25", linewidth=1)
    if key == "radius":
        ax.axvline(0.37, color=blue, linestyle=":")
        ax.axhline(0.50, color=blue, linestyle="--")
        ax.axhline(0.75, color=blue, linestyle="-.")
    if key == "magnitude":
        ax.axvline(25.8, color=blue, linestyle=":")
        ax.axhline(25.8, color=blue, linestyle=":")
    if log_axes:
        ax.set(xscale="log", yscale="log")
    ax.set(xlim=(xedges[0], xedges[-1]), ylim=(yedges[0], yedges[-1]), title=title, xlabel=xlabel, ylabel=ylabel)
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        color="white",
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
    )
    ax.text(0.98, 0.02, f"{100 * shown:.2f}% shown", transform=ax.transAxes, ha="right", color="white", fontsize=8)

fig.colorbar(mesh, ax=axes, label="Galaxies per 2D bin", shrink=0.88)
fig.legend(
    handles=[
        Line2D([0], [0], color=orange, marker="o", label="Conditional median"),
        Patch(facecolor=orange, alpha=0.28, label="Central 68%"),
        Patch(facecolor=orange, alpha=0.13, label="Central 95%"),
    ],
    loc="outside lower center",
    ncol=3,
    frameon=False,
)
fig.suptitle(r"Raw $g=0$ measurements versus truth")

for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"raw.{suffix}", bbox_inches="tight")
pd.concat([table.assign(quantity=key) for key, table in profiles.items()]).to_csv(
    OUT / "raw.csv", index=False
)
audit.to_csv(OUT / "raw_audit.csv", index=False)
print(f"wrote {OUT / 'raw.png'}")
