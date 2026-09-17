import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _data import OUT, RESPONSE_AXES, RESPONSE_BINS, load_responses


def profile(frame, x_column, response_column, null_column):
    edges = np.unique(
        np.quantile(frame[x_column], np.linspace(0, 1, RESPONSE_BINS + 1))
    )
    groups = np.clip(
        np.searchsorted(edges, frame[x_column], side="right") - 1,
        0,
        len(edges) - 2,
    )
    rows = []
    for index in range(len(edges) - 1):
        group = frame.loc[groups == index]
        case_means = group.groupby("case")[[response_column, null_column]].mean()
        rows.append(
            {
                "bin": index,
                "x": group[x_column].median(),
                "n": len(group),
                "response": case_means[response_column].mean(),
                "response_se": case_means[response_column].sem(),
                "null": case_means[null_column].mean(),
            }
        )
    return pd.DataFrame(rows)


self_response, blend_response = load_responses()
catalogues = (
    ("R_self", self_response, "R_self", "R_self_cross", "#D55E00"),
    ("R_blend", blend_response, "R_blend", "R_blend_cross", "#0072B2"),
)
profiles = {
    (name, x): profile(frame, x, response, null)
    for name, frame, response, null, _color in catalogues
    for x, _label, _log in RESPONSE_AXES
}

mpl.rcParams.update(
    {
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), sharey="row", layout="constrained")
labels = iter("ABCDEFGH")
for row, (name, frame, _response, _null, color) in enumerate(catalogues):
    for column, (x_column, xlabel, log_x) in enumerate(RESPONSE_AXES):
        ax = axes[row, column]
        table = profiles[name, x_column]
        ax.axhline(0, color="0.82", linewidth=0.9)
        ax.errorbar(
            table.x,
            table.response,
            yerr=table.response_se,
            fmt="o-",
            color=color,
            markersize=3.2,
            capsize=1.8,
        )
        ax.plot(table.x, table.null, "o:", color="0.42", markersize=2.7)
        if x_column == "g0_MAG_AUTO":
            ax.axvline(25.8, color="#009E73", linestyle=":")
            ax.set_xlim(table.x.min() - 0.35, 25.82)
        if x_column == "g0_FLUX_RADIUS_arcsec":
            ax.axvline(0.6, color="#009E73", linestyle=":")
            ax.set_xlim(0.59, table.x.max() * 1.08)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        if column == 0:
            ax.set_ylabel(
                r"$R_{self}$ finite response"
                if name == "R_self"
                else r"$R_{blend}$ finite response"
            )
        if row == 0:
            ax.set_title(xlabel.replace(r"$g=0$ ", ""))
        ax.text(0.02, 0.98, next(labels), transform=ax.transAxes, va="top", fontweight="bold")
        ax.text(0.98, 0.98, f"{name}\nN={len(frame):,}", transform=ax.transAxes, ha="right", va="top", color=color)

fig.legend(
    handles=[
        Line2D([0], [0], color="#D55E00", marker="o", label=r"$R_{self}$"),
        Line2D([0], [0], color="#0072B2", marker="o", label=r"$R_{blend}$"),
        Line2D([0], [0], color="0.42", marker="o", linestyle=":", label="Perpendicular null"),
        Line2D([0], [0], color="#009E73", linestyle=":", label=r"Measured $g=0$ boundary"),
    ],
    loc="outside lower center",
    ncol=4,
    frameon=False,
)
fig.suptitle(r"Measured responses on the fixed $g=0$ domain")
for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"response.{suffix}", bbox_inches="tight")
pd.concat(
    [table.assign(catalogue=name, x_quantity=x) for (name, x), table in profiles.items()]
).to_csv(OUT / "response.csv", index=False)
print(f"wrote {OUT / 'response.png'}")
