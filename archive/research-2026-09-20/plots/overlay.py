import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _data import OUT, RESPONSE_AXES, RESPONSE_BINS, load_models, load_responses


def profile(frame, x_column, measured_column, model_column):
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
        means = group.groupby("case")[[measured_column, model_column]].mean()
        residual = means[model_column] - means[measured_column]
        rows.append(
            {
                "bin": index,
                "x": group[x_column].median(),
                "n": len(group),
                "measured": means[measured_column].mean(),
                "measured_se": means[measured_column].sem(),
                "model": means[model_column].mean(),
                "model_se": means[model_column].sem(),
                "residual": residual.mean(),
                "residual_se": residual.sem(),
            }
        )
    return pd.DataFrame(rows)


self_response, blend_response = load_responses()
self_response, blend_response = load_models(self_response, blend_response)
catalogues = (
    ("R_self", self_response, "R_self", "R_self_model", "#D55E00"),
    ("R_blend", blend_response, "R_blend", "R_blend_model", "#0072B2"),
)
profiles = {
    (name, x): profile(frame, x, measured, model)
    for name, frame, measured, model, _color in catalogues
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
fig = plt.figure(figsize=(16, 9.4), layout="constrained")
grid = fig.add_gridspec(4, 4, height_ratios=(3.2, 1, 3.2, 1))
main_axes = np.empty((2, 4), dtype=object)
labels = iter("ABCDEFGH")

for row, (name, frame, _measured, _model, color) in enumerate(catalogues):
    for column, (x_column, xlabel, log_x) in enumerate(RESPONSE_AXES):
        main = fig.add_subplot(grid[2 * row, column], sharey=main_axes[row, 0] if column else None)
        residual = fig.add_subplot(grid[2 * row + 1, column], sharex=main)
        main_axes[row, column] = main
        table = profiles[name, x_column]
        main.axhline(0, color="0.86", linewidth=0.8)
        main.errorbar(
            table.x,
            table.measured,
            yerr=table.measured_se,
            fmt="o-",
            color=color,
            markersize=3.2,
            capsize=1.7,
        )
        main.plot(table.x, table.model, "s--", color="black", markersize=2.8)
        main.fill_between(
            table.x,
            table.model - table.model_se,
            table.model + table.model_se,
            color="black",
            alpha=0.09,
        )
        residual.axhline(0, color="0.35", linewidth=0.9)
        residual.errorbar(
            table.x,
            table.residual,
            yerr=table.residual_se,
            fmt="o-",
            color=color,
            markersize=2.6,
            capsize=1.5,
        )
        if x_column == "g0_MAG_AUTO":
            main.axvline(25.8, color="#009E73", linestyle=":")
            residual.axvline(25.8, color="#009E73", linestyle=":")
            main.set_xlim(table.x.min() - 0.35, 25.82)
        if x_column == "g0_FLUX_RADIUS_arcsec":
            main.axvline(0.6, color="#009E73", linestyle=":")
            residual.axvline(0.6, color="#009E73", linestyle=":")
            main.set_xlim(0.59, table.x.max() * 1.08)
        if log_x:
            main.set_xscale("log")
        main.tick_params(labelbottom=False)
        residual.set_xlabel(xlabel)
        if column == 0:
            main.set_ylabel(r"$R_{self}$" if name == "R_self" else r"$R_{blend}$")
            residual.set_ylabel("model - measured", fontsize=8)
        if row == 0:
            main.set_title(xlabel.replace(r"$g=0$ ", ""))
        main.text(0.02, 0.98, next(labels), transform=main.transAxes, va="top", fontweight="bold")
        main.text(0.98, 0.98, f"{name}\nN={len(frame):,}", transform=main.transAxes, ha="right", va="top", color=color)

fig.legend(
    handles=[
        Line2D([0], [0], color="#D55E00", marker="o", label=r"Measured $R_{self}$"),
        Line2D([0], [0], color="#0072B2", marker="o", label=r"Measured $R_{blend}$"),
        Line2D([0], [0], color="black", marker="s", linestyle="--", label="Model"),
    ],
    loc="outside lower center",
    ncol=3,
    frameon=False,
)
fig.suptitle(r"Measured and predicted response on the fixed $g=0$ domain")
for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"overlay.{suffix}", bbox_inches="tight")
pd.concat(
    [table.assign(catalogue=name, x_quantity=x) for (name, x), table in profiles.items()]
).to_csv(OUT / "overlay.csv", index=False)
print(f"wrote {OUT / 'overlay.png'}")
