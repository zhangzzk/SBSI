import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _data import OUT, RESPONSE_BINS, load_models, load_responses


def profile(frame, x_column, measured_column, model_column):
    x = frame[x_column].to_numpy(float)
    residual = frame[model_column].to_numpy(float) - frame[measured_column].to_numpy(float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, RESPONSE_BINS + 1)))
    groups = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for index in range(len(edges) - 1):
        take = groups == index
        case_means = pd.Series(residual[take]).groupby(
            frame.loc[take, "case"].to_numpy()
        ).mean()
        rows.append(
            {
                "bin": index,
                "x": np.median(x[take]),
                "n": int(take.sum()),
                "residual": case_means.mean(),
                "se": case_means.sem(),
            }
        )
    return pd.DataFrame(rows)


self_response, blend_response = load_responses()
self_response, blend_response = load_models(self_response, blend_response)
catalogues = (
    ("R_self", self_response, "R_self", "R_self_model", "#D55E00", "o"),
    ("R_blend", blend_response, "R_blend", "R_blend_model", "#0072B2", "s"),
)
profiles = {
    (name, condition): profile(frame, x, measured, model)
    for name, frame, measured, model, _color, _marker in catalogues
    for condition, x in (("prediction", model), ("measurement", measured))
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
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), layout="constrained")
for label, ax, condition, xlabel in zip(
    "AB",
    axes,
    ("prediction", "measurement"),
    (r"Model-predicted response $R_{model}$", r"Measured response $R_{measured}$"),
):
    ax.axhline(0, color="0.25", linewidth=1)
    ax.grid(axis="y", color="0.91", linewidth=0.7)
    for name, _frame, _measured, _model, color, marker in catalogues:
        table = profiles[name, condition]
        ax.errorbar(
            table.x,
            table.residual,
            yerr=table.se,
            fmt=marker + "-",
            color=color,
            markersize=4,
            capsize=1.8,
        )
    ax.set(xlabel=xlabel, ylabel=r"$R_{model}-R_{measured}$")
    ax.set_title(f"Conditioned on {condition}")
    ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold")

fig.legend(
    handles=[
        Line2D([0], [0], color="#D55E00", marker="o", label=r"$R_{self}$"),
        Line2D([0], [0], color="#0072B2", marker="s", label=r"$R_{blend}$"),
    ],
    loc="outside lower center",
    ncol=2,
    frameon=False,
)
fig.suptitle(r"Response residual on the fixed $g=0$ domain")
for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"residual.{suffix}", bbox_inches="tight")
pd.concat(
    [table.assign(catalogue=name, conditioned_on=condition) for (name, condition), table in profiles.items()]
).to_csv(OUT / "residual.csv", index=False)
print(f"wrote {OUT / 'residual.png'}")
