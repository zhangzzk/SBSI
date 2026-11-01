"""Figure: faint e2 excess by stamp variant, and paired (variant - production) e2 shifts, per MAG_AUTO band."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

exc = json.load(open(sys.argv[1]))["results"]
pair = json.load(open(sys.argv[2]))["results"]
out = sys.argv[3]
bands = ["0.0-23.0", "23.0-24.0", "24.0-25.0", "25.0-25.8", "25.8-99.0"]
labels = ["<23", "23–24", "24–25", "25–25.8", ">25.8"]
x = np.arange(len(bands))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, (v, c) in enumerate([("cat", "0.5"), ("prod48", "C3"), ("cen49", "C0"), ("cen65", "C2")]):
    y = [exc[f"{v}|{b}"]["e2"] * 1e4 for b in bands]
    e = [max(exc[f"{v}|{b}"]["e2_err_cases"], exc[f"{v}|{b}"]["e2_err_galaxies"]) * 1e4 for b in bands]
    ax[0].errorbar(x + (i - 1.5) * 0.08, y, e, fmt="o", color=c, label=v, capsize=2)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set(xticks=x, xticklabels=labels, xlabel="MAG_AUTO", ylabel="mean measured − true e2  [1e-4]",
          title="e2 excess, same galaxies (16 ConstGold cases, +leg)")
ax[0].legend(fontsize=8)
for i, (p, c) in enumerate([("cen49-prod48", "C0"), ("cen65-prod48", "C2")]):
    y = [pair[f"{p}|{b}"]["dg2"] * 1e4 for b in bands]
    e = [max(pair[f"{p}|{b}"]["dg2_err_cases"], pair[f"{p}|{b}"]["dg2_err_galaxies"]) * 1e4 for b in bands]
    ax[1].errorbar(x + (i - 0.5) * 0.1, y, e, fmt="o", color=c, label=p, capsize=2)
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set(xticks=x, xticklabels=labels, xlabel="MAG_AUTO", ylabel="paired Δe2 per galaxy  [1e-4]",
          title="centred stamp − production 48 px stamp")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=130)
print(out)
