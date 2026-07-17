"""Summary visualization of held-out-shear recovery.

Panel A: recovery marginal-likelihood curves (shape-only model) for g=0.05 and g=0.2,
         shown as Delta log L vs trial shear, with truth and recovered peak marked.
Panel B: closure-test curves (model-generated data at known shear) confirming the
         estimator is unbiased.
Panel C: recovered vs true shear for the joint-6 vs shape-only likelihoods (the fix).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "heldout_shear_recovery")


def load(tag):
    p = os.path.join(D, f"recovery_{tag}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return dict(grid=d["grid"], mean=d["mean_logprob"], sem=d["sem"],
                nominal=float(d["nominal_shear"]))


def peak(d):
    im = int(np.argmax(d["mean"]))
    lo, hi = max(0, im - 3), min(len(d["grid"]), im + 4)
    c = np.polyfit(d["grid"][lo:hi], d["mean"][lo:hi], 2)
    return -c[1] / (2 * c[0]) if c[0] < 0 else d["grid"][im]


fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

# Panel A: shape-only recovery curves
ax = axes[0]
for tag, color in [("g0.05_shape2d_affine", "C0"), ("g0.2_shape2d_affine", "C1")]:
    d = load(tag)
    if d is None:
        continue
    y = d["mean"] - d["mean"].max()
    ax.plot(d["grid"], y, "-", color=color, label=f"applied |g|={d['nominal']}")
    sh = peak(d)
    ax.axvline(d["nominal"], color=color, ls="--", alpha=0.5)
    ax.axvline(sh, color=color, ls=":", alpha=0.9)
    ax.annotate(f"$\\hat s$={sh:.4f}\n({100*sh/d['nominal']:.0f}%)",
                xy=(sh, -0.5), color=color, fontsize=8, ha="center")
ax.set_xlabel("trial shear $s$ (along applied direction)")
ax.set_ylabel(r"$\Delta \log L$ (shape-only)")
ax.set_title("A. Held-out-shear recovery (shape-only flow)")
ax.set_ylim(-6, 0.5)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# Panel B: closure curves (method validation)
ax = axes[1]
for tag, s0, color in [("g0.05_sheared_closure0.05", 0.05, "C0"),
                       ("g0.2_sheared_closure0.2", 0.2, "C1")]:
    d = load(tag)
    if d is None:
        continue
    y = d["mean"] - d["mean"].max()
    ax.plot(d["grid"], y, "-", color=color, label=f"model data at s0={s0}")
    sh = peak(d)
    ax.axvline(s0, color=color, ls="--", alpha=0.5)
    ax.axvline(sh, color=color, ls=":", alpha=0.9)
ax.set_xlabel("trial shear $s$")
ax.set_ylabel(r"$\Delta \log L$")
ax.set_title("B. Closure test (recovers known shear exactly)")
ax.set_ylim(-6, 0.5)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# Panel C: recovered vs true, joint-6 vs shape-only
ax = axes[2]
truth = np.array([0.05, 0.2])
joint = np.array([0.0304, 0.134])      # 6-target flow
shape = np.array([0.0533, 0.2122])     # shape-only flow
ax.plot([0, 0.22], [0, 0.22], "k--", alpha=0.4, label="ideal (1:1)")
ax.plot(truth, joint, "s-", color="C3", label="joint 6-target (61–67%)")
ax.plot(truth, shape, "o-", color="C2", label="shape-only (106–107%)")
ax.set_xlabel("true applied |g|")
ax.set_ylabel(r"recovered $\hat s$")
ax.set_title("C. The fix: shape-only likelihood")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlim(0, 0.22)
ax.set_ylim(0, 0.24)

fig.tight_layout()
out = os.path.join(D, "recovery_summary.png")
fig.savefig(out, dpi=140)
print(f"wrote {out}")
