"""Render the SBSI concept banner: one sheared galaxy field, no primary/secondary
distinction, two apertures with pair connectors, formula R = R_self + sum R_blend.

Pure numpy/matplotlib Sersic rendering (illustration only, no pipeline code).
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

PIX = 0.2            # arcsec / pixel
NX, NY = 264, 88     # 52.8" x 17.6"
THETA_MAX = 7.0      # arcsec pairing aperture
N_GAL = 30
SS = 3               # supersampling


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def anamorphic(pa, stretch1, stretch2):
    """Linear map that turns a circular profile into one stretched by
    (stretch1, stretch2) along position angle pa."""
    r = rot(pa)
    return r @ np.diag([stretch1, stretch2]) @ r.T


def sersic_stamp(flux, re_arcsec, n, a_map):
    """Evaluate a sheared Sersic profile on its stamp at supersampled scale.

    ``a_map`` sends circular source coordinates to the observed sheared
    ellipse, so the profile is evaluated at ``a_map^{-1} x``.
    """
    step = PIX / SS
    re_pix = re_arcsec / step
    half = max(int(6 * re_pix), 6)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    inv_a = np.linalg.inv(a_map)
    coords = np.stack((xx.ravel(), yy.ravel()), axis=1) @ inv_a.T
    rr = np.sqrt((coords ** 2).sum(axis=1)).reshape(2 * half + 1, 2 * half + 1)
    bn = 1.9992 * n - 0.3271
    img = np.exp(-bn * ((np.maximum(rr, 1e-3) / re_pix) ** (1.0 / n) - 1.0))
    total = img.sum()
    if total <= 0:
        return None, 0, 0
    return img * (flux / total), half, half


def render(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(14, NX - 14, N_GAL)
    y = rng.uniform(9, NY - 9, N_GAL)
    mag = rng.uniform(19.5, 26.2, N_GAL)
    re_arcsec = rng.uniform(0.25, 1.05, N_GAL)
    n_sersic = rng.uniform(1.0, 2.3, N_GAL)
    e_int = rng.uniform(0.0, 0.3, N_GAL)
    pa_int = rng.uniform(0, np.pi, N_GAL)
    g = rng.uniform(0.03, 0.10, N_GAL)
    pa_g = rng.uniform(0, np.pi, N_GAL)

    flux = 10 ** (-0.4 * (mag - 30.0))
    image = np.zeros((NY * SS, NX * SS))
    for i in range(N_GAL):
        # observed shape = intrinsic ellipticity composed with applied shear
        a_map = anamorphic(pa_g[i], 1 + g[i], 1 - g[i]) @ anamorphic(
            pa_int[i], 1.0, 1.0 - e_int[i]
        )
        stamp, half, _ = sersic_stamp(
            flux[i], re_arcsec[i], n_sersic[i], a_map
        )
        if stamp is None:
            continue
        cx, cy = int(round(x[i] * SS)), int(round(y[i] * SS))
        x0, x1 = cx - half, cx + half + 1
        y0, y1 = cy - half, cy + half + 1
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1 = stamp.shape[1] - max(0, x1 - NX * SS)
        sy1 = stamp.shape[0] - max(0, y1 - NY * SS)
        image[max(0, y0):min(NY * SS, y1), max(0, x0):min(NX * SS, x1)] += \
            stamp[sy0:sy1, sx0:sx1]
    image += rng.normal(scale=2.0, size=image.shape)
    image = image.reshape(NY, SS, NX, SS).mean(axis=(1, 3))

    return dict(x=x, y=y, mag=mag, re=re_arcsec, g=g, pa_g=pa_g,
                e=e_int, pa_int=pa_int, image=image)


def pick_two(field):
    """Two well-separated galaxies with a healthy neighbour count."""
    x, y = field["x"], field["y"]
    rmax_pix = THETA_MAX / PIX
    counts = np.array([
        np.sum((np.hypot(x - x[i], y - y[i]) < rmax_pix) & (np.arange(N_GAL) != i))
        for i in range(N_GAL)
    ])
    ok = (counts >= 3) & (y > rmax_pix + 2) & (y < NY - rmax_pix - 2) \
        & (x > rmax_pix + 4) & (x < NX - rmax_pix - 4)
    candidates = np.flatnonzero(ok)
    best = None
    for a in candidates:
        for b in candidates:
            if b <= a:
                continue
            sep = np.hypot(x[a] - x[b], y[a] - y[b])
            if sep > 2.0 * rmax_pix and counts[a] + counts[b] >= 6:
                score = -abs(field["mag"][a] - 22.2) - abs(field["mag"][b] - 22.2)
                if best is None or score > best[0]:
                    best = (score, a, b)
    if best is None:
        raise SystemExit("no good aperture pair for this seed")
    return best[1], best[2]


GOLD = "#f2c14e"
CYAN = "#41d6ff"
WHITE = "#ffffff"


def draw(field, picked, out_path):
    image = field["image"]
    x, y, g, pa_g, re = field["x"], field["y"], field["g"], field["pa_g"], field["re"]
    rmax_pix = THETA_MAX / PIX

    vmin, vmax = ZScaleInterval().get_limits(image)
    fig, ax = plt.subplots(figsize=(16.5, 5.5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image, vmin=float(vmin), vmax=float(vmax),
              cmap="Spectral_r", origin="lower")

    lw_scale = 1.1 + 1.0 * (26.5 - field["mag"]) / 7.0
    for i in range(N_GAL):
        ax.add_patch(Circle(
            (x[i], y[i]), radius=min(42 * re[i], 26),
            edgecolor=WHITE, facecolor="none", linestyle=(0, (4, 3)),
            linewidth=lw_scale[i], alpha=0.95,
        ))
        half = g[i] * 58
        dx, dy = half * np.cos(pa_g[i]), half * np.sin(pa_g[i])
        ax.plot([x[i] - dx, x[i] + dx], [y[i] - dy, y[i] + dy],
                color=CYAN, linewidth=2.0, solid_capstyle="round")

    for t, tag in zip(picked, ("a", "b")):
        ax.add_patch(Circle((x[t], y[t]), radius=rmax_pix,
                            edgecolor=GOLD, facecolor="none", linewidth=2.6))
        ax.add_patch(Circle((x[t], y[t]), radius=min(42 * re[t], 26),
                            edgecolor=GOLD, facecolor="none", linewidth=2.6))
        half = g[t] * 58
        dx, dy = half * np.cos(pa_g[t]), half * np.sin(pa_g[t])
        ax.plot([x[t] - dx, x[t] + dx], [y[t] - dy, y[t] + dy],
                color=CYAN, linewidth=2.9, solid_capstyle="round")
        for j in range(N_GAL):
            if j == t:
                continue
            if np.hypot(x[j] - x[t], y[j] - y[t]) < rmax_pix:
                ax.plot([x[t], x[j]], [y[t], y[j]], color=GOLD,
                        linestyle=(0, (5, 4)), linewidth=1.8, alpha=0.95)

    # aperture label on the first circle
    ax.annotate(r"$\theta_{\max}$",
                xy=(x[picked[0]] + rmax_pix * np.cos(0.8),
                    y[picked[0]] + rmax_pix * np.sin(0.8)),
                xytext=(14, 10), textcoords="offset points",
                color=GOLD, fontsize=15)

    ax.text(0.022, 0.82,
            r"$R \;=\; R_{\mathrm{self}} \;+\; \sum R_{\mathrm{blend}}$",
            transform=ax.transAxes, color="white", fontsize=26,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="black",
                      edgecolor="0.6", alpha=0.55))

    handles = [
        Line2D([0], [0], color=CYAN, lw=2,
               label=r"applied shear $\gamma_i$ (random, all galaxies)"),
        Line2D([0], [0], color=WHITE, lw=1.4, linestyle=(0, (4, 3)),
               label="galaxy"),
        Line2D([0], [0], color=GOLD, lw=2,
               label=r"target and its neighbours within $\theta_{\max}$"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.45,
              fontsize=13, labelcolor="white" if False else "black")

    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1]
    seed = int(sys.argv[2])
    field = render(seed)
    picked = pick_two(field)
    draw(field, picked, out)
    print(f"seed={seed} picked={picked} -> {out}")
