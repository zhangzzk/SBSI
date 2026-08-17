"""Compare tutorial p_detect (min over 10" pairs) with BlendEMU's own
predict_detection convention (nearest neighbour within 3", NaN-isolated)."""

import os
import sys
import warnings

import numpy as np
import pandas as pd

MASTER = "/home/z/Zekang.Zhang/SBSI-master"
sys.path.insert(0, MASTER)
sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
os.environ["SBSI_CACHE_DIR"] = os.path.join(MASTER, "models")
os.environ["BLENDEMU_MODELS"] = os.path.join(MASTER, "models/blendemu")

from sbsi import (  # noqa: E402
    EmulatorPairingConfig,
    get_model,
    load_catalogue,
    load_emulator,
    prepare_forward_catalogue,
)

OBS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}

models = get_model("V3")
emu = load_emulator(models, conditions=OBS, device="cpu")
cat = load_catalogue(os.path.join(MASTER, "examples/data/example_catalog.feather"))
prepared = prepare_forward_catalogue(
    cat, config=EmulatorPairingConfig.from_emulator(emu)
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    det = emu.predict_on_pairs(prepared.emulator_pairs, task="detection", rescaled=True)
    # BlendEMU's intended per-galaxy convention.
    cla = emu.predict_detection(cat.reset_index(drop=True))

p_notebook = det.groupby("primary_row")["detection_prob"].min()
# primary_row indexes the reset-index catalogue, matching cla row order.
p_blendemu = pd.Series(
    cla["detection_prob"].to_numpy(float), index=np.arange(len(cla))
).reindex(p_notebook.index)

print(f"galaxies compared: {len(p_notebook)} (of {len(cat)} in the input catalogue)")
for name, s in (
    ("tutorial cell (min over <=20 pairs, <=10\")", p_notebook),
    ("BlendEMU predict_detection (nearest, <=3\"/NaN)", p_blendemu),
):
    print(f"{name}:\n  mean={s.mean():.4f} median={s.median():.4f} "
          f"q25={s.quantile(.25):.4f} min={s.min():.6f} max={s.max():.6f}")

n_iso = int((cla["distance"] > 3).sum())
print(f"\nnearest neighbour beyond 3\" (isolated path, NaN features): "
      f"{n_iso} ({n_iso / len(cla):.2%})")

# Isolated-galaxy NaN-path probability vs the extrapolated far pairs:
iso_p = p_blendemu[cla.loc[p_blendemu.index.values, "distance"].to_numpy() > 3]
print(f"NaN-isolated detection_prob: mean={iso_p.mean():.4f} "
      f"min={iso_p.min():.4f} max={iso_p.max():.4f}")

# If we fix p but keep R_total per primary from the notebook, the weighted mean
# rescales as sum(R*p)/sum(p_notebook-side weights). We only have notebook means,
# so report the p-side shift and the implied weighted-total shift under a
# first-order approximation.
r_total_mean = 1.1193
w_notebook = 0.9414
print(f"\nnotebook: mean(R_total)={r_total_mean:.4f}, mean(R_total*P)={w_notebook:.4f} "
      f"=> implied mean(P | notebook weighting)={w_notebook / r_total_mean:.4f}")
print(f"first-order corrected mean(R_total*P) if P follows BlendEMU convention "
      f"and P is uncorrelated with R_total: {r_total_mean * p_blendemu.mean():.4f}")
