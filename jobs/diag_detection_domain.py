"""Diagnose the distance_scaled extrapolation warning in the tutorial notebook.

Rebuilds the tutorial pair table, compares it against the classifier's
training domain, and quantifies what the saturated classifier returns for
out-of-domain pairs and how the per-primary min reduction is affected.
"""

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

print("== per-task training settings recovered from the emulator ==")
for task in ("regression", "classification"):
    s = emu.select.get(task, {})
    print(f"{task}: cuts={s.get('cuts')} r_max={s.get('r_max')} k={s.get('k')}")
for task, bnds, feats in (
    ("regression", emu.boundaries_reg, emu.bst_reg.feature_names),
    ("classification", emu.boundaries_cla, emu.bst_cla.feature_names),
):
    if bnds is None:
        print(f"boundaries[{task}]: None")
        continue
    lo_hi = dict(zip(feats, np.asarray(bnds).T))
    print(f"boundaries[{task}] distance_scaled: {lo_hi.get('distance_scaled')}")

pairing = EmulatorPairingConfig.from_emulator(emu)
print("\n== pairing used by the tutorial (from_emulator default task) ==")
print(pairing)

cat = load_catalogue(os.path.join(MASTER, "examples/data/example_catalog.feather"))
prepared = prepare_forward_catalogue(cat, config=pairing)
pairs = prepared.emulator_pairs
d = pairs["distance"].to_numpy(float)
ds = pairs["distance_scaled"].to_numpy(float)
print(f"\n== pair table ==\npairs: {len(pairs)}  primaries: {len(prepared.flow_inputs)}")
print("raw distance [arcsec] percentiles 0/1/5/25/50/75/95/100:",
      np.percentile(d, [0, 1, 5, 25, 50, 75, 95, 100]).round(3))
print("fraction raw distance > 3\" :", round(float((d > 3).mean()), 4))
print("fraction raw distance > 10\":", round(float((d > 10).mean()), 4))
print("distance_scaled percentiles 0/50/75/90/100:",
      np.percentile(ds, [0, 50, 75, 90, 100]).round(3))
print("fraction distance_scaled > 5.59:", round(float((ds > 5.59).mean()), 4))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    det = emu.predict_on_pairs(pairs, task="detection", rescaled=True)
p = det["detection_prob"].to_numpy(float)
far = ds > 5.59

print("\n== detection_prob vs domain ==")
print("near (ds<=5.59) quantiles 0/1/5/25/50/75/100:",
      np.percentile(p[~far], [0, 1, 5, 25, 50, 75, 100]).round(4))
print("far  (ds> 5.59) quantiles 0/1/5/25/50/75/100:",
      np.percentile(p[far], [0, 1, 5, 25, 50, 75, 100]).round(4))

binned = pd.DataFrame({"ds": ds, "p": p})
bins = [0, 1, 2, 3, 4, 5, 5.59, 7, 9, 11, 14]
binned = binned.groupby(pd.cut(binned.ds, bins), observed=True)["p"]
print("\nbinned detection_prob (median / min / count):")
for interval, stats in binned.agg(["median", "min", "count"]).iterrows():
    print(f"  {str(interval):>14}: median={stats['median']:.4f} min={stats['min']:.4f} n={int(stats['count'])}")

print("\n== per-primary min reduction ==")
p_min_all = det.groupby("primary_row")["detection_prob"].min()
p_min_indom = (
    det.loc[~far].groupby("primary_row")["detection_prob"].min()
    .reindex(p_min_all.index).fillna(1.0)
)
p_min_3arcsec = (
    det.loc[d <= 3].groupby("primary_row")["detection_prob"].min()
    .reindex(p_min_all.index).fillna(1.0)
)
for name, s in (
    ("notebook: min over all pairs      ", p_min_all),
    ("min over in-domain pairs (ds<=5.59)", p_min_indom),
    ("min over raw distance <= 3\" pairs  ", p_min_3arcsec),
):
    print(f"{name}: mean={s.mean():.4f} median={s.median():.4f} "
          f"q25={s.quantile(.25):.4f} min={s.min():.6f}")

argmin_rows = det.loc[det.groupby("primary_row")["detection_prob"].idxmin()]
print("\n== which pair sets each primary's min ==")
print("argmin distance_scaled percentiles 5/25/50/75/95:",
      np.percentile(argmin_rows["distance_scaled"], [5, 25, 50, 75, 95]).round(3))
print("argmin fraction beyond classifier boundary:",
      round(float((argmin_rows["distance_scaled"] > 5.59).mean()), 4))
print("argmin fraction raw distance > 3\":",
      round(float((argmin_rows["distance"] > 3).mean()), 4))
changed = (p_min_all < p_min_indom - 1e-9).sum()
print(f"primaries whose min comes only from an out-of-domain pair: {changed} "
      f"({changed / len(p_min_all):.2%})")
