"""Ablation on the cached scene feature table: isolate what makes the coherent forward model flat.
Loads results/scene_field_features.feather (no field re-extraction). Compares feature sets and prints
per-r_blend-bin held-out m + feature importance, plus a 2-fold (train20/test20) robustness check.
"""
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import xgboost as xgb

SBSI = "/home/z/Zekang.Zhang/SBSI"
SHELLS = [2.0, 4.0, 7.0, 10.0]
OWN = ["own_r", "own_Re", "own_n", "own_q"]
SCENE = (["own_flux", "n_nbr", "sum_f", "sum_f_d2", "sum_f_d1", "sum_f_Re2",
          "near_d", "near_f", "flux_ratio", "scene_trace"]
         + [f"n_{int(s)}" for s in SHELLS] + [f"sf_{int(s)}" for s in SHELLS])

df = pf.read_table(SBSI + "/results/scene_field_features.feather").to_pandas()
y = df["response"].to_numpy(float)
case = df["case"].to_numpy()
rb = df["r_blend"].to_numpy(float)
ucase = np.sort(np.unique(case))
base = dict(max_depth=7, eta=0.05, subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", tree_method="hist")


def cv(feats, folds):
    X = df[feats].to_numpy(float)
    pred = np.full(len(df), np.nan)
    imp = {}
    for hold in folds:
        te = np.isin(case, hold); tr = ~te
        bst = xgb.train(base, xgb.DMatrix(X[tr], label=y[tr], feature_names=feats), num_boost_round=400)
        pred[te] = bst.predict(xgb.DMatrix(X[te], feature_names=feats))
        for k, v in bst.get_score(importance_type="gain").items():
            imp[k] = imp.get(k, 0) + v
    return pred, imp


def report(tag, p):
    edges = [-1e9, 0.02, 0.05, 0.10, 0.25, 1e9]
    labs = ["ISO", "[.02,.05)", "[.05,.10)", "[.10,.25)", ">=.25"]
    line = f"[{tag}] global m={y.mean()/p.mean()-1:+.2%} | "
    for i in range(5):
        s = (rb >= edges[i]) & (rb < edges[i + 1])
        if s.sum() < 2000:
            continue
        line += f"{labs[i]}={y[s].mean()/p[s].mean()-1:+.2%} "
    print(line)


rng = np.random.default_rng(0)
folds5 = np.array_split(rng.permutation(ucase), 5)
folds2 = np.array_split(rng.permutation(ucase), 2)

print("=== 5-fold held-out per-r_blend-bin m ===")
for tag, feats in [("own", OWN), ("own+rblend(=decomp)", OWN + ["r_blend"]),
                   ("own+scene(no emu)", OWN + SCENE), ("own+scene+rblend(B)", OWN + SCENE + ["r_blend"])]:
    p, imp = cv(feats, folds5)
    report(tag, p)
    if "B)" in tag:
        tot = sum(imp.values())
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:10]
        print("  top-10 gain importance (B): " + ", ".join(f"{k}={v/tot:.1%}" for k, v in top))
        rb_share = imp.get("r_blend", 0) / tot
        scene_share = sum(imp.get(f, 0) for f in SCENE) / tot
        print(f"  importance share: r_blend={rb_share:.1%}  scene={scene_share:.1%}  own={sum(imp.get(f,0) for f in OWN)/tot:.1%}")

print("\n=== 2-fold (train20/test20) robustness, variant B ===")
p2, _ = cv(OWN + SCENE + ["r_blend"], folds2)
report("B 2-fold", p2)

print("\n=== INDEPENDENT-axis check: own+rblend (no scene) vs B (own+scene+rblend) ===")
own_r = df["own_r"].to_numpy(float); n_nbr = df["n_nbr"].to_numpy(float)
p_dec, _ = cv(OWN + ["r_blend"], folds5)
p_B, _ = cv(OWN + SCENE + ["r_blend"], folds5)


def by_axis(name, x, edges, p, tag):
    line = f"  [{tag}] by {name}: "
    for i in range(len(edges) - 1):
        s = (x >= edges[i]) & (x < edges[i + 1])
        if s.sum() < 5000:
            continue
        line += f"{edges[i]:g}-{edges[i+1]:g}={y[s].mean()/p[s].mean()-1:+.2%} "
    print(line)


for tag, p in [("own+rblend", p_dec), ("B", p_B)]:
    by_axis("r-mag", own_r, [18, 24, 25, 26, 28.1], p, tag)
    by_axis("n_nbr", n_nbr, [0, 10, 20, 40, 200], p, tag)
    mc = [y[case == c].mean() / p[case == c].mean() - 1 for c in ucase]
    print(f"  [{tag}] per-held-out-case m: mean={np.mean(mc):+.3%} std={np.std(mc):.3%}")
print("SCENE_ABLATION_DONE")
