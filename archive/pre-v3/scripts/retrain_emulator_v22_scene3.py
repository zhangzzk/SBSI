"""Train V2.2 BlendEMU with three intrinsic full-scene flux features.

One lever changes from ``lsst_r_extnbr_v22``: the added scene features. The
half-shear labels, cases 40--199, cuts, random split, rescaling, and XGBoost
parameters are unchanged. Constgold is never read.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BE = "/home/z/Zekang.Zhang/blendemu"
for path in (ROOT, BE, os.path.join(BE, "scripts")):
    if path not in sys.path: sys.path.insert(0, path)
from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
from blendemu.scene_features import SCENE_FLUX_PAIR_COLUMNS  # noqa: E402
import train_emulator as TE  # noqa: E402
import retrain_extnbr as RE  # noqa: E402

BASE_TAG = "lsst_r_extnbr_v22"
TAG = "lsst_r_extnbr_v22_scene3"
EXPECTED_CUTS = np.asarray([[13,29],[18,25.8],[0,10],[0.5,1.5],[0,10]], float)

def summarize(name, booster, frame, target, mean, std):
    pred = data_utils.reverse_standardize(booster.predict(
        xgb.DMatrix(frame), iteration_range=data_utils.get_xgb_iteration_range(booster)), mean, std)
    label = data_utils.reverse_standardize(np.asarray(target), mean, std)
    resid = pred - label
    print(f"  {name}: N={len(label):,} label={label.mean():.6f} pred={pred.mean():.6f} "
          f"pred/label-1={100*(pred.mean()/label.mean()-1):+.3f}% "
          f"residual_std={resid.std(ddof=1):.6f} R2={r2_score(label,pred):.6f}", flush=True)

def main():
    cfg = load_config(os.environ["CONFIG_PATH"]); tr = cfg["training"]
    if tr["model_tag"] != TAG: raise SystemExit(f"wrong tag {tr['model_tag']!r}")
    if not np.array_equal(np.asarray(tr["regression_cuts"],float), EXPECTED_CUTS):
        raise SystemExit(f"V2.2 cuts drifted: {tr['regression_cuts']}")
    if tuple(tr["features"][-3:]) != SCENE_FLUX_PAIR_COLUMNS:
        raise SystemExit(f"wrong scene features: {tr['features'][-3:]}")
    if os.environ.get("HELDOUT_MIN_CASE", "0") != "40":
        raise SystemExit("HELDOUT_MIN_CASE must be 40")
    RE._NEED = list(dict.fromkeys([*RE._NEED, *SCENE_FLUX_PAIR_COLUMNS]))
    print(f"### V2.2 SCENE3 EMULATOR tag={TAG} ###\n  features={list(SCENE_FLUX_PAIR_COLUMNS)}")
    loaded = RE.load_regression_data_lowmem(cfg)
    DMtrain, DMtest, xtrain, xtest, ytrain, ytest, ymean, ystd = loaded
    task = json.load(open(os.path.join(BE,f"models/emulator_metadata_{BASE_TAG}.json")))["tasks"]["regression"]
    params = dict(task["params"])
    params.update(objective="reg:squarederror", n_jobs=-1, device=os.environ.get("XGB_DEVICE","cpu"),
                  tree_method="hist", booster="gbtree", disable_default_eval_metric=1)
    print(f"  parameters inherited from {BASE_TAG}: {params}")
    metric=lambda p,d:("r2",r2_score(d.get_label(),p)); history={}; start=time.time()
    bst=xgb.train(params,DMtrain,evals=[(DMtrain,"train"),(DMtest,"eval")],evals_result=history,
                  num_boost_round=2000,verbose_eval=100,custom_metric=metric,maximize=True,
                  callbacks=[TE._early_stopping_callback(cfg,"regression","r2",maximize=True)])
    trees=data_utils.get_xgb_iteration_range(bst)[1]
    print(f"  time={time.time()-start:.1f}s best_iter={bst.best_iteration} trees={trees}")
    summarize("train",bst,xtrain,ytrain,ymean,ystd); summarize("validation",bst,xtest,ytest,ymean,ystd)
    model=TE._fname(tr["model_dir"],"regression_model.json",cfg)
    curve=TE._fname(tr["model_dir"],"regression_train_curve.npz",cfg)
    boundary=np.asarray([[xtrain[f].min(),xtrain[f].max()] for f in tr["features"]])
    bst.save_model(model); np.savez(curve,train_r2=history["train"]["r2"],eval_r2=history["eval"]["r2"])
    meta=TE._update_metadata(cfg,"regression",model,tr["features"],boundary,params,
        standardization=(ymean,ystd),train_curve_path=curve,metrics={
        "best_iteration":bst.best_iteration,"best_trees":trees,"best_score":bst.best_score,
        "score_name":"r2","params_inherited_from":BASE_TAG,"train_rows":len(xtrain),
        "validation_rows":len(xtest),"scene_features":list(SCENE_FLUX_PAIR_COLUMNS),
        "scene_feature_source":"intrinsic full rendered input scene; no measurements",
        "sbsi_domain":{"domain":"v2.2","primary_mag_max":25.8,"primary_re_min_arcsec":0.5}})
    print(f"saved model={model}\nmetadata={meta}\nV22_SCENE3_EMU_TRAIN_DONE",flush=True)
if __name__ == "__main__": main()
