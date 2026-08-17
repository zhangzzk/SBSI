"""Build constgold lookup for the V2.2 scene-aware BlendEMU regression."""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
import pyarrow.feather as pf
from sbs_shear.paths import CONST_SIM_DIR
BE="/home/z/Zekang.Zhang/blendemu"; TAG="lsst_r_extnbr_v22_scene3"; TILE="tile180.0_-0.5"
COND=dict(pixel_size=0.2,zero_point=30.,psf_fwhm=0.73,moffat_beta=2.224,pixel_rms=0.312)
def input_feather(case,sign,base): return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--cases",type=int,nargs="+",required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--sign",default="0.02"); ap.add_argument("--base",default=CONST_SIM_DIR); a=ap.parse_args()
    if os.path.exists(a.output): raise SystemExit(f"REFUSING to overwrite {a.output}")
    sys.path.insert(0,BE)
    from blendemu.inference import BlendingPredictor
    from blendemu.scene_features import SCENE_FLUX_PAIR_COLUMNS, add_scene_flux_features
    pred=BlendingPredictor.load(os.path.join(BE,"models"),tag=TAG,conditions=COND,device="cpu")
    missing=set(SCENE_FLUX_PAIR_COLUMNS)-set(pred.bst_reg.feature_names)
    if missing: raise RuntimeError(f"model missing scene features: {sorted(missing)}")
    parts=[]
    for case in a.cases:
        path=input_feather(case,a.sign,a.base)
        if not os.path.exists(path): print(f"case{case}: MISSING",flush=True); continue
        t=pf.read_table(path).to_pandas().rename(columns=lambda x:x.replace("_input",""))
        add_scene_flux_features(t,zero_point=COND["zero_point"],copy=False)
        pairs=pred.predict_response(t,t); pk=[c for c in pairs if c.startswith("index")][0]
        rb=pairs.groupby(pk,sort=False)["response"].sum()
        parts.append(pd.DataFrame({"case":case,"input_index":rb.index.to_numpy(np.int64),"R_blend":rb.to_numpy(float)}))
        print(f"case{case}: {len(rb):,} primaries {len(pairs):,} pairs <R_blend>={rb.mean():.6f}",flush=True)
    if not parts: raise RuntimeError("no cases scored")
    out=pd.concat(parts,ignore_index=True); out.to_feather(a.output)
    print(f"wrote {a.output}: {len(out):,} rows over {out.case.nunique()} cases")
if __name__ == "__main__": main()
