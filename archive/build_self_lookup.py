"""Per-(case,input_index) emulator SELF-response lookup for the constant cases (mirror of
build_blend_lookup.py but calls predict_self_response). Gives the emulator's independent
self-response R_self so we can test a clean 2-input combiner g(R_self, R_blend) vs linear."""
import argparse, os, sys
import numpy as np, pandas as pd, pyarrow.feather as pf
CBASE="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
BLEND_MODELS="/home/z/Zekang.Zhang/blendemu/models"
COND=dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE="tile180.0_-0.5"
def input_feather(case, sign="0.02", base=CBASE):
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases",type=int,nargs="+",required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--sign",default="0.02")
    ap.add_argument("--tag",default="lsst_r_extdom"); a=ap.parse_args()
    sys.path.insert(0,"/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred=BlendingPredictor.load(BLEND_MODELS, tag=a.tag, conditions=COND, device="cpu", load_self=True)
    parts=[]
    for c in a.cases:
        fp=input_feather(c,a.sign)
        if not os.path.exists(fp): print(f"case{c}: MISSING",flush=True); continue
        t=pf.read_table(fp).to_pandas()
        t=t.rename(columns={col:col.replace("_input","") for col in t.columns})
        reg=pred.predict_self_response(t,t)
        pk=[col for col in reg.columns if col.startswith("index")][0]
        rs=reg.groupby(pk)["self_response"].mean()
        parts.append(pd.DataFrame({"case":c,"input_index":rs.index.to_numpy().astype(np.int64),
                                   "R_self":rs.to_numpy(float)}))
        print(f"case{c}: {len(rs):,} gals, R_self mean={rs.mean():.4f}",flush=True)
    out=pd.concat(parts,ignore_index=True); out.to_feather(a.output)
    print(f"wrote {a.output}: {len(out):,} rows",flush=True)
if __name__=="__main__": main()
