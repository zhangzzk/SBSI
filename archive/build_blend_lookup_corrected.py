"""Rebuild R_blend for the constant cases, applying the emulator's own validation-derived additive
correction Delta(r_input_s, distance) per pair before summing. Outputs R_blend (corrected) and
R_blend_raw. Debiases R_blend using ONLY the emulator's held-out validation (no constgold truth)."""
import argparse, os, sys, numpy as np, pandas as pd, pyarrow.feather as pf
CBASE="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
COND=dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312); TILE="tile180.0_-0.5"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases",type=int,nargs="+",required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--corr",default="/home/z/Zekang.Zhang/SBSI/results/emu_correction_rs_dist.npz")
    ap.add_argument("--tag",default="lsst_r_extnbr_ho"); a=ap.parse_args()
    sys.path.insert(0,"/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred=BlendingPredictor.load("/home/z/Zekang.Zhang/blendemu/models",tag=a.tag,conditions=COND,device="cpu")
    C=np.load(a.corr); D=C["Delta"]; rse=C["rs_edges"]; de=C["d_edges"]; nd=len(de)-1
    parts=[]
    for c in a.cases:
        fp=f"{CBASE}/case{c}_0.02/real0/catalogues/input/gals_info_{TILE}.feather"
        if not os.path.exists(fp): print(f"case{c} MISSING",flush=True); continue
        t=pf.read_table(fp).to_pandas(); t=t.rename(columns={col:col.replace("_input","") for col in t.columns})
        reg=pred.predict_response(t,t)
        rs=reg["r_input_s"].to_numpy(float); dd=reg["distance"].to_numpy(float); resp=reg["response"].to_numpy(float)
        ri=np.clip(np.digitize(rs,rse)-1,0,len(rse)-2); di=np.clip(np.digitize(dd,de)-1,0,nd-1)
        delta=D[ri,di]; corr=resp+delta
        pk="index_input_p"
        g=pd.DataFrame({"k":reg[pk].to_numpy(),"raw":resp,"cor":corr}).groupby("k").sum()
        parts.append(pd.DataFrame({"case":c,"input_index":g.index.to_numpy().astype(np.int64),
                                   "R_blend":g["cor"].to_numpy(),"R_blend_raw":g["raw"].to_numpy()}))
        print(f"case{c}: <R_blend_raw>={g['raw'].mean():.4f} <R_blend_corr>={g['cor'].mean():.4f}",flush=True)
    out=pd.concat(parts,ignore_index=True); out.to_feather(a.output)
    print(f"wrote {a.output}: {len(out):,} rows; global raw={out['R_blend_raw'].mean():.4f} "
          f"corr={out['R_blend'].mean():.4f}",flush=True); print("BLEND_CORR_DONE",flush=True)
if __name__=="__main__": main()
