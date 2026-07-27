#!/usr/bin/env python
"""Does the constgold-vs-det_meas self-response gap track TRUE neighbour proximity (7" scale)?
Owner correction 2026-07-23: the pre-baked `neighbored` flag is the 3" DETECTION convention; for shear
response the scale is 7" (R_blend emulator range). The two sims share the IDENTICAL input field, so one
truth nn_dist lookup joins both. Prediction if the gap is neighbour-blend (handled by R_blend):
  det_meas (primary-only, neighbours cancel in the forward diff) -> FLAT vs nn_dist_bright,
  constgold (whole field sheared, neighbour blend survives)      -> RISES as bright neighbours approach,
  gap = constgold - det_meas -> ~0 for objects with NO bright neighbour within 7".
Self-response defs identical to compare_selfresp_sims.py. FIREWALL: constgold read-only.
"""
import time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
CAT="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
CDIR="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
NN="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/nn_dist_c0-39.feather"
NNEDGES=np.array([0.,3.,5.,7.,10.,np.inf])

def _read(path,cols,maxcase):
    parts=[]
    with ipc.open_file(path) as r:
        av=set(r.schema.names); use=[c for c in cols if c in av]
        for bi in range(r.num_record_batches):
            b=pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            if maxcase is not None:
                if int(b["case"].min())>maxcase: break
                b=b[b["case"]<=maxcase]
            if len(b): parts.append(b)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)

def _truecut(df):  # detected + true cut, NO neighbour cut (we bin by nn instead)
    det=df["detected"].astype(bool).to_numpy() if "detected" in df.columns else np.ones(len(df),bool)
    m=det&(df["Re_input_p"].to_numpy(float)>0.3)&(df["r_input_p"].to_numpy(float)<26.0)
    return df[m].reset_index(drop=True)

def _boot(resp,case,nboot=300,seed=0):
    if len(resp)<5: return (np.nan,np.nan,len(resp))
    rng=np.random.default_rng(seed); uniq=np.unique(case); bs=[]
    for _ in range(nboot):
        d=rng.choice(uniq,size=len(uniq),replace=True); mm=np.isin(case,d)
        if mm.sum()>2: bs.append(float(np.mean(resp[mm])))
    return (float(np.mean(resp)),float(np.nanstd(bs)),len(resp))

def constgold(nn,maxcase=39):
    cols=["measured_e1_plus","measured_e2_plus","measured_e1_minus","measured_e2_minus",
          "applied_g1","applied_g2","detected","Re_input_p","r_input_p","case","input_index"]
    R=_truecut(_read(CDIR+"/constant_response_catalogue_train.feather",cols,maxcase))
    g=np.hypot(R["applied_g1"],R["applied_g2"]).to_numpy(float); gmed=float(np.median(g))
    gh1=R["applied_g1"].to_numpy(float)/g; gh2=R["applied_g2"].to_numpy(float)/g
    resp=0.5*((R["measured_e1_plus"]-R["measured_e1_minus"])*gh1+(R["measured_e2_plus"]-R["measured_e2_minus"])*gh2)/gmed
    R=R.assign(resp=resp.to_numpy(float),mag=R["r_input_p"].to_numpy(float))
    R=R.merge(nn,on=["case","input_index"],how="left")
    return R[np.isfinite(R["resp"])].reset_index(drop=True)

def detmeas(nn,maxcase=19):
    cols=["measured_ngmix_g1","measured_ngmix_g2","gamma1_input_p","gamma2_input_p",
          "detected","Re_input_p","r_input_p","case","input_index"]
    G0=_truecut(_read(CAT+"det_meas_ngmix_g0.0_train.feather",cols,maxcase))
    G=_truecut(_read(CAT+"det_meas_ngmix_g0.02_test.feather",cols,maxcase))
    gp=np.hypot(G["gamma1_input_p"],G["gamma2_input_p"]).to_numpy(float); G=G[gp>1e-6].reset_index(drop=True)
    G0=G0.drop_duplicates(["case","input_index"]); G=G.drop_duplicates(["case","input_index"])
    m=G.merge(G0[["case","input_index","measured_ngmix_g1","measured_ngmix_g2"]],on=["case","input_index"],suffixes=("_g","_0"))
    gp=np.hypot(m["gamma1_input_p"],m["gamma2_input_p"]).to_numpy(float); gmed=float(np.median(gp))
    gh1=m["gamma1_input_p"].to_numpy(float)/gp; gh2=m["gamma2_input_p"].to_numpy(float)/gp
    de1=(m["measured_ngmix_g1_g"]-m["measured_ngmix_g1_0"]).to_numpy(float)
    de2=(m["measured_ngmix_g2_g"]-m["measured_ngmix_g2_0"]).to_numpy(float)
    resp=(de1*gh1+de2*gh2)/gmed
    m=m.assign(resp=resp,mag=m["r_input_p"].to_numpy(float))
    m=m.merge(nn,on=["case","input_index"],how="left")
    return m[np.isfinite(m["resp"])].reset_index(drop=True)

def bin_report(df,label,maglo,maghi,col="nn_dist_bright"):
    d=df[(df["mag"]>=maglo)&(df["mag"]<maghi)]
    print(f"\n  [{label}] mag[{maglo},{maghi})  N={len(d):,}  (bin by {col})")
    out={}
    dv=d[col].to_numpy(float); dv=np.where(np.isfinite(dv),dv,1e9)
    for k in range(len(NNEDGES)-1):
        sel=(dv>=NNEDGES[k])&(dv<NNEDGES[k+1])
        tag=f"{NNEDGES[k]:g}-{NNEDGES[k+1]:g}\""
        mv,e,n=_boot(d["resp"].to_numpy(float)[sel],d["case"].to_numpy(int)[sel])
        out[tag]=(mv,e,n)
        print(f"    nn_bright {tag:9s}: R_self={mv:+.4f} +- {e:.4f}  (N={n:,})")
    return out

def main():
    t0=time.time()
    nn=pf.read_table(NN).to_pandas()[["case","input_index","nn_dist_any","nn_dist_bright"]]
    print(f"nn lookup: {len(nn):,} rows, {nn['case'].nunique()} cases  ({time.time()-t0:.1f}s)",flush=True)
    cg=constgold(nn); print(f"constgold: {len(cg):,} true-cut detected  ({time.time()-t0:.1f}s)",flush=True)
    dm=detmeas(nn);   print(f"det_meas : {len(dm):,} matched true-cut  ({time.time()-t0:.1f}s)",flush=True)
    for lab,lo,hi in [("FAINT",25.,26.),("MID",24.,25.),("BRIGHT",18.,24.)]:
        print("\n"+"="*78+f"\n{lab} self-response vs nearest-BRIGHTER-neighbour distance\n"+"="*78)
        ocg=bin_report(cg,"constgold",lo,hi)
        odm=bin_report(dm,"det_meas ",lo,hi)
        print(f"\n  GAP (constgold - det_meas), {lab}:")
        for k in range(len(NNEDGES)-1):
            tag=f"{NNEDGES[k]:g}-{NNEDGES[k+1]:g}\""
            g=ocg[tag][0]-odm[tag][0]
            print(f"    nn_bright {tag:9s}: gap={g:+.4f}   (cg N={ocg[tag][2]//1000}k, dm N={odm[tag][2]//1000}k)")
    # strict 7" isolation head-to-head (no bright neighbour within 7")
    print("\n"+"="*78+"\nSTRICT 7\" ISOLATION (nn_dist_bright > 7): FAINT head-to-head\n"+"="*78)
    for lab,df in [("constgold",cg),("det_meas",dm)]:
        d=df[(df["mag"]>=25)&(df["mag"]<26)]
        sel=(~np.isfinite(d["nn_dist_bright"].to_numpy(float)))|(d["nn_dist_bright"].to_numpy(float)>7)
        mv,e,n=_boot(d["resp"].to_numpy(float)[sel],d["case"].to_numpy(int)[sel])
        print(f"  {lab:10s} R_self = {mv:+.4f} +- {e:.4f}  (N={n:,})")
    print("SELFRESP_ISO_DONE",flush=True)

if __name__=="__main__":
    main()
