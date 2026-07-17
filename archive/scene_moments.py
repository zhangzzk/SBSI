"""Extract flux-weighted SCENE second-moment traces (incl. neighbour separations) from the input
fields, for a DERIVED coherent-response model R = R0 * T_scene/(T_scene+T_psf). Caches per-object
moments so the derived fit iterates cheaply. Also computes scene quadrupole (for an optional
responsivity 2(1-e^2) correction)."""
import numpy as np, pandas as pd, pyarrow.feather as pf
from scipy.spatial import cKDTree
CBASE="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"; ZP=30.0; TILE="tile180.0_-0.5"
SCALES=[0.4,0.6,0.9,1.3]; RMAX=5.0
def flux(m): return np.power(10.0,-0.4*(m-ZP))
def per_obj(cases):
    t=pf.read_table(CBASE+"/constant_response_catalogue_train.feather",columns=["case","input_index","response","sum_et"]).to_pandas()
    t=t[t["case"].isin(cases)]
    return t.groupby(["case","input_index"],sort=False).agg(response=("response","mean"),sum_et=("sum_et","mean")).reset_index()
def field_moments(case,tgt):
    ip=f"{CBASE}/case{case}_0.02/real0/catalogues/input/gals_info_{TILE}.feather"
    inp=pf.read_table(ip,columns=["index_input","RA_input","DEC_input","r_input","Re_input"]).to_pandas()
    ra=inp["RA_input"].to_numpy(float); dec=inp["DEC_input"].to_numpy(float)
    ra0=ra.mean(); dec0=dec.mean()
    x=(ra-ra0)*np.cos(np.deg2rad(dec0))*3600.0; y=(dec-dec0)*3600.0
    f=flux(inp["r_input"].to_numpy(float)); Re=inp["Re_input"].to_numpy(float); idx=inp["index_input"].to_numpy()
    tree=cKDTree(np.column_stack([x,y]))
    row={int(i):r for r,i in enumerate(idx)}
    tr=np.array([row.get(int(i),-1) for i in tgt]); ok=tr>=0; tro=tr[ok]
    tx=x[tro]; ty=y[tro]
    nb=tree.query_ball_point(np.column_stack([tx,ty]),r=RMAX,workers=-1)
    nT=len(tgt); cols={"own_r":np.full(nT,np.nan),"own_Re":np.full(nT,np.nan),"_m":ok}
    cols["own_tr"]=np.zeros(nT)
    for s in SCALES:
        cols[f"A_{s}"]=np.zeros(nT); cols[f"nsz_{s}"]=np.zeros(nT); cols[f"nsp_{s}"]=np.zeros(nT)
        cols[f"Q1_{s}"]=np.zeros(nT); cols[f"Q2_{s}"]=np.zeros(nT)
    oidx=np.where(ok)[0]
    for jj,ti in enumerate(oidx):
        r0=tro[jj]; ft=f[r0]; Ret=Re[r0]; cols["own_r"][ti]=inp["r_input"].to_numpy()[r0]; cols["own_Re"][ti]=Ret
        nn=np.array([n for n in nb[jj] if n!=r0])
        cols["own_tr"][ti]=ft*2*Ret**2
        if len(nn)==0:
            for s in SCALES: cols[f"A_{s}"][ti]=ft
            continue
        dx=x[nn]-tx[jj]; dy=y[nn]-ty[jj]; d2=dx*dx+dy*dy; fn=f[nn]; Ren=Re[nn]
        for s in SCALES:
            w=np.exp(-d2/(2*s*s))
            cols[f"A_{s}"][ti]=ft+ (w*fn).sum()
            cols[f"nsz_{s}"][ti]=(w*fn*2*Ren**2).sum()
            cols[f"nsp_{s}"][ti]=(w*fn*d2).sum()
            cols[f"Q1_{s}"][ti]=(w*fn*(dx*dx-dy*dy)).sum()
            cols[f"Q2_{s}"][ti]=(w*fn*(2*dx*dy)).sum()
    out=pd.DataFrame(cols); out["input_index"]=np.asarray(tgt); out["case"]=case; return out
def main():
    cases=list(range(40)); ro=per_obj(set(cases)); parts=[]
    for c in cases:
        tg=ro[ro["case"]==c]["input_index"].to_numpy(); parts.append(field_moments(c,tg))
        print(f"case {c} done",flush=True)
    mom=pd.concat(parts,ignore_index=True)
    df=ro.merge(mom,on=["case","input_index"],how="inner")
    df=df[df["_m"]].reset_index(drop=True)
    rb=pf.read_table("results/blend_lookup_extnbrho_c0-39.feather").to_pandas()
    kc="R_blend" if "R_blend" in rb.columns else [x for x in rb.columns if x not in ("case","input_index")][0]
    df=df.merge(rb[["case","input_index",kc]].rename(columns={kc:"r_blend"}),on=["case","input_index"],how="left")
    df["r_blend"]=df["r_blend"].fillna(0.0)
    df.to_feather("results/scene_moments.feather")
    print(f"cached {len(df):,} rows -> results/scene_moments.feather",flush=True); print("SCENE_MOMENTS_DONE",flush=True)
if __name__=="__main__": main()
