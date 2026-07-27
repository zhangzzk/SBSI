#!/usr/bin/env python
"""Decisive discriminator for the Stage-1 +0.49-vs-+0.60 gap (cont.125): remove the g=0 anchor.
The g0.02 and g0.05 half-shear legs shear each primary in the SAME direction, so
    R = [e(0.05) - e(0.02)].ghat / (0.05-0.02)
is a forward difference with NO g=0 reference (both legs sheared, like constgold's non-zero
anchoring). Compare faint R to: g0-anchored HS +0.49  and  constgold antithetic +/-0.02 +0.60.
  ~+0.60 -> the g=0 reference/detection-selection was the culprit.
  ~+0.49 -> not the anchor; it's seed-correlation or a genuine faint-end ngmix asymmetry.
"""
import time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc
CAT="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
MAG_EDGES=np.array([18.,24.,25.,26.])
def _read(path,cols,maxcase):
    parts=[]
    with ipc.open_file(path) as r:
        av=set(r.schema.names); use=[c for c in cols if c in av]
        for bi in range(r.num_record_batches):
            b=pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            if int(b["case"].min())>maxcase: break
            b=b[b["case"]<=maxcase]
            if len(b): parts.append(b)
    return pd.concat(parts,ignore_index=True)
def _iso(df):
    det=df["detected"].astype(bool).to_numpy() if "detected" in df.columns else np.ones(len(df),bool)
    m=det&(~df["neighbored"].astype(bool).to_numpy())&(df["Re_input_p"].to_numpy(float)>0.3)&(df["r_input_p"].to_numpy(float)<26.)
    return df[m].reset_index(drop=True)
def _perbin(resp,mag,case,nboot=300):
    rng=np.random.default_rng(0); uniq=np.unique(case); out={}
    mi=np.clip(np.digitize(mag,MAG_EDGES)-1,0,len(MAG_EDGES)-2)
    for k in range(len(MAG_EDGES)-1):
        sel=mi==k; tag=f"[{MAG_EDGES[k]:g},{MAG_EDGES[k+1]:g})"
        if sel.sum()<5: out[tag]=(np.nan,np.nan,int(sel.sum())); continue
        rs=resp[sel]; cs=case[sel]; mean=float(np.mean(rs)); bs=[]
        for _ in range(nboot):
            d=rng.choice(uniq,size=len(uniq),replace=True); mm=np.isin(cs,d)
            if mm.sum()>2: bs.append(float(np.mean(rs[mm])))
        out[tag]=(mean,float(np.nanstd(bs)),int(sel.sum()))
    return out
cols=["measured_ngmix_g1","measured_ngmix_g2","gamma1_input_p","gamma2_input_p",
      "neighbored","detected","Re_input_p","r_input_p","case","input_index"]
t0=time.time()
A=_iso(_read(CAT+"det_meas_ngmix_g0.02_test.feather",cols,19))
B=_iso(_read(CAT+"det_meas_ngmix_g0.05_val.feather",cols,19))
for d in (A,B):
    gp=np.hypot(d["gamma1_input_p"],d["gamma2_input_p"]); d.drop(d.index[gp<=1e-6],inplace=True)
A=A.drop_duplicates(["case","input_index"]); B=B.drop_duplicates(["case","input_index"])
keepA=["case","input_index","measured_ngmix_g1","measured_ngmix_g2"]
m=B.merge(A[keepA],on=["case","input_index"],suffixes=("_05","_02"))
gp=np.hypot(m["gamma1_input_p"],m["gamma2_input_p"]).to_numpy(float)
gh1=m["gamma1_input_p"].to_numpy(float)/gp; gh2=m["gamma2_input_p"].to_numpy(float)/gp
de1=m["measured_ngmix_g1_05"].to_numpy(float)-m["measured_ngmix_g1_02"].to_numpy(float)
de2=m["measured_ngmix_g2_05"].to_numpy(float)-m["measured_ngmix_g2_02"].to_numpy(float)
dg=0.05-0.02
resp=(de1*gh1+de2*gh2)/dg
mag=m["r_input_p"].to_numpy(float); case=m["case"].to_numpy(int)
good=np.isfinite(resp); resp,mag,case=resp[good],mag[good],case[good]
print(f"ANTITHETIC-LIKE 0.02->0.05 (no g=0 anchor)  matched N={len(m):,} finite={len(resp):,}  ({time.time()-t0:.1f}s)",flush=True)
pb=_perbin(resp,mag,case)
print("  R_self per mag bin  [mean +- case-boot]:")
for tag,(mv,e,n) in pb.items():
    print(f"    mag{tag:10s} R={mv:+.4f} +- {e:.4f}  (N={n:,})")
print("\n  FAINT compare:  0.02->0.05 = %.4f | HS g0-anchored ~+0.49 | constgold +/-0.02 = +0.602"%pb["[25,26)"][0])
print("ANTITHETIC_0205_DONE",flush=True)
