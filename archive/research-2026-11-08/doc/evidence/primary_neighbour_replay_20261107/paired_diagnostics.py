"""Fixed-pilot, paired endpoint contrasts and exact decomposition of fit-repeat changes."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fixed_pilot", H/"report.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
K = r.CENTER
ROLES = ["primary", "neighbours", "coherent"]
MODES = ["predicted_generated", "actual_generated", "actual_measured_selection"]

def seed_change(sim, w, cases, ids):
    # For each corner, exactly: difference of means = (dY - mean(mu)*dM)/mean(M).
    # This permits a symmetric split into shape changes and changes of usable cohort,
    # including the denominator effect even when the repeat counts differ.
    x = sim[:, :, :, K, :2]
    totals = np.einsum("n,nhsf->hsf", w, x)
    mu = totals[..., 1] / totals[..., 0]
    delta = x[:, 1] - x[:, 0]
    v = w[:, None] * (delta[..., 1] - mu.mean(0) * delta[..., 0]) / totals[..., 0].mean(0)
    np.testing.assert_allclose(v.sum(0), mu[1]-mu[0], rtol=0, atol=1e-13)
    flip = x[:, 1, :, 0] != x[:, 0, :, 0]
    both = np.all(x[..., 0] == 1, axis=1)
    def roles(y): return (y @ r.ROLES.T)
    by_id = roles(v)
    value = by_id.sum(0)
    compare = r.estimate(sim[:, 1:2], w) - r.estimate(sim[:, :1], w)
    np.testing.assert_allclose(value, compare[:, K, 0], rtol=0, atol=3e-13)
    switch = roles(np.where(flip, v, 0)).sum(0)
    shape = roles(np.where(both, v, 0)).sum(0)
    np.testing.assert_allclose(shape+switch,value,atol=3e-13,rtol=0)
    out = dict(repeat1_minus_repeat0=value.tolist(), corner_selected_counts=x[...,0].sum(0).astype(int).tolist(),
               corner_usable_flip_counts=flip.sum(0).tolist(), corner_both_selected_counts=both.sum(0).tolist(),
               corner_mean_shape_changes=v.sum(0).tolist(), retained_in_both=shape.tolist(), switched_usable=switch.tolist(),
               cases={str(c):by_id[cases==c].sum(0).tolist() for c in r.CASES}, concentration={})
    for j,role in enumerate(ROLES):
        a=by_id[:,j]; ix=np.argsort(-np.abs(a),kind="stable")
        out["concentration"][role]=dict(absolute_sum=float(abs(a).sum()),
            top_abs_share={str(n):float(abs(a[ix[:n]]).sum()/abs(a).sum()) for n in [1,5,10,25]},
            top5=[dict(case=int(cases[i]),input_index=int(ids[i]),signed_change=float(a[i])) for i in ix[:5]])
    return out

def main():
    assert os.environ.get("SLURM_JOB_ID")
    source=r.load(H/"report.json")
    frames=[]; simulations=[]; models={}
    for c, rec in zip(r.CASES,source["receipts"],strict=True):
        sf=r.OUT/f"sample_{c}.feather"; mf=r.OUT/f"model{c}.npz"
        df=pd.read_feather(sf); ids=df.index_input.to_numpy()
        assert rec["case"]==c and r.sha(sf)==rec["sample_sha256"]
        mr=r.load(mf.with_suffix(".json"))
        assert mr["sample_sha256"]==rec["sample_sha256"] and mr["data_sha256"]==r.sha(mf)
        coh=[r.load(r.OUT/f"coherent{c}_{leg}.json") for leg in [0,1]]
        pix=dict(r.load(r.OUT/f"pixel_gate{c}.json")["cache"])
        sim=np.zeros((len(ids),2,4,9,3))
        us=np.zeros((len(ids),2,4))
        for i,gid in enumerate(ids):
            originals=[q["results"][str(gid)] for q in coh]
            mixes=[r.load(r.OUT/f"mixed{c}_{k}"/f"{gid}.json") for k in [0,1]]
            for k,m in enumerate(mixes):
                assert (m["case"],m["input_index"],m["background_leg"],m["source_sha256"],m["sample_sha256"],m["pixel_sha256"])==(c,gid,k,rec["measured_sha256"],rec["sample_sha256"],pix[gid])
            for k,ob in enumerate([originals[0],mixes[1]["result"],mixes[0]["result"],originals[1]]):
                sim[i,:,k],us[i,:,k]=r.obs(ob)
        with np.load(mf) as z:
            np.testing.assert_array_equal(z["input_index"],ids)
            np.testing.assert_array_equal(z["weights"],df.weight)
            ix=[list(z["cuts"]).index(k) for k in r.CUT_NAMES]
            u=list(z["cuts"]).index("uncut")
            prob=z["probability"][[0,0,1,1]].T
            for i,arm in enumerate(z["arms"]):
                m=z["moments"][i].transpose(1,0,2,3)
                merged=np.concatenate([m[...,:1],m[...,1:3]+m[...,3:5]],axis=-1)
                v=merged[:,:,ix]
                for name,x in zip(MODES,[
                    np.repeat((v*prob[:,:,None,None])[:,None],2,axis=1),
                    v[:,None]*us[:,:,:,None,None],
                    merged[:,:,u][:,None,:,None]*sim[...,:1]],strict=True):
                    models.setdefault((str(arm),name),[]).append(x)
        df["case"]=c
        df["block"]=np.clip((df.x0*4/18181).astype(int),0,3)+4*np.clip((df.y0*4/18181).astype(int),0,3)
        frames.append(df);simulations.append(sim)
    df=pd.concat(frames);sim=np.concatenate(simulations)
    w,case,st,b,ids=[df[k].to_numpy() for k in ["weight","case","stratum","block","index_input"]]
    out=dict(report_sha256=r.sha(H/"report.json"),cuts=r.CUT_NAMES,roles=ROLES,components=["R11","R21"],
             corners=["++","+-","-+","--"],seed_diagnostic=seed_change(sim,w,case,ids),paired={})
    for a,barm in [("initial","control"),("control","weighted")]:
        dest=out["paired"][a+"_to_"+barm]={}
        for mode in MODES:
            x,y=[np.concatenate(models[arm,mode]) for arm in (a,barm)]
            s=r.summarize(x,y,w,case,st,b)
            d={k:s[k] for k in ["value","case_jackknife_se","stratified_sampling_se","optimizer_MC_se"]}
            expected=np.array(source["arms"][barm][mode]["value"])-source["arms"][a][mode]["value"]
            np.testing.assert_allclose(d["value"],expected,atol=4e-13,rtol=0)
            d["individual_scene"]={str(c):(r.estimate(x,w*(case==c))-r.estimate(y,w*(case==c))).tolist() for c in r.CASES}
            dest[mode]=d
    p=H/"paired_diagnostics.json"; temp=p.with_suffix(".tmp");temp.write_text(json.dumps(out,indent=1,allow_nan=False)+"\n");temp.replace(p)
    print("PASS: seed accounting; radius/mag and flags; six paired x nine cuts x both output components x three physical roles; endpoint replay",flush=True)

if __name__=="__main__": main()
