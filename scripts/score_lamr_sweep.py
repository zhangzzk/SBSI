#!/usr/bin/env python
"""Rank the lam_r sweep by how well R_flow tracks the half-shear target R_hs across mag/size/nbr_flux.
Reads halfshear_flowfig_lr{LR}.npz (same objects/order). Metric = RMS of per-fine-bin fractional
residual (R_flow/R_hs-1) over the union of mag+size+nbr_flux bins, plus the bright over-shoot and
faint under-prediction so the trade-off is explicit. Lower |resid| = better."""
import os, glob, numpy as np
DIR="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk"

def binres(x, rhs, rf, nb=10, logx=False, minn=300, ql=2, qh=98):
    g=np.isfinite(x)&np.isfinite(rhs)&np.isfinite(rf)&(np.abs(rhs)>1e-3)
    x,rhs,rf=x[g],rhs[g],rf[g]
    lo,hi=np.percentile(x,[ql,qh]); xx=np.log10(np.clip(x,1e-12,None)) if logx else x
    lo=np.log10(max(lo,1e-12)) if logx else lo; hi=np.log10(max(hi,1e-12)) if logx else hi
    ed=np.linspace(lo,hi,nb+1); idx=np.digitize(xx,ed)-1
    res=[]
    for b in range(nb):
        s=idx==b
        if s.sum()<minn: continue
        res.append(np.mean(rf[s])/np.mean(rhs[s])-1.0)
    return np.array(res)

def score(npz):
    z=np.load(npz)
    rhs,rf=z["R_hs"],z["R_flow"]; mag,size,nbf=z["mag"],z["size"],z["nbr_flux"]
    g=z["good"].astype(bool); iso=g&z["iso7"].astype(bool)
    out={}
    for lab,sel in [("ALL",g),("ISO",iso)]:
        rm=binres(mag[sel],rhs[sel],rf[sel]); rs=binres(size[sel],rhs[sel],rf[sel])
        rn=binres(nbf[sel],rhs[sel],rf[sel],logx=True) if lab=="ALL" else np.array([])
        allb=np.concatenate([rm,rs,rn])
        out[lab]=dict(rms=float(np.sqrt(np.mean(allb**2))*100),
                      maxabs=float(np.max(np.abs(allb))*100),
                      glob=float(np.mean(rf[sel])/np.mean(rhs[sel])-1)*100,
                      nbf_rms=float(np.sqrt(np.mean(rn**2))*100) if rn.size else np.nan)
    # explicit trade-off markers (ISO mag fine bins: bright=first, faint=last)
    rm_iso=binres(mag[iso],rhs[iso],rf[iso])
    out["bright_overshoot%"]=float(rm_iso[0]*100) if rm_iso.size else np.nan
    out["faint_underpred%"]=float(rm_iso[-1]*100) if rm_iso.size else np.nan
    return out

rows=[]
for npz in sorted(glob.glob(f"{DIR}/halfshear_flowfig_lr*.npz")):
    lr=os.path.basename(npz).replace("halfshear_flowfig_lr","").replace(".npz","")
    try: s=score(npz)
    except Exception as e: print(f"lr{lr}: FAIL {e}"); continue
    rows.append((lr,s))
# also include the original rblendaxis(=lr450 joint) and old th400 for reference if present
for tag,fn in [("450joint","halfshear_flowfig_rblendaxis.npz"),("th400old","halfshear_flowfig_th400.npz")]:
    p=f"{DIR}/{fn}"
    if os.path.exists(p):
        try: rows.append((tag,score(p)))
        except Exception: pass

print(f"\n{'model':>10} | {'ALL rms':>8} {'ALL max':>8} {'ALL glob':>8} {'nbf rms':>8} | {'ISO rms':>8} {'ISO glob':>8} | {'bright+':>8} {'faint':>8}")
print("-"*100)
def keyf(r):
    return r[1]["ALL"]["rms"]
for lr,s in sorted(rows,key=keyf):
    print(f"{'lr'+lr:>10} | {s['ALL']['rms']:8.2f} {s['ALL']['maxabs']:8.2f} {s['ALL']['glob']:+8.2f} {s['ALL']['nbf_rms']:8.2f} | "
          f"{s['ISO']['rms']:8.2f} {s['ISO']['glob']:+8.2f} | {s['bright_overshoot%']:+8.1f} {s['faint_underpred%']:+8.1f}")
print("\nlower ALL-rms = better overall tracking; nbf-rms = neighbour-flux (blend) tracking; bright+ = bright-iso over-shoot; faint = faint-iso residual")
print("SCORE_DONE")
