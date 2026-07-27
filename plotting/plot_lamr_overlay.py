#!/usr/bin/env python
"""Overlay the lam_r sweep vs the half-shear target: R_hs (blue) + old th400 (grey, distance axis)
+ lr150/lr250/lr450 (r_blend axis). Panels: mag / size / neighbour-flux (all objects) and mag/size
(7"-iso). Also prints a corrected score (nbf panel restricted to nbf>1e-3, matching the plot)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DIR="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk"
MODELS=[("th400 (distance)","halfshear_flowfig_th400.npz","#999999","-s"),
        ("lr150 (r_blend)","halfshear_flowfig_lr150.npz","#009E73","-D"),
        ("lr250 (r_blend)","halfshear_flowfig_lr250.npz","#E69F00","-D"),
        ("lr450 (r_blend)","halfshear_flowfig_lr450.npz","#D55E00","-D")]
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":200,"font.size":11,"axes.grid":False,
  "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"lines.linewidth":1.7})

def binxy(x,y,nb=10,logx=False,minn=300,ql=2,qh=98,xmin=None):
    g=np.isfinite(x)&np.isfinite(y)&(np.abs(y)>1e-6)
    if xmin is not None: g&=x>xmin
    x,y=x[g],y[g]
    lo,hi=np.percentile(x,[ql,qh]); xx=np.log10(np.clip(x,1e-12,None)) if logx else x
    lo=np.log10(max(lo,1e-12)) if logx else lo; hi=np.log10(max(hi,1e-12)) if logx else hi
    ed=np.linspace(lo,hi,nb+1); idx=np.digitize(xx,ed)-1
    cx,cy=[],[]
    for b in range(nb):
        s=idx==b
        if s.sum()<minn: continue
        cx.append(10**(0.5*(ed[b]+ed[b+1])) if logx else 0.5*(ed[b]+ed[b+1])); cy.append(np.mean(y[s]))
    return np.array(cx),np.array(cy)

# load (same objects/order across all)
Z={lab:np.load(f"{DIR}/{fn}") for lab,fn,_,_ in MODELS if os.path.exists(f"{DIR}/{fn}")}
ref=next(iter(Z.values())); rhs=ref["R_hs"]; mag=ref["mag"]; size=ref["size"]; nbf=ref["nbr_flux"]
g=ref["good"].astype(bool); iso=g&ref["iso7"].astype(bool)

def draw(axes, sel, axis_specs):
    for ax,(arr,xl,logx,xmin) in zip(axes,axis_specs):
        cx,cy=binxy(arr[sel],rhs[sel],logx=logx,xmin=xmin)
        ax.plot(cx,cy,"-o",color="#0072B2",ms=5,lw=2.2,label=r"target $R_{\rm hs}$",zorder=5)
        for lab,fn,col,mk in MODELS:
            if lab.split()[0] not in [k.split()[0] for k in Z]: continue
            z=Z[lab]; rf=z["R_flow"]
            cx,cy=binxy(arr[sel],rf[sel],logx=logx,xmin=xmin)
            ax.plot(cx,cy,mk,color=col,ms=4,lw=1.5,label=lab)
        if logx: ax.set_xscale("log")
        ax.set_xlabel(xl); ax.axhline(0,color="0.85",lw=0.7,ls=":")

# fig: all objects (mag/size/nbf)
fig,ax=plt.subplots(1,3,figsize=(15.5,5),sharey=True)
draw(ax,g,[(mag,r"true mag $r_{\rm input,p}$",False,None),(size,r"true size $R_e$ [pix]",False,None),
           (nbf,r"neighbour flux (near shell, >1e-3)",True,1e-3)])
ax[0].set_ylabel("self-response R"); ax[0].legend(loc="lower left",fontsize=8)
fig.suptitle("lam_r sweep vs half-shear target — all objects (neighbour-flux panel = the blend fix)",y=1.02,fontsize=12.5)
fig.tight_layout(); p1=f"/home/z/Zekang.Zhang/SBSI/figures/figv2_lamr_overlay_all.png"; fig.savefig(p1,dpi=200,bbox_inches="tight"); plt.close(fig)

# fig: isolated (mag/size)
fig,ax=plt.subplots(1,2,figsize=(12,5),sharey=True)
draw(ax,iso,[(mag,r"true mag $r_{\rm input,p}$",False,None),(size,r"true size $R_e$ [pix]",False,None)])
ax[0].set_ylabel("self-response R"); ax[0].legend(loc="lower left",fontsize=8)
fig.suptitle("lam_r sweep vs half-shear target — 7\"-isolated (bright over-shoot vs faint fix)",y=1.02,fontsize=12.5)
fig.tight_layout(); p2=f"/home/z/Zekang.Zhang/SBSI/figures/figv2_lamr_overlay_iso.png"; fig.savefig(p2,dpi=200,bbox_inches="tight"); plt.close(fig)
print("saved",p1); print("saved",p2)

# corrected score: nbf panel restricted to nbf>1e-3
def resid(arr,sel,logx=False,xmin=None):
    cx,cyt=binxy(arr[sel],rhs[sel],logx=logx,xmin=xmin)
    out={}
    for lab,fn,_,_ in MODELS:
        if lab not in Z: continue
        _,cyf=binxy(arr[sel],Z[lab]["R_flow"][sel],logx=logx,xmin=xmin)
        n=min(len(cyt),len(cyf)); out[lab]=(cyf[:n]/cyt[:n]-1)
    return out
print(f"\n{'model':>18} | {'mag rms%':>8} {'size rms%':>9} {'nbf rms%':>8} | {'bright+%':>8} {'faint%':>7}")
for lab,fn,_,_ in MODELS:
    if lab not in Z: continue
    rm=resid(mag,g)[lab]; rs=resid(size,g)[lab]; rn=resid(nbf,g,logx=True,xmin=1e-3)[lab]
    rmi=resid(mag,iso)[lab]
    print(f"{lab:>18} | {np.sqrt(np.mean(rm**2))*100:8.1f} {np.sqrt(np.mean(rs**2))*100:9.1f} {np.sqrt(np.mean(rn**2))*100:8.1f} | "
          f"{rmi[0]*100:+8.1f} {rmi[-1]*100:+7.1f}")
print("OVERLAY_DONE")
