#!/usr/bin/env python
"""Standard fig2 (all objects: mag/size/nbr_flux) + fig5 (7"-iso: mag/size) for ONE model's half-shear
eval npz: target R_hs (blue) vs R_flow (orange), per-bin % residual annotated. Usage: <npz> <suffix> <title>."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NPZ, SUF, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
OUT="/home/z/Zekang.Zhang/SBSI/figures"; BLUE,ORANGE="#0072B2","#D55E00"
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":200,"font.size":11,"axes.grid":False,
  "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"lines.linewidth":1.8})

def binned(x,ys,ym,nb=11,logx=False,ql=1,qh=99,minn=200):
    g=np.isfinite(x)&np.isfinite(ys)&np.isfinite(ym); x,ys,ym=x[g],ys[g],ym[g]
    lo,hi=np.percentile(x,[ql,qh]); xx=np.log10(np.clip(x,1e-12,None)) if logx else x
    lo=np.log10(max(lo,1e-12)) if logx else lo; hi=np.log10(max(hi,1e-12)) if logx else hi
    ed=np.linspace(lo,hi,nb+1); idx=np.digitize(xx,ed)-1
    cx,sm,slo,shi,mm=[],[],[],[],[]
    for b in range(nb):
        s=idx==b
        if s.sum()<minn: continue
        cx.append(10**(0.5*(ed[b]+ed[b+1])) if logx else 0.5*(ed[b]+ed[b+1]))
        sm.append(np.mean(ys[s])); slo.append(np.percentile(ys[s],16)); shi.append(np.percentile(ys[s],84)); mm.append(np.mean(ym[s]))
    return map(np.asarray,(cx,sm,slo,shi,mm))

def panel(ax,x,ys,ym,xl,logx=False):
    cx,sm,slo,shi,mm=binned(x,ys,ym,logx=logx)
    ax.fill_between(cx,slo,shi,color=BLUE,alpha=0.15,lw=0,label="half-shear 16-84%")
    ax.plot(cx,sm,"-o",color=BLUE,ms=5,lw=1.9,label=r"target $R_{\rm hs}$")
    ax.plot(cx,mm,"-s",color=ORANGE,ms=5,lw=1.9,label=r"flow $R_{\rm flow}$")
    for xx,a,b in zip(cx,sm,mm):
        if np.isfinite(a) and np.isfinite(b) and abs(a)>1e-6:
            ax.annotate(f"{(b/a-1)*100:+.0f}%",(xx,b),textcoords="offset points",xytext=(0,-12),ha="center",fontsize=7,color="#8a5a00")
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xl); ax.axhline(0,color="0.8",lw=0.7,ls=":")

z=np.load(NPZ); Rhs,Rf=z["R_hs"],z["R_flow"]; mag,size,nbf=z["mag"],z["size"],z["nbr_flux"]
g=z["good"].astype(bool); iso=g&z["iso7"].astype(bool)

# fig5 (iso)
fig,(a,b)=plt.subplots(1,2,figsize=(12,5),sharey=True)
panel(a,mag[iso],Rhs[iso],Rf[iso],r"true mag $r_{\rm input,p}$"); panel(b,size[iso],Rhs[iso],Rf[iso],r"true size $R_e$ [pix]")
a.set_ylabel("self-response R"); a.legend(loc="lower left",fontsize=8.5)
a.text(0.98,0.03,f"7\"-iso N={int(iso.sum()):,}",transform=a.transAxes,fontsize=7.5,va="bottom",ha="right",color="0.4")
fig.suptitle(f"fig5 — {TITLE}: self-response vs half-shear target, 7\"-isolated",y=1.01,fontsize=12.5)
fig.tight_layout(); p5=f"{OUT}/figv2_fig5_{SUF}.png"; fig.savefig(p5,dpi=200,bbox_inches="tight"); plt.close(fig)

# fig2 (all)
fig,ax=plt.subplots(1,3,figsize=(15,5),sharey=True)
panel(ax[0],mag[g],Rhs[g],Rf[g],r"true mag $r_{\rm input,p}$"); panel(ax[1],size[g],Rhs[g],Rf[g],r"true size $R_e$ [pix]")
k=g&np.isfinite(nbf)&(nbf>1e-3); panel(ax[2],nbf[k],Rhs[k],Rf[k],r"neighbour flux (near shell)",logx=True)
ax[0].set_ylabel("self-response R"); ax[0].legend(loc="lower left",fontsize=8.5)
ax[0].text(0.98,0.03,f"all N={int(g.sum()):,}",transform=ax[0].transAxes,fontsize=7.5,va="bottom",ha="right",color="0.4")
fig.suptitle(f"fig2 — {TITLE}: R_flow vs half-shear self-response across mag/size/blend (all objects)",y=1.02,fontsize=12.5)
fig.tight_layout(); p2=f"{OUT}/figv2_fig2_{SUF}.png"; fig.savefig(p2,dpi=200,bbox_inches="tight"); plt.close(fig)
print("saved",p5); print("saved",p2)
