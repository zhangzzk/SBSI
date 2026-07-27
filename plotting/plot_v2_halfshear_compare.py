#!/usr/bin/env python
"""Before/after: V2 flow self-response vs half-shear target, sw_th400 (distance axis) vs rblendaxis
(r_blend blend-flux axis, lam_r=450). Same objects/order from the two eval npz -> overlay R_flow_old,
R_flow_new against the shared half-shear target R_hs. fig2=all objects (mag/size/nbr_flux), fig5=7"-iso."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OLD="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_flowfig_th400.npz"
NEW="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_flowfig_rblendaxis.npz"
OUT="/home/z/Zekang.Zhang/SBSI/figures"
BLUE,GREY,ORANGE="#0072B2","#999999","#D55E00"
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":200,"font.size":11,"axes.grid":False,
    "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"lines.linewidth":1.8})

def binned(x,y,nb=11,logx=False,ql=1,qh=99,minn=200):
    g=np.isfinite(x)&np.isfinite(y); x,y=x[g],y[g]
    lo,hi=np.percentile(x,[ql,qh])
    xx=np.log10(np.clip(x,1e-12,None)) if logx else x
    lo_e=np.log10(max(lo,1e-12)) if logx else lo; hi_e=np.log10(max(hi,1e-12)) if logx else hi
    ed=np.linspace(lo_e,hi_e,nb+1); idx=np.digitize(xx,ed)-1
    cx,ym=[],[]
    for b in range(nb):
        s=idx==b
        if s.sum()<minn: continue
        cx.append(10**(0.5*(ed[b]+ed[b+1])) if logx else 0.5*(ed[b]+ed[b+1])); ym.append(np.mean(y[s]))
    return np.array(cx),np.array(ym)

def panel(ax,x,rhs,rold,rnew,xlabel,logx=False):
    cx,ys=binned(x,rhs,logx=logx); _,yo=binned(x,rold,logx=logx); _,yn=binned(x,rnew,logx=logx)
    ax.plot(cx,ys,"-o",color=BLUE,ms=5,lw=2.0,label=r"half-shear target $R_{\rm hs}$")
    ax.plot(cx,yo,"-s",color=GREY,ms=4,lw=1.6,label=r"old flow (distance axis)")
    ax.plot(cx,yn,"-D",color=ORANGE,ms=4.5,lw=1.9,label=r"new flow (r_blend axis, $\lambda_r$=450)")
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.axhline(0,color="0.8",lw=0.7,ls=":")

o=np.load(OLD); n=np.load(NEW)
assert o["R_hs"].shape==n["R_hs"].shape
rhs=o["R_hs"]; mag=o["mag"]; size=o["size"]; nbf=o["nbr_flux"]
g=(o["good"].astype(bool))&(n["good"].astype(bool))
iso=g&o["iso7"].astype(bool)
rold=o["R_flow"]; rnew=n["R_flow"]

# fig2 all
fig,ax=plt.subplots(1,3,figsize=(15,5),sharey=True)
panel(ax[0],mag[g],rhs[g],rold[g],rnew[g],r"primary true mag $r_{\rm input,p}$")
panel(ax[1],size[g],rhs[g],rold[g],rnew[g],r"primary true size $R_e$ [pix]")
k=g&np.isfinite(nbf)&(nbf>1e-3)
panel(ax[2],nbf[k],rhs[k],rold[k],rnew[k],r"neighbour flux (near shell)",logx=True)
ax[0].set_ylabel("self-response  R"); ax[0].legend(loc="lower left",fontsize=8.5)
fig.suptitle("Before/after: does R_flow track the half-shear self-response across BLENDING?  (all objects)",fontsize=12.5,y=1.02)
fig.tight_layout(); p2=f"{OUT}/figv2_fig2_compare.png"; fig.savefig(p2,dpi=200,bbox_inches="tight"); plt.close(fig)

# fig5 iso
fig,ax=plt.subplots(1,2,figsize=(12,5),sharey=True)
panel(ax[0],mag[iso],rhs[iso],rold[iso],rnew[iso],r"primary true mag $r_{\rm input,p}$")
panel(ax[1],size[iso],rhs[iso],rold[iso],rnew[iso],r"primary true size $R_e$ [pix]")
ax[0].set_ylabel("self-response  R"); ax[0].legend(loc="lower left",fontsize=8.5)
fig.suptitle("Before/after: clean self-response (7\"-isolated), old distance axis vs new r_blend axis",fontsize=12.5,y=1.02)
fig.tight_layout(); p5=f"{OUT}/figv2_fig5_compare.png"; fig.savefig(p5,dpi=200,bbox_inches="tight"); plt.close(fig)

print("saved",p2); print("saved",p5)
for lab,sel in [("ISO",iso),("ALL",g)]:
    a=np.mean(rhs[sel]); b=np.mean(rold[sel]); c=np.mean(rnew[sel])
    print(f"  {lab}: <R_hs>={a:+.4f}  old={b:+.4f}({(b/a-1)*100:+.1f}%)  new={c:+.4f}({(c/a-1)*100:+.1f}%)  N={int(sel.sum()):,}")
print("CMP_DONE")
