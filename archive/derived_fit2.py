"""Derived RATIO model with dilution/boost asymmetry:
  N = own_tr + a*nbr_sep + b*nbr_size            (target size + coherent separation boost)
  D = own_tr + nbr_size + nbr_sep + Tpsf*A        (total post-seeing trace; PSF per unit flux)
  R = R0 * N/D
own_tr=f_t*2Re_t^2; nbr_size=Sum w f 2Re_n^2 (dilution); nbr_sep=Sum w f d^2 (coherent); A=weighted flux.
Physical expectation: a~1 (separation responds coherently), b~0 (neighbour's own size only dilutes),
Tpsf~PSF trace, R0~1. Isolated -> R0*2Re^2/(2Re^2+Tpsf) (resolution). 5-fold by case."""
import numpy as np, pyarrow.feather as pf
from scipy.optimize import least_squares
df=pf.read_table("results/scene_moments.feather").to_pandas()
y=df["response"].to_numpy(float); rb=df["r_blend"].to_numpy(float); mag=df["own_r"].to_numpy(float)
ot=df["own_tr"].to_numpy(float); case=df["case"]
ucase=np.sort(np.unique(case)); rng=np.random.default_rng(0); folds=np.array_split(rng.permutation(ucase),5)
SCALES=[0.4,0.6,0.9,1.3]
def rep(tag,p):
    edges=[-1e9,0.02,0.05,0.10,0.25,1e9]; labs=["ISO","[.02,.05)","[.05,.10)","[.10,.25)",">=.25"]; worst=0
    s=f"[{tag}] glob {y.mean()/p.mean()-1:+.2%} |rb "
    for i in range(5):
        m=(rb>=edges[i])&(rb<edges[i+1])
        if m.sum()<2000: continue
        v=y[m].mean()/p[m].mean()-1; worst=max(worst,abs(v)); s+=f"{labs[i]}={v:+.2%} "
    s+="|mag "
    for lo,hi in [(18,24),(24,25),(25,26),(26,28.1)]:
        m=(mag>=lo)&(mag<hi); v=y[m].mean()/p[m].mean()-1; worst=max(worst,abs(v)); s+=f"{lo}={v:+.2%} "
    print(s+f"| WORST={worst:+.2%}",flush=True); return worst
for s in SCALES:
    A=df[f"A_{s}"].to_numpy(float); nsz=df[f"nsz_{s}"].to_numpy(float); nsp=df[f"nsp_{s}"].to_numpy(float)
    def model(p):
        R0,Tpsf,a,b=p
        N=ot + a*nsp + b*nsz
        D=ot + nsz + nsp + Tpsf*A
        return R0*N/np.clip(D,1e-12,None)
    def resid(p,idx): return model(p)[idx]-y[idx]
    pred=np.full(len(y),np.nan); ps=[]
    for hold in folds:
        te=np.isin(case,hold); tr=np.where(~te)[0]
        r=least_squares(lambda p: resid(p,tr),x0=[1.0,0.3,1.0,0.0],bounds=([0.3,0.01,-2,-2],[2,3,3,3]),max_nfev=300)
        pred[te]=model(r.x)[te]; ps.append(r.x)
    rep(f"RATIO s={s}",pred)
    print(f"    median [R0,Tpsf,a,b]={np.round(np.median(ps,axis=0),4)}",flush=True)
print("DERIVED_FIT2_DONE",flush=True)
