"""Derived coherent-response model from scene moments:
  R = R0 * [T/(T+Tpsf)] * (1 - kappa*e_scene^2)
T = flux-weighted scene second-moment trace (incl neighbour separations) = B/A;
e_scene^2 from the weighted quadrupole; PSF resolution factor T/(T+Tpsf); responsivity (1-kappa e^2).
Two modes: PURE (physical constants fixed, no fit) and CALIB (fit R0,Tpsf,kappa per weight scale,
5-fold by case). Report per-r_blend-bin and per-magnitude held-out m."""
import numpy as np, pyarrow.feather as pf
from scipy.optimize import least_squares
df=pf.read_table("results/scene_moments.feather").to_pandas()
y=df["response"].to_numpy(float); rb=df["r_blend"].to_numpy(float); mag=df["own_r"].to_numpy(float); case=df["case"]
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
    T=df[f"B_{s}"].to_numpy(float)/np.clip(df[f"A_{s}"].to_numpy(float),1e-12,None)
    e2=(df[f"Q1_{s}"].to_numpy(float)**2+df[f"Q2_{s}"].to_numpy(float)**2)/np.clip(df[f"B_{s}"].to_numpy(float)**2,1e-12,None)
    def model(p,T,e2): return p[0]*(T/(T+p[1]))*(1-p[2]*e2)
    def resid(p,T,e2,y): return model(p,T,e2)-y
    # CALIB: fit R0,Tpsf,kappa CV by case
    pred=np.full(len(y),np.nan); ps=[]
    for hold in folds:
        te=np.isin(case,hold); tr=~te
        r=least_squares(resid,x0=[1.0,0.3,1.0],bounds=([0.3,0.01,0],[2,3,5]),args=(T[tr],e2[tr],y[tr]),max_nfev=200)
        pred[te]=model(r.x,T[te],e2[te]); ps.append(r.x)
    w=rep(f"DERIV s={s} CALIB(R0,Tpsf,kappa)",pred)
    print(f"    median coeffs [R0,Tpsf,kappa]={np.round(np.median(ps,axis=0),4)}",flush=True)
print("DERIVED_FIT_DONE",flush=True)
