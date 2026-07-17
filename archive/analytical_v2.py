"""Physical mixture model for the coherent response:
  R_full = Rself(mag,size)*(1-w(rb)) + Rcap*w(rb)
The coherent blend pulls each galaxy's response from its isolated value Rself toward a common
ceiling Rcap (bright neighbour light dominates the measured shape); w(rb) is the blend weight
(0 isolated -> saturates with crowding). Rself is a size-shifted magnitude sigmoid. 5-fold by case."""
import numpy as np, pyarrow.feather as pf
from scipy.optimize import least_squares
df=pf.read_table("results/scene_field_features.feather").to_pandas()[["case","input_index","response","r_blend","own_r","own_Re"]]
sl=pf.read_table("results/self_lookup_const_c0-39.feather").to_pandas()
df=df.merge(sl,on=["case","input_index"],how="left"); df["R_self"]=df["R_self"].fillna(df["R_self"].median())
y=df["response"].to_numpy(float); rb=df["r_blend"].to_numpy(float); mag=df["own_r"].to_numpy(float)
logRe=np.log(np.clip(df["own_Re"].to_numpy(float),0.05,None)); rse=df["R_self"].to_numpy(float); case=df["case"]
ucase=np.sort(np.unique(case)); rng=np.random.default_rng(0); folds=np.array_split(rng.permutation(ucase),5)
def rep(tag,p):
    edges=[-1e9,0.02,0.05,0.10,0.25,1e9]; labs=["ISO","[.02,.05)","[.05,.10)","[.10,.25)",">=.25"]; worst=0
    s=f"[{tag}] glob {y.mean()/p.mean()-1:+.2%} |rb "
    for i in range(5):
        m=(rb>=edges[i])&(rb<edges[i+1]); 
        if m.sum()<2000: continue
        v=y[m].mean()/p[m].mean()-1; worst=max(worst,abs(v)); s+=f"{labs[i]}={v:+.2%} "
    s+="|mag "
    for lo,hi in [(18,24),(24,25),(25,26),(26,28.1)]:
        m=(mag>=lo)&(mag<hi); v=y[m].mean()/p[m].mean()-1; worst=max(worst,abs(v)); s+=f"{lo}={v:+.2%} "
    print(s+f"| WORST={worst:+.2%}",flush=True); return worst

def Rself_f(p,mag,logRe):
    Rcap,m0,ksz,wd=p[0],p[1],p[2],p[3]
    return Rcap/(1+np.exp((mag-m0-ksz*logRe)/wd))
def model(p,mag,logRe,rb):
    Rcap=p[0]; wmax=p[4]; r0=p[5]
    Rs=Rself_f(p,mag,logRe); w=wmax*(1-np.exp(-rb/r0))
    return Rs*(1-w)+Rcap*w
def resid(p,mag,logRe,rb,y): return model(p,mag,logRe,rb)-y
x0=[1.0,24.7,1.0,0.5,0.65,0.08]; lb=[0.5,22,-5,0.1,0,1e-3]; ub=[1.6,27,5,3,1,2]
def cv():
    p=np.full(len(y),np.nan)
    for hold in folds:
        te=np.isin(case,hold); tr=~te
        r=least_squares(resid,x0=x0,bounds=(lb,ub),args=(mag[tr],logRe[tr],rb[tr],y[tr]),max_nfev=300)
        p[te]=model(r.x,mag[te],logRe[te],rb[te])
    return p
print("=== PHYSICAL MIXTURE  R_full = Rself(mag,size)*(1-w(rb)) + Rcap*w(rb) ===",flush=True)
rep("MIX 6p", cv())
rF=least_squares(resid,x0=x0,bounds=(lb,ub),args=(mag,logRe,rb,y),max_nfev=600)
print(f"  fitted [Rcap,m0,ksize,width,wmax,r0]={np.round(rF.x,4)}",flush=True)

# variant: let w depend on magnitude (wmax -> wmax + wm*(mag-25))
def model2(p,mag,logRe,rb):
    Rcap=p[0]; Rs=Rself_f(p,mag,logRe); wmax=p[4]+p[6]*(mag-25); r0=p[5]
    w=np.clip(wmax,0,1)*(1-np.exp(-rb/r0)); return Rs*(1-w)+Rcap*w
def resid2(p,*a): return model2(p,*a[:-1])-a[-1]
x0b=x0+[0.0]; lbb=lb+[-0.3]; ubb=ub+[0.3]
def cv2():
    p=np.full(len(y),np.nan)
    for hold in folds:
        te=np.isin(case,hold); tr=~te
        r=least_squares(resid2,x0=x0b,bounds=(lbb,ubb),args=(mag[tr],logRe[tr],rb[tr],y[tr]),max_nfev=400)
        p[te]=model2(r.x,mag[te],logRe[te],rb[te])
    return p
rep("MIX +mag-dep w (7p)", cv2())
print("ANALYTICAL_V2_DONE",flush=True)
