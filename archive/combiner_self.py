"""Faithful stage-2 test: combine the emulator's SELF-response R_self and blend-response R_blend to
predict the measured coherent R_full. Linear (incl. production 1:1 sum) vs nonlinear g(.), 5-fold by
case, per-r_blend-bin AND per-magnitude held-out m."""
import numpy as np, pyarrow.feather as pf
from sklearn.linear_model import LinearRegression
import xgboost as xgb
df=pf.read_table("results/scene_field_features.feather").to_pandas()[["case","input_index","response","r_blend","own_r","own_Re","sum_et"]]
sl=pf.read_table("results/self_lookup_const_c0-39.feather").to_pandas()
df=df.merge(sl,on=["case","input_index"],how="left")
df["R_self"]=df["R_self"].fillna(df["R_self"].median())
y=df["response"].to_numpy(float); rb=df["r_blend"].to_numpy(float); rs=df["R_self"].to_numpy(float)
mag=df["own_r"].to_numpy(float); case=df["case"].to_numpy()
print(f"rows={len(df):,}  <R_full>={y.mean():.4f} <R_self>={rs.mean():.4f} <R_blend>={rb.mean():.4f}  "
      f"<R_self+R_blend>={ (rs+rb).mean():.4f}",flush=True)
ucase=np.sort(np.unique(case)); rng=np.random.default_rng(0); folds=np.array_split(rng.permutation(ucase),5)
def rep(tag,p):
    edges=[-1e9,0.02,0.05,0.10,0.25,1e9]; labs=["ISO","[.02,.05)","[.05,.10)","[.10,.25)",">=.25"]
    s=f"[{tag}] global {y.mean()/p.mean()-1:+.2%} | "
    for i in range(5):
        m=(rb>=edges[i])&(rb<edges[i+1])
        if m.sum()<2000: continue
        s+=f"{labs[i]}={y[m].mean()/p[m].mean()-1:+.2%} "
    for lo,hi in [(18,24),(24,25),(25,26),(26,28.1)]:
        m=(mag>=lo)&(mag<hi); s+=f"| r{lo}={y[m].mean()/p[m].mean()-1:+.2%}"
    print(s,flush=True)
# production-style pure sum (no fit)
rep("PROD R_self+R_blend (1:1 sum)", rs+rb)
def cv(fn,cols):
    X=df[cols].to_numpy(float); p=np.full(len(df),np.nan)
    for hold in folds:
        te=np.isin(case,hold); tr=~te; p[te]=fn(X[tr],y[tr],X[te])
    return p
lin=lambda Xtr,ytr,Xte: LinearRegression().fit(Xtr,ytr).predict(Xte)
def xg(depth):
    def f(Xtr,ytr,Xte):
        b=xgb.train(dict(max_depth=depth,eta=0.1,subsample=0.8,objective="reg:squarederror",tree_method="hist"),
                    xgb.DMatrix(Xtr,label=ytr),num_boost_round=200); return b.predict(xgb.DMatrix(Xte))
    return f
rep("LIN a*Rself+b*Rblend+c", cv(lin,["R_self","r_blend"]))
rep("g(Rself,Rblend) XGB d3", cv(xg(3),["R_self","r_blend"]))
rep("g(Rself,Rblend,mag) XGB d3", cv(xg(3),["R_self","r_blend","own_r"]))
rep("g(Rself,Rblend,mag,Re) XGB d4", cv(xg(4),["R_self","r_blend","own_r","own_Re"]))
# headline: winning 3-input model with per-case error + additive c
pbest=cv(xg(3),["R_self","r_blend","own_r"])
sumet=df["sum_et"].to_numpy(float)
mc=np.array([y[case==c].mean()/pbest[case==c].mean()-1 for c in ucase])
print(f"[HEADLINE g(Rself,Rblend,mag)] held-out m = {mc.mean():+.3%} +/- {mc.std()/np.sqrt(len(mc)):.3%} (per-case SE)  spread std={mc.std():.2%}",flush=True)
print(f"  additive c=<sum_et>/2 global = {sumet.mean()/2:+.5f}",flush=True)
edges=[-1e9,0.02,0.05,0.10,0.25,1e9]; labs=["ISO","[.02,.05)","[.05,.10)","[.10,.25)",">=.25"]
cl="  c by r_blend: "
for i in range(5):
    m=(df["r_blend"].to_numpy()>=edges[i])&(df["r_blend"].to_numpy()<edges[i+1])
    if m.sum()<2000: continue
    cl+=f"{labs[i]}={sumet[m].mean()/2:+.5f} "
print(cl,flush=True)
np.savez("results/combiner_self_heldout.npz",case=case,input_index=df["input_index"].to_numpy(),
         R_true=y,R_pred=pbest,R_self=rs,R_blend=rb,mag=mag)
print("COMBINER_SELF_DONE",flush=True)
