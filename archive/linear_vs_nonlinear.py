import numpy as np, pyarrow.feather as pf
from sklearn.linear_model import LinearRegression
df=pf.read_table("results/scene_field_features.feather").to_pandas()
y=df["response"].to_numpy(float); rb=df["r_blend"].to_numpy(float); case=df["case"].to_numpy()
SHELLS=[2.,4.,7.,10.]
OWN=["own_r","own_Re","own_n","own_q"]
SCENE=(["own_flux","n_nbr","sum_f","sum_f_d2","sum_f_d1","sum_f_Re2","near_d","near_f","flux_ratio","scene_trace"]
       +[f"n_{int(s)}" for s in SHELLS]+[f"sf_{int(s)}" for s in SHELLS])
df["rb2"]=rb**2; df["rb_flux"]=rb*df["own_flux"].to_numpy(); df["rb_Re"]=rb*df["own_Re"].to_numpy()
ucase=np.sort(np.unique(case))
rng=np.random.default_rng(0); folds=np.array_split(rng.permutation(ucase),5)
def cv_lin(feats):
    X=df[feats].to_numpy(float); p=np.full(len(df),np.nan)
    for hold in folds:
        te=np.isin(case,hold); tr=~te
        p[te]=LinearRegression().fit(X[tr],y[tr]).predict(X[te])
    return p
def rep(tag,p):
    edges=[-1e9,0.02,0.05,0.10,0.25,1e9]; labs=["ISO","[.02,.05)","[.05,.10)","[.10,.25)",">=.25"]
    s=f"[{tag}] global {y.mean()/p.mean()-1:+.2%} | "
    for i in range(5):
        m=(rb>=edges[i])&(rb<edges[i+1])
        if m.sum()<2000: continue
        s+=f"{labs[i]}={y[m].mean()/p[m].mean()-1:+.2%} "
    print(s,flush=True)
print("=== LINEAR (OLS) 5-fold-by-case held-out per-r_blend-bin m ===",flush=True)
rep("LIN own+rblend", cv_lin(OWN+["r_blend"]))
rep("LIN own+scene+rblend", cv_lin(OWN+SCENE+["r_blend"]))
rep("LIN own+rblend+rb^2+rb*own(super-add terms)", cv_lin(OWN+["r_blend","rb2","rb_flux","rb_Re"]))
print("LINNL_DONE",flush=True)
