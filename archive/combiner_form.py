import numpy as np, pyarrow.feather as pf
import xgboost as xgb
df=pf.read_table("results/scene_field_features.feather").to_pandas()
y=df["response"].to_numpy(float); rb=df["r_blend"].to_numpy(float)
mag=df["own_r"].to_numpy(float); Re=df["own_Re"].to_numpy(float); case=df["case"].to_numpy()
ucase=np.sort(np.unique(case)); rng=np.random.default_rng(0); folds=np.array_split(rng.permutation(ucase),5)
def rep(tag,p):
    edges=[-1e9,0.02,0.05,0.10,0.25,1e9]; labs=["ISO","[.02,.05)","[.05,.10)","[.10,.25)",">=.25"]
    s=f"[{tag}] global {y.mean()/p.mean()-1:+.2%} | "
    for i in range(5):
        m=(rb>=edges[i])&(rb<edges[i+1])
        if m.sum()<2000: continue
        s+=f"{labs[i]}={y[m].mean()/p[m].mean()-1:+.2%} "
    # independent axis: by magnitude
    for lo,hi in [(18,24),(24,25),(25,26),(26,28.1)]:
        mm=(mag>=lo)&(mag<hi); s+=f"| r{lo}-{hi}={y[mm].mean()/p[mm].mean()-1:+.2%}"
    print(s,flush=True)

# (1) 2D lookup g(mag x r_blend): predict held-out object as the TRAIN-fold cell mean
mag_edges=np.quantile(mag,np.linspace(0,1,9)); rb_edges=np.array([-1,0,.02,.05,.1,.25,.5,1,10])
mi=np.clip(np.digitize(mag,mag_edges)-1,0,len(mag_edges)-2)
ri=np.clip(np.digitize(rb,rb_edges)-1,0,len(rb_edges)-2)
cell=mi*(len(rb_edges)-1)+ri; ncell=(len(mag_edges)-1)*(len(rb_edges)-1)
p_lut=np.full(len(df),np.nan)
for hold in folds:
    te=np.isin(case,hold); tr=~te
    tot=np.bincount(cell[tr],weights=y[tr],minlength=ncell); cnt=np.bincount(cell[tr],minlength=ncell)
    cm=np.where(cnt>0,tot/np.maximum(cnt,1),y[tr].mean())
    p_lut[te]=cm[cell[te]]
rep(f"2D-LUT g(mag8 x rblend8)={ncell}cells", p_lut)

# (2) shallow XGBoost depth-2 and depth-3 on [mag,Re,rblend]
def cv_xgb(feats_arr,depth,ntree):
    p=np.full(len(df),np.nan)
    base=dict(max_depth=depth,eta=0.1,subsample=0.8,objective="reg:squarederror",tree_method="hist")
    for hold in folds:
        te=np.isin(case,hold); tr=~te
        bst=xgb.train(base,xgb.DMatrix(feats_arr[tr],label=y[tr]),num_boost_round=ntree)
        p[te]=bst.predict(xgb.DMatrix(feats_arr[te]))
    return p
F=np.column_stack([mag,Re,rb])
rep("XGB depth2 x100 [mag,Re,rblend]", cv_xgb(F,2,100))
rep("XGB depth3 x150 [mag,Re,rblend]", cv_xgb(F,3,150))
print("COMBINER_FORM_DONE",flush=True)
