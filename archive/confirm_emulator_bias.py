"""R_blend regression emulator mean-bias on its OWN held-out test set, memory-light (reads only the
raw columns needed). Reproduces train_emulator's cuts/rescale/split/target. Predicted vs true per-pair
response delta_et1/shear: global + by target mag, neighbour mag, distance."""
import sys, numpy as np, pyarrow.feather as pf
sys.path.insert(0,"/home/z/Zekang.Zhang/blendemu")
from blendemu.config import load_config
from blendemu import data_utils
from blendemu.inference import BlendingPredictor
from scripts.train_emulator import _catalogue_shear_scale
from sklearn.model_selection import train_test_split
import xgboost as xgb
CFG="/home/z/Zekang.Zhang/blendemu/configs/fs2_lsst_r_extnbr_ho.yaml"
CAT="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
COND=dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
cfg=load_config(CFG); tr=cfg['training']; rc=tr['rescale']
raw=["Re_input_p","Re_input_s","r_input_p","r_input_s","sersic_n_input_p","sersic_n_input_s","distance","delta_et1"]
print("reading columns...",flush=True)
df=pf.read_table(CAT,columns=raw).to_pandas()
print(f"raw {len(df):,}",flush=True)
df=data_utils.source_select_reg(df,cuts=tr['regression_cuts']); print(f"after cuts {len(df):,}",flush=True)
df=data_utils.rescale(df,pixel_rms=rc['pixel_rms'],pixel_size=rc['pixel_size'],zero_mag=rc['zero_mag'],
                      psf_fwhm=rc['psf_fwhm'],moffat_beta=rc['moffat_beta'])
shear=_catalogue_shear_scale(cfg,'response'); print("shear scale:",shear,flush=True)
y=(df["delta_et1"]/shear).to_numpy(float)
feats=tr['features']
xtr,xte,ytr,yte=train_test_split(df[feats],y,test_size=tr['test_size'],random_state=tr['random_state'])
pred=BlendingPredictor.load("/home/z/Zekang.Zhang/blendemu/models",tag="lsst_r_extnbr_ho",conditions=COND,device="cpu")
p=pred.bst_reg.predict(xgb.DMatrix(xte))*pred.y_std+pred.y_mean   # destandardized predicted response
t=np.asarray(yte)                                                # true response (delta_et1/shear)
print(f"\nTEST rows={len(t):,}  <true>={t.mean():.5f}  <pred>={p.mean():.5f}  "
      f"MEAN-BIAS <pred>/<true>-1 = {p.mean()/t.mean()-1:+.2%}",flush=True)
# bins by raw props (align via reset index)
xte2=xte.reset_index(drop=True)
raww=df.reset_index(drop=True)
# recover raw mag/dist for test rows via the split index
idx=xte.index.to_numpy()
for col in ["r_input_p","r_input_s","distance"]:
    v=raww.loc[idx,col].to_numpy(float); qs=np.quantile(v,[0,.25,.5,.75,1.0])
    line=f"  by {col}: "
    for i in range(4):
        m=(v>=qs[i])&(v<=qs[i+1]) if i==3 else (v>=qs[i])&(v<qs[i+1])
        if m.sum()<1000: continue
        line+=f"[{qs[i]:.2f},{qs[i+1]:.2f}]={p[m].mean()/t[m].mean()-1:+.1%}(N{m.sum()//1000}k) "
    print(line,flush=True)
print("EMU_BIAS_DONE",flush=True)
