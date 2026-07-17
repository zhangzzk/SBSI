"""Build the R_blend emulator's ADDITIVE bias-correction table Delta(r_s, distance) = <true - pred>
from its OWN held-out validation (no constgold truth). Applied per-pair later to debias R_blend."""
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
df=pf.read_table(CAT,columns=raw).to_pandas()
df=data_utils.source_select_reg(df,cuts=tr['regression_cuts'])
df=data_utils.rescale(df,pixel_rms=rc['pixel_rms'],pixel_size=rc['pixel_size'],zero_mag=rc['zero_mag'],
                      psf_fwhm=rc['psf_fwhm'],moffat_beta=rc['moffat_beta'])
shear=_catalogue_shear_scale(cfg,'response')
y=(df["delta_et1"]/shear).to_numpy(float)
pred=BlendingPredictor.load("/home/z/Zekang.Zhang/blendemu/models",tag="lsst_r_extnbr_ho",conditions=COND,device="cpu")
p=pred.bst_reg.predict(xgb.DMatrix(df[tr['features']]))*pred.y_std+pred.y_mean  # predicted response, all rows
resid=y-p   # true - pred (additive correction to ADD to the emulator prediction)
rs=df["r_input_s"].to_numpy(float); dd=df["distance"].to_numpy(float)
rs_edges=np.array([13,24,25,25.5,26,26.5,27,27.5,28,28.5,29.01])
d_edges=np.array([0,1,2,3,4,5,6,7,8,9,10.01])
ri=np.clip(np.digitize(rs,rs_edges)-1,0,len(rs_edges)-2)
di=np.clip(np.digitize(dd,d_edges)-1,0,len(d_edges)-2)
nr=len(rs_edges)-1; nd=len(d_edges)-1
D=np.zeros((nr,nd)); C=np.zeros((nr,nd),dtype=np.int64)
cell=ri*nd+di
sump=np.bincount(cell,weights=resid,minlength=nr*nd); cnt=np.bincount(cell,minlength=nr*nd)
D=(sump/np.maximum(cnt,1)).reshape(nr,nd); C=cnt.reshape(nr,nd)
print("global <true-pred> =",resid.mean(),flush=True)
print("Delta by r_s (dist-collapsed):",flush=True)
for i in range(nr):
    m=C[i].sum(); dm=np.sum(D[i]*C[i])/max(m,1)
    print(f"  r_s[{rs_edges[i]:.1f},{rs_edges[i+1]:.1f}): <Delta>={dm:+.5f}  N={m:,}",flush=True)
np.savez("/home/z/Zekang.Zhang/SBSI/results/emu_correction_rs_dist.npz",Delta=D,counts=C,rs_edges=rs_edges,d_edges=d_edges)
print("saved",flush=True); print("EMU_CORR_DONE",flush=True)
