"""Gate mixed-sign interpretation on unchanged-input rendering replay."""
import os,json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim')
a=D/'constant_g002/sims/case1000_0.02/real0';b=D/'constant_split_20261105/control/sims/case1000_0.02/real0'
report={'case':1000,'passed':False,'checks':{}}
try:
 for rel in ['images/original/tile180.0_-0.5_bandr_rot0.fits','images/original/psf_tile180.0_-0.5_bandr/psf_ima.fits']:
  with fits.open(a/rel,memmap=True) as fa,fits.open(b/rel,memmap=True) as fb:
   x,y=fa[0].data,fb[0].data;assert x.shape==y.shape
   different=0;maximum=0.
   for lo in range(0,len(x),512):
    xx,yy=x[lo:lo+512],y[lo:lo+512];different+=int(np.count_nonzero(xx!=yy));maximum=max(maximum,float(np.max(np.abs(xx.astype(float)-yy))))
   report['checks'][rel]={'different_pixels':different,'max_abs_difference':maximum,'shape':list(x.shape)}
   assert different==0,f'Pixel replay failed: {rel}: {different}, {maximum}'
 for rel,key in [('catalogues/input/gals_info_tile180.0_-0.5.feather','index_input'),('catalogues/SExtractor/tile180.0_-0.5_bandr_rot0.feather','NUMBER'),('catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather','id_input')]:
  x=pd.read_feather(a/rel).sort_values(key).reset_index(drop=True);y=pd.read_feather(b/rel).sort_values(key).reset_index(drop=True)
  pd.testing.assert_frame_equal(x,y,check_exact=True);report['checks'][rel]={'rows':len(x),'exact':True}
 report['passed']=True
finally:
 (O/'control_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
