"""Quantify signed-size branch switching in the uniform seeded replay sample."""
import os,json
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_split_20261105/fit_audit')
raw=json.loads((D/'fit_flags_parameters.json').read_text());records=[]
for r in raw['records']:
    a,b=r['fits'][0],r['fits'][2]
    ga=np.array(a['g']);gb=np.array(b['g'])
    if r['MAG_AUTO']<25 and r['FLUX_RADIUS']>3 and a.get('flags')==0 and b.get('flags')==0 and ga@ga<1 and gb@gb<1:
        records.append([r['leg'],a['T'],b['T'],*(ga-gb),float(ga@gb)])
x=np.array(records);switch=x[:,1]*x[:,2]<0;norm=np.linalg.norm(x[:,3:5],axis=1);large=norm>.1
out={'sample':'Uniform512 IDs from case1000 union selection, both legs; results conditional on per-leg cuts and both fits usable/successful. Same ID may appear in both legs.','n_selected_case_leg_rows':len(x),'negative_T_fraction_across_fits':float((x[:,1:3]<0).mean()),'sign_switch_fraction':float(switch.mean()),'large_change_fraction':float(large.mean()),'fraction_large_changes_with_T_sign_switch':float(switch[large].mean()),'share_squared_g1_change_from_T_sign_switch':float(np.square(x[switch,3]).sum()/np.square(x[:,3]).sum()),'legs':{}}
for leg in [0,1]:
    y=x[x[:,0]==leg];out['legs'][str(leg)]={'n':len(y),'negative_T_fraction_each_repeat':(y[:,1:3]<0).mean(0).tolist()}
modes=json.loads((D/'fit_modes.json').read_text());rr=modes['records'];t=np.array([[f['T'] for f in r['fits']] for r in rr]);g=np.array([[f['g'] for f in r['fits']] for r in rr]);psf=np.array([[f['psf_full_pars'] for f in r['fits']] for r in rr])
out['selected_stress_examples']={'selection':modes['selection'],'n':len(rr),'T_sign_switches':int((t[:,0]*t[:,1]<0).sum()),'opposite_ellipticity_directions':int((np.sum(g[:,0]*g[:,1],1)<0).sum()),'maximum_PSF_parameter_change':float(np.max(np.abs(psf[:,0]-psf[:,1])))}
out['interpretation']='Negative T is explicitly permitted by original prior. This identifies large optimizer scatter, not a proof of mean calibration bias; no positivity cut or changed fitter was applied. The original success flag and disk-ellipticity usability do not exclude these solutions.'
(O/'fit_parameters_report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
