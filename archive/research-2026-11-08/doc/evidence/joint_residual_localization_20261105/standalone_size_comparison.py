"""Compare standalone size contributions using existing, replay-verified sums."""
import os,json
from pathlib import Path
import numpy as np
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
O=Path(__file__).resolve().parent
flow=json.loads((O/'flow_cell_report.json').read_text())
old=json.loads((O/'neighbour_report.json').read_text())['cuts']['radius3_mag25']
with np.load(O/'neighbour_sums.npz') as z:
    a=z['sums'][:,list(z['cuts']).index('radius3_mag25')]
    assert list(z['quantities'])==['n0','e01','e02','shape','model','selection_raw','mass1','mass2','pairs0','n1']
w=np.vstack([np.ones(40),np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)])
b=np.einsum('bc,csq->bsq',w,a);total=b.sum(1);den=total[:,0];mu=total[:,1:3]/den[:,None]
def ent(v):return {'value':float(v[0]),'se':float(v[1:].std(ddof=1))}
out={'definition':'Signed absolute response-error CONTRIBUTIONS to each experiment global selected-population response; not within-bin errors or m. Hard radius>3 pixels and MAG_AUTO<25. True size is circularized Re.','flow':'40 held-out training-family cases, original flow/selection; R11 and halftrace reported separately','emulator':'40 original development cases, h=.2 forward independently directed half-neighbour shears; projected halftrace per represented selected primary, including neighbour-count weighting. Baseline shape error and omitted selection response shown separately. Common usable labels; not the coherent constant-shear estimator.','bootstrap':'10000 paired whole-case resamples seed20260925 within each experiment; no cross-experiment pairing','groups':{}}
for name,idx in [('all',[0,1,2,3,4]),('size_015_04',[1,2]),('size_lt015',[0]),('size_015_025',[1]),('size_025_04',[2]),('size_04_07',[3]),('size_ge07',[4])]:
    t=b[:,idx].sum(1)
    shape=t[:,3]/den;model=t[:,4]/den
    selection=(t[:,5]-(mu*t[:,6:8]).sum(1))/den
    entry={'baseline_mass_fraction':float(t[0,0]/den[0]),'simulation_baseline_shape':ent(shape),'model_shape':ent(model),'baseline_error':ent(shape-model),'selection_response':ent(selection),'selected_error':ent(shape+selection-model)}
    out['groups'][name]={'flow':flow['groups'][name],'emulator':entry}
for key in ['simulation_baseline_shape','model_shape','baseline_error','selection_response','selected_error']:
    for stat in ['value','se']:np.testing.assert_allclose(out['groups']['all']['emulator'][key][stat],old[key][stat],atol=1e-12,rtol=0)
# The disjoint intrinsic-size bins must add to the full error in every bootstrap.
shape_error=(b[:,:,3]-b[:,:,4])/den[:,None]
sel=(b[:,:,5]-(mu[:,None,:]*b[:,:,6:8]).sum(-1))/den[:,None]
np.testing.assert_allclose((shape_error+sel).sum(1),(total[:,3]-total[:,4]+total[:,5]-(mu*total[:,6:8]).sum(1))/den,atol=1e-12,rtol=0)
(O/'standalone_size_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for name in ['all','size_015_04','size_015_025','size_025_04']:
    g=out['groups'][name];print(name,'flow R11',g['flow']['R11']['error_generated_selection'],'emulator',g['emulator'],flush=True)
