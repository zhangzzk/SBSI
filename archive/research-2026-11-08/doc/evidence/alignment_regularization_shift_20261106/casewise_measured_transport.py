"""Inspect per-scene signs without treating scenes as independent galaxies."""
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106')
assert os.environ.get('SLURM_JOB_ID')
rows=[];all_arrays=[]
def summarize(ss,ed,pr):
    s=ss[:,3].sum(-2);ed=ed[3].sum(-2);pr=pr[3].sum(-2)
    shape=selection=prediction=0.
    edge_states=[(0,2),(1,3),(0,1),(2,3)]
    for g in [0,1]:
        for edge in ([2,3] if g==0 else [0,1]):
            a,b=edge_states[edge];dbar=.5*(s[a,g,0]+s[b,g,0]);fbar=.5*(s[a,g,1]/s[a,g,0]+s[b,g,1]/s[b,g,0])
            shape+=ed[edge,g,0]/dbar/.16
            selection+=(ed[edge,g,1]-fbar*(s[a,g,0]-s[b,g,0]))/dbar/.16
            prediction+=.5*(pr[edge,g,0]+pr[edge,g,1])/dbar/.16
    return {'simulation_shape':shape,'model_shape':prediction,'shape_error':shape-prediction,
            'selection_response':selection,'total_error':shape+selection-prediction}
for case in [1000,1040,1080,1120]:
    with np.load(ROOT/'measured_transport'/f'case{case}.npz') as z:
        arrays=[z[n] for n in ['sim','edges','pred']]
    rows.append({'case':case,**summarize(*arrays)});all_arrays.append(arrays)
pooled=summarize(*(sum(a[i] for a in all_arrays) for i in range(3)))
reference=json.loads((ROOT/'measured_transport_report.json').read_text())['cuts']['mag25_radius_gt0.60']['equal_group_average']
for key,other in [('simulation_shape','simulation_shape'),('model_shape','anchored_model_shape'),('shape_error','shape_error'),('selection_response','simulation_selection'),('total_error','total_error')]:
    np.testing.assert_allclose(pooled[key],reference[other]['value'],atol=2e-12,rtol=0)
out={'cases':rows,'pooled':pooled,'definition':'Casewise equal A/B average, common usability and target cut; descriptive signs, not new uncertainty estimate.'}
(HERE/'casewise_measured_transport.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
