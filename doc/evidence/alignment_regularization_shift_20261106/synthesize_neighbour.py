"""Compare paired conditional-input diagnostics without adding unrelated errors."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106')
assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
parser=argparse.ArgumentParser();parser.add_argument('--nine-only',action='store_true');args=parser.parse_args()
ARMS=['nine_weak'] if args.nine_only else ['baseline','nine_weak']
FOURSTATE='fourstate_nine_weak_report' if args.nine_only else 'fourstate_report'
OUTPUT='neighbour_input_nine_synthesis' if args.nine_only else 'neighbour_input_synthesis'
CASES=[1000,1040,1080,1120]
CUT='mag25_radius_gt0.60'
def ent(x):return {'value':float(x[0]),'case_se':float(x[1:].std(ddof=1))}
def load(name):
    with np.load(ROOT/(name+'.npz')) as z:
        np.testing.assert_array_equal(z['cases'],CASES);assert int(z['seed'])==202611069301
        return z['values'].copy(),list(z['cuts']),list(z['quantities'])
g,gcuts,gq=load(FOURSTATE)
measured,mcuts,mq=load('measured_transport_report')
reference=measured[mcuts.index(CUT)]
simshape=reference[:,mq.index('simulation_shape')];selection=reference[:,mq.index('simulation_selection')]
rows={};models={}
for i,arm in enumerate(ARMS):
    generated=g[i,gcuts.index(CUT)]
    np.testing.assert_allclose(generated[:,gq.index('simulation_shape_term')],simshape,atol=2e-12,rtol=0)
    np.testing.assert_allclose(generated[:,gq.index('simulation_selection_term')],selection,atol=2e-12,rtol=0)
    models['generated_'+arm]=generated[:,gq.index('model')]
    cc,ccuts,cq=load('conditional_transport_'+arm+'_report');conditional=cc[ccuts.index(CUT)]
    np.testing.assert_allclose(conditional[:,cq.index('simulation_shape')],simshape,atol=2e-12,rtol=0)
    np.testing.assert_allclose(conditional[:,cq.index('simulation_selection')],selection,atol=2e-12,rtol=0)
    models['measured_photometry_'+arm]=conditional[:,cq.index('anchored_model_shape')]
models['measured_shape_and_photometry']=reference[:,mq.index('anchored_model_shape')]
for name,prediction in models.items():
    rows[name]={'model_shape_response':ent(prediction),'shape_error':ent(simshape-prediction),
                'total_error_including_selection':ent(simshape+selection-prediction)}
changes={}
for arm in ARMS:
    changes[arm]={'generated_to_measured_photometry':ent(models['measured_photometry_'+arm]-models['generated_'+arm]),
                  'conditional_to_measured_shape':ent(models['measured_shape_and_photometry']-models['measured_photometry_'+arm])}
counts={}
w=np.vstack([np.ones(4),np.random.default_rng(202611069301).multinomial(4,np.full(4,.25),10000)])
for arm in ARMS:
    measured_mass=[];generated_mass=[]
    for case in CASES:
        with np.load(ROOT/'fourstate'/arm/f'case{case}.npz') as z:
            k=list(z['cuts']).index(CUT)
            measured_mass.append(z['sim'][1,:,k,:,0].sum())
            generated_mass.append(z['model'][2,:,k,:,0].sum())
    counts[arm]=ent((w@np.asarray(generated_mass))/(w@np.asarray(measured_mass)))
out={'cases':CASES,'cut':CUT,'definition':'Common-all-four usability; equal average of separately normalized A/B opposite-group responses. Errors=simulation-model; case bootstrap SE; fixed checkpoints.',
     'simulation_shape':ent(simshape),'simulation_selection':ent(selection),'rows':rows,'paired_model_response_changes':changes,
     'generated_to_measured_selected_mass_ratio':counts,
     'limitations':['Four development scenes, not independent acceptance.',
                    'Generated-to-measured-photometry substitution changes photometry and selection as well as conditioning; not a unique causal separation of those effects.',
                    'Conditional-to-measured-shape comparison retains the same measured photometry and selection weights.',
                    'These within-group opposite responses are not the globally normalized coherent contribution and must not be doubled.']}
(ROOT/(OUTPUT+'.json')).write_text(json.dumps(out,indent=2)+'\n')
for name in [FOURSTATE,'measured_transport_report',*['conditional_transport_'+arm+'_report' for arm in ARMS],OUTPUT]:
    (HERE/(name+'.json')).write_bytes((ROOT/(name+'.json')).read_bytes())
def fmt(d):return f"{d['value']:+.6f} ± {d['case_se']:.6f}"
lines=['# Matched neighbour-response input diagnostics','',out['definition'],'',
       'Target: measured/generated radius>3pixels (0.6arcsec), magnitude<25 as indicated.',
       f"Measured shape response: {fmt(out['simulation_shape'])}; measured selection response: {fmt(out['simulation_selection'])}.",'',
       '| Inputs | Predicted shape response | Shape error | Error including selection |',
       '|---|---:|---:|---:|']
for name,row in rows.items():lines.append('| '+name+' | '+' | '.join(fmt(row[k]) for k in ['model_shape_response','shape_error','total_error_including_selection'])+' |')
lines+=['','## Paired changes in model response','','| Flow | Generated to measured photometry | Conditional to measured shape |','|---|---:|---:|']
for arm,row in changes.items():lines.append('| '+arm+' | '+fmt(row['generated_to_measured_photometry'])+' | '+fmt(row['conditional_to_measured_shape'])+' |')
lines+=['','## Limits','']+['- '+s for s in out['limitations']]
lines+=['','Generated/measured selected-mass ratios on the common cohort: '+', '.join(arm+' '+fmt(value) for arm,value in counts.items())+'. Matching counts does not establish matching response-weighted input distributions.']
(HERE/('neighbour_input_nine_results.md' if args.nine_only else 'neighbour_input_results.md')).write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
