#!/usr/bin/env python
"""CPU-only fixed analytic phase control on authorized saved development states."""
import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.analytic_phase_control import (EPSILON,RULE,MOMENT_COLUMNS,
    phase_coefficient,analytic_phase_control,summarize_score_comparison)
from sbsi.phase_control import FrozenPhaseControl,digest,load_phase_diagnostic


GROUPS={'v1':4,'v5':8,'v6v100':8,'v7':8}


def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    return value


def load_saved(root,phase_root,group,index,frozen):
    run=root/f'dev_{group}_{index}'; path=phase_root/f'dev_{group}_{index}.json'
    diagnostic,arrays=load_phase_diagnostic(path)
    protocol=json.loads((run/'protocol.json').read_text())
    result=json.loads((run/'result.json').read_text())
    if (diagnostic.get('stride')!=1 or diagnostic.get('retained_particles')!=protocol['particles']
            or diagnostic.get('derivative_protocol')!='phase-fine-difference-v2'
            or diagnostic['run']!=str(run.resolve())
            or result['status']!='completed' or any(result.get('extinct',[]))
            or result['protocol_sha256']!=digest(run/'protocol.json')
            or result['particles_sha256']!=digest(run/'particles.npz')
            or result['observations']!=protocol['observations']
            or result['final_stage']['beta']!=1.):
        raise ValueError('unchanged complete full-particle endpoint v2 diagnostic required')
    expected=(len(protocol['observations']),protocol['particles'])
    if (arrays['raw_scores'].shape!=expected+(2,)
            or arrays['phase_control'].shape!=expected
            or arrays['delta_log_kernel'].shape!=expected):
        raise ValueError('all particles in original order and original N required')
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
        if not np.array_equal(saved['sampled_particle_indices'],np.arange(protocol['particles'])):
            raise ValueError('stride1/full ordered states required')
    with np.load(run/'particles.npz',allow_pickle=False) as saved:
        summaries=saved['summaries'].copy()
        if (summaries.shape!=expected+(9,) or not np.isfinite(summaries).all()
                or not np.array_equal(saved['scores'],arrays['raw_scores'])):
            raise ValueError('saved summaries and raw score must exactly align')
    identity=diagnostic['target_identity']
    if identity!=frozen.metadata['target_identity'] or protocol['observations']!=frozen.metadata['observations']:
        raise ValueError('fixed coefficient identity mismatch')
    if group=='v1':
        if protocol['seed']!=frozen.metadata['pilot_seeds'][index]:
            raise ValueError('only the four declared original pilots permitted')
    else:
        frozen.validate_production(protocol['seed'],protocol['observations'],identity)
        if diagnostic.get('coefficients',{}).get('sha256')!=digest(phase_root/'frozen_v1_coefficients.json'):
            raise ValueError('constant-control artifact changed')
    constant=arrays['raw_scores']-frozen.coefficients[:,None,:]*arrays['phase_control'][...,None]
    if group!='v1':
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
            if not np.array_equal(constant,saved['corrected_scores']):
                raise ValueError('frozen constant comparison must replay exactly')
    control=analytic_phase_control(summaries[...,:2],arrays['phase_control'])
    a,prime=phase_coefficient(summaries[...,:2])
    if diagnostic['log_normalizer']!=result['log_normalizer']:
        raise ValueError('normalizer changed')
    return dict(seed=protocol['seed'],observations=protocol['observations'],identity=identity,
        particles=protocol['particles'],log_normalizer=result['log_normalizer'],
        raw_scores=arrays['raw_scores'],constant_scores=constant,
        analytic_scores=arrays['raw_scores']-control,analytic_control=control,
        coefficient_a=a,phase_derivative_a=prime,phase_score=arrays['phase_control'],
        delta_log_kernel=arrays['delta_log_kernel'],
        source_sha256={str(p.resolve()):digest(p) for p in
            [run/'protocol.json',run/'result.json',run/'particles.npz',path,path.with_suffix('.npz')]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('fresh analytic-control output directory required')
    coefficient_path=args.phase_root/'frozen_v1_coefficients.json'
    frozen=FrozenPhaseControl.load(coefficient_path)
    populations={g:[load_saved(args.root,args.phase_root,g,i,frozen) for i in range(n)] for g,n in GROUPS.items()}
    flat=[p for runs in populations.values() for p in runs]
    if len({p['seed'] for p in flat})!=len(flat) or any(p['identity']!=flat[0]['identity'] for p in flat):
        raise ValueError('all populations must have distinct seeds and identical targets')
    args.output.mkdir()
    groups={}; output_hashes={}
    for group,runs in populations.items():
        rows=summarize_score_comparison([p['log_normalizer'] for p in runs],
            [p['raw_scores'] for p in runs],[p['constant_scores'] for p in runs],
            [p['analytic_control'] for p in runs],[p['phase_score'] for p in runs],
            [p['delta_log_kernel'] for p in runs])
        for row,obs in zip(rows,runs[0]['observations']): row['observation']=obs
        groups[group]=dict(observations=rows,seeds=[p['seed'] for p in runs],
            particles=[p['particles'] for p in runs],
            constant_control_comparison='in-sample descriptive pilot comparison' if group=='v1' else 'independent evaluation of frozen pilot coefficient',
            analytic_control_comparison='fixed analytic rule; no fit on any group')
        for i,p in enumerate(runs):
            out=args.output/f'dev_{group}_{i}.npz'
            fields={key:p[key] for key in ['raw_scores','constant_scores','analytic_scores','analytic_control',
                'coefficient_a','phase_derivative_a','phase_score','delta_log_kernel']}
            np.savez_compressed(out,**fields)
            output_hashes[str(out.resolve())]=digest(out)
    report=dict(schema='sbsi-analytic-phase-control-v1',epsilon=EPSILON,rule=RULE,
        groups=groups,moment_columns=MOMENT_COLUMNS,target_identity=flat[0]['identity'],
        source_sha256={k:v for p in flat for k,v in p['source_sha256'].items()},
        frozen_constant_artifact=dict(path=str(coefficient_path.resolve()),sha256=digest(coefficient_path)),
        output_sha256=output_hashes,script_sha256=digest(__file__),
        module_sha256=digest(Path(__file__).resolve().parents[1]/'sbsi/analytic_phase_control.py'),
        normalization='Original full N within every population; original Zhat weighting between populations. F64 diagnostic Zhat is multiplied by mean(exp(delta_logK)).',
        raw_score_and_normalizer_gates_unchanged=True,accepted_sampler=False,
        limitations=['No exact zero-mean claim for the raw F32-sampled numerical measure.',
            'F64 is a numerical diagnostic of the continuous periodic identity, not exact rounded arithmetic.',
            'Finite independent-population MCSE can miss shared rare modes.',
            'Coefficient sign, epsilon and formula were fixed before outcomes; no fitting or recentering.',
            'No new model forward, GPU, heldout, sampler or target changes.'])
    with (args.output/'result.json').open('x') as handle:
        json.dump(serial(report),handle,indent=2,allow_nan=False);handle.write('\n')
    print('ANALYTIC_PHASE_CONTROL',args.output,flush=True)


if __name__=='__main__': main()
