"""Lightweight arithmetic on seven saved fixed-probe summaries; no model work."""
import json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_factor_swap_20261107')
cfg=json.loads((HERE/'manifest.json').read_text())
assembly=json.loads((ROOT/'assembly_completed.json').read_text())
probes={name:json.loads((ROOT/name/'probe.json').read_text()) for name in cfg['arms']}
def terms(a,b,ba,ab):
    result={'photo_at_old_shape':ba-a,'shape_at_old_photo':ab-a,
            'interaction':b-ba-ab+a,'total':b-a,
            'photo_at_new_shape':b-ab,'shape_at_new_photo':b-ba}
    assert math.isclose(result['photo_at_old_shape']+result['shape_at_old_photo']+result['interaction'],result['total'],rel_tol=1e-10,abs_tol=1e-12)
    return result
out={'definition':'Fixed common-random-number own-data probes; changes in model response, without case uncertainty. These are not joint constant-shear residuals.','comparisons':{},'clipping_diagnostic':{}}
for contrast in cfg['factorial_comparisons']:
    a,b,ba,ab=[probes[contrast[k]] for k in ['old','new','photo_only','shape_only']]
    result={'nll':terms(*[x['nll']['nll'] for x in [a,b,ba,ab]])}
    for split in ['train','validation']:
        result[split]={}
        for i,cut in enumerate(cfg['target_cuts']):
            result[split][cut]={label:terms(*[x[split]['mean_response'][i][j] for x in [a,b,ba,ab]]) for j,label in enumerate(['R11','R12','R21','R22'])}
    out['comparisons'][contrast['name']]=result
old=ROOT.parent/'neighbour_moment_guard_20261106'
for arm in ['control','weighted']:
    history=json.loads((old/arm/'history.json').read_text());norm=[row['gradient_norm'] for row in history]
    out['clipping_diagnostic'][arm]={'steps':len(norm),'clipped_steps':sum(x>5 for x in norm),'minimum_norm':min(norm),'maximum_norm':max(norm),'mean_norm':statistics.mean(norm),'interpretation':'Global clipping is an available indirect coupling; this diagnostic does not measure its causal contribution.'}
out['direct_factor_max_abs']=max(x['conditional_shape_log_density_max_abs'] for x in assembly['arms'].values())
for path in [ROOT/'own_probe_factorial.json',HERE/'own_probe_factorial.json']:path.write_text(json.dumps(out,indent=2)+'\n')
lines=['# Frozen-factor own-data response probes','',out['definition'],'',
       'Target: radius>3 pixels and magnitude<25. Each row is a change in model response.','',
       '| Comparison | Split | Component | Photometry at old shapes | Shapes at old photometry | Interaction | Total |',
       '|---|---|---|---:|---:|---:|---:|']
for name,result in out['comparisons'].items():
    for split in ['train','validation']:
        for component in ['R11','R22']:
            row=result[split]['mag25_radius_gt0.60'][component]
            lines.append(f'| {name} | {split} | {component} | '+' | '.join(f'{row[key]:+.6f}' for key in ['photo_at_old_shape','shape_at_old_photo','interaction','total'])+' |')
(HERE/'own_probe_results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
