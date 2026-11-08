"""Write readable results only after the independent numerical audit passes."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
r=json.loads((H/'report.json').read_text());a=json.loads((H/'completed_audit.json').read_text());assert hashlib.sha256((H/'report.json').read_bytes()).hexdigest()==a['report_sha256']
cfg=json.loads((H/'manifest.json').read_text())
def val(d,i=None):
 v=d['value'] if i is None else d['value'][i];e=d['jackknife_se'] if i is None else d['jackknife_se'][i]
 return f'{v:+.4f} ±{e:.4f}'
s='''# Frozen joint Cartesian: full comparison

## Both components and all nine cuts

Same40 scenes, new paired-seed measured reference, independently selected
per-leg populations, generated photometry and predicted usability. Values are
m=100*(Rmeasured/Rmodel−1), in percent; errors are whole-scene jackknife standard
errors in percentage points. Both models use64 portable antithetic draws and
the same case/block seeds. No training or empirical calibration correction.

| Radius lower bound (pixels) | Magnitude upper bound | Joint m1 (%) | Joint m2 (%) | Latest m1 (%) | Latest m2 (%) |
|---:|---:|---:|---:|---:|---:|
'''
for c in cfg['cuts']:
 row=r['cuts'][c[0]];s+='| '+' | '.join([f'{c[1][1]:.1f}',f'{c[2]:.1f}',val(row['joint_m_percent'],0),val(row['joint_m_percent'],1),val(row['latest_m_percent'],0),val(row['latest_m_percent'],1)])+' |\n'
s+='\n## Joint nine-cut objective\n\n'
for name,d in r['joint_nine_cut_rms_percent'].items():s+=f'- {name}: {val(d)}'+(' percentage points' if 'minus' in name else '%')+'.\n'
s+='\n## Central matrices and random-stream check\n\n'
center=r['cuts']['mag25_radius_gt0.60']
for name in ['measured_matrix','joint_matrix','latest_matrix']:
 s+=name+':\n\n```\n'+json.dumps(center[name]['value'])+'\n```\n\n'
s+='The prior small recheck used historical CUDA draws. This run uses the same\nportable stream as the latest flow. The old central estimate is retained:\n\n'
for name,d in r['central_g1_stream_comparison'].items():s+=f'- {name}: {val(d)}.\n'
s+='\n## Verification and scope\n\n'
s+=f"All80 case/axis output receipts, measured sums, fixed model identities and\nhalf-bank identities passed; {a['independent_scalar_point_jackknife_checks']} independent scalar point/jackknife checks passed.\n"
s+='''The A40-8Q pilot separately replays the historical stream and gates the full
run. Per-cut half-bank sensitivity, paired contrasts and confidence intervals
are preserved in `report.json`. The reported uncertainties condition on the
fitted models and existing scenes; they are not training-seed uncertainties or
fresh acceptance evidence. The held broad array remains paused.
[Response conventions](../../CONVENTIONS.md#6-response-estimators),
[protocol](rule.md#authorized-comparison).
'''
(H/'results.md').write_text(s)
print('READABLE_RESULTS_WRITTEN',flush=True)
