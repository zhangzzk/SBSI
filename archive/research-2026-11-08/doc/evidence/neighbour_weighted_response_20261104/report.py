"""Count contributing neighbours and bootstrap per-primary response residuals."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
P=Path('/project/ls-gruen/users/zekang.zhang')
D=P/'blendemu_runs/disk_response_moment_galsbi_cut_20261103/pairs'
manifest=json.loads((D/'manifest.json').read_text())
with np.load(P/'sbsi_caches/galsbi_emulator_check_20261104/sums.npz') as z:
    sums=z['sums'][:,:,:4,:,:].sum((1,2,3)); q=list(z['quantities'])
counts=[]
for case in range(40):
    ids=[]; pairs=0
    for part in manifest['parts']:
        if part['case']!=case: continue
        with np.load(D/part['path']) as z:
            x=z['x']; primary=z['primary']; keep=x[:,8]<25
            assert np.all(np.exp(x[keep,7])>3)
            ids.append(np.unique(primary[keep])); pairs+=int(keep.sum())
    n=len(np.unique(np.concatenate(ids)))
    assert pairs==sums[case,0], (case,pairs,sums[case,0])
    counts.append(n)
    print(f'case {case}: {pairs} pairs / {n} represented primaries = {pairs/n:.6f}',flush=True)
n=np.asarray(counts)
w=np.random.default_rng(20260925).multinomial(40,np.full(40,1/40),size=10000)
s=sums[:,q.index('label')]; m=sums[:,q.index('galsbi_dev_pred')]
bn=w@n
res={'definition':'Sum of pair response residuals divided by distinct represented primary galaxies, simulation minus model.',
 'population':'40 development cases; measured baseline radius>3 pixels and magnitude<25; denominator includes primaries with at least one retained valid pair only; retains original pair-label exclusions.',
 'pairs':int(sums[:,0].sum()),'represented_primaries':int(n.sum()),
 'mean_contributing_neighbours':float(sums[:,0].sum()/n.sum()),
 'mean_contributing_neighbours_se':float(np.std((w@sums[:,0])/bn,ddof=1)),
 'simulation_per_primary':float(s.sum()/n.sum()),'model_per_primary':float(m.sum()/n.sum()),
 'sim_minus_model_per_primary':float((s-m).sum()/n.sum()),
 'paired_case_se':float(np.std((w@(s-m))/bn,ddof=1)),
 'bootstrap':'10000 paired whole-case resamples, seed20260925, recomputing neighbour count and response together; fixed model.',
 'limitations':'Summed projected pair-response diagnostic; not a direct coherent-shear galaxy response or Mobius-composed emulator error. Different populations, shear amplitudes and selection treatment from joint constant-shear check.',
 'cases':list(range(40)),'primaries_by_case':counts}
np.savez(O/'primary_counts.npz',cases=np.arange(40),primaries=n,pairs=sums[:,0])
(O/'report.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k not in ['cases','primaries_by_case']},indent=2),flush=True)
