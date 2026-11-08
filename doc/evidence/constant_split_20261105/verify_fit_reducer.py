"""Analytical sample fixture for fit-repeat response and uncertainty scaling."""
from pathlib import Path
import tempfile,json
import numpy as np
HERE=Path(__file__).parent
arrays=[];sizes=[];case_reports=[]
rng=np.random.default_rng(7)
for c in range(4):
    n=500;pop=5000;a=np.ones((3,2,n,2))
    e=rng.uniform(-.2,.2,n)
    for leg,sign in enumerate([1,-1]):
        a[0,leg,:,1]=e+sign*.02*.8
        a[1,leg,:,1]=a[0,leg,:,1]+sign*.0001
        a[2,leg,:,1]=a[1,leg,:,1]+rng.normal(0,.001,n)
    arrays.append(a);sizes.append((n,pop))
source=(HERE/'fit_reduce.py').read_text();tail=source[source.index('factors=np.array'):]
with tempfile.TemporaryDirectory(prefix='sbsi_fit_verify_') as tmp:
    env=dict(globals(),O=Path(tmp));exec(compile(tail,'<actual fit aggregation>','exec'),env)
    result=json.loads((Path(tmp)/'fit_report.json').read_text())['differences']
    r=result['repeat0_minus_original'];np.testing.assert_allclose(r['value'],.005,atol=1e-13)
    assert r['conditional_subsampling_se']<1e-13
    # Only one member has iid fitter noise in this fixture; the single-fit
    # projection divides pair-difference variance by two by definition.
    expected=.001/(.04*np.sqrt(4*5000))
    ratio=result['repeat1_minus_repeat0']['estimated_single_fit_full_catalogue_se']/expected
    assert .95<ratio<1.05,ratio
(HERE/'fit_reducer_verification.json').write_text(json.dumps({'passed':True,'checks':['known signed response change','constant shift has zero subsampling variance','independent fitter noise scales to full population correctly'],'fixture':'Synthetic; not scientific evidence'},indent=2)+'\n')
print('Fit reducer analytical fixture PASSED')
