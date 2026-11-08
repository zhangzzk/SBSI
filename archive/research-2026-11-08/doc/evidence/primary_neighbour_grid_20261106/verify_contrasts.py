"""Verify finite-corner closure including changing selected denominators."""
import json
from pathlib import Path
import numpy as np

def contrasts(f,h):
    pp,pm,mp,mm=f
    primary=(pp-mp+pm-mm)/(4*h)
    neighbours=(pp-pm+mp-mm)/(4*h)
    coherent=(pp-mm)/(2*h)
    np.testing.assert_allclose(primary+neighbours,coherent,rtol=0,atol=1e-12)
    return primary,neighbours,coherent

rng=np.random.default_rng(202611069901)
for _ in range(100):
    weights=rng.uniform(.1,10,40);selection=rng.integers(0,2,(4,40));shape=rng.uniform(-.9,.9,(4,40))
    den=(weights*selection).sum(1);assert (den>0).all()
    contrasts((weights*selection*shape).sum(1)/den,.02)
h=.02
def mean(p,n):return .8*p+.1*n+3*p*p*n+5*p*n*n
corners=np.array([mean(h,h),mean(h,-h),mean(-h,h),mean(-h,-h)])
p,n,c=contrasts(corners,h)
np.testing.assert_allclose([p,n,c],[.8+5*h*h,.1+3*h*h,.9+8*h*h],rtol=0,atol=1e-12)
axis=(mean(h,0)-mean(-h,0)+mean(0,h)-mean(0,-h))/(2*h)
np.testing.assert_allclose(c-axis,8*h*h,rtol=0,atol=1e-12)
out={'passed':True,'definition':'Analytical design fixtures; no simulated images or calibration result',
     'checks':['Four-corner additivity with different selected denominators',
               'Primary and neighbour finite contrasts for known nonlinear example',
               'Axis-only sum need not equal coherent finite difference']}
(Path(__file__).parent/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print('Scene-replay estimator fixtures PASSED')
