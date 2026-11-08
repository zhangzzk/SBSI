"""Small analytical tests: own-only, opposite-only, nonlinear and selection."""
from pathlib import Path
import json
import numpy as np
from contrasts import contrasts

g=.02*np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
fraction=np.array([.3,.7]);self_response=np.array([.8,.6]);neighbour_response=np.array([.04,-.03])
for a,b in [(self_response,np.zeros(2)),(np.zeros(2),neighbour_response),(self_response,neighbour_response)]:
    F=fraction*(.1+a*g+b*g[:,::-1])
    c=contrasts(F[None])
    np.testing.assert_allclose(c['own_group'],fraction@a,atol=1e-14)
    np.testing.assert_allclose(c['opposite_group'],fraction@b,atol=1e-14)
# Nonlinear interaction need not be attributed to pure self/neighbour response,
# but the two paths must still recover the exact coherent finite difference.
F=fraction*(.1+self_response*g+neighbour_response*g[:,::-1]+200*g*g[:,::-1]+150*g**2*g[:,::-1])
c=contrasts(F[None]);np.testing.assert_allclose(c['coherent_total'],(F[0].sum()-F[3].sum())/.04)
# State-dependent counts: global denominator, core/rest decomposition additive.
N=np.array([[30,70],[33,65],[28,75],[32,67]])
F=N*(.1+self_response*g+neighbour_response*g[:,::-1])/N.sum(1)[:,None]
all_c=contrasts(F[None]);core_c=contrasts((F*np.array([.2,.8]))[None]);rest_c=contrasts((F*np.array([.8,.2]))[None])
for key in all_c:np.testing.assert_allclose(core_c[key]+rest_c[key],all_c[key],atol=1e-14)
# Exactly the reducer's population-axis selection, including singleton slices.
s=np.arange(3*4*2*2*3).reshape(3,4,2,2,3)
np.testing.assert_array_equal(np.take(s[...,1],[0,1],axis=3).sum(3),s[...,0,1]+s[...,1,1])
np.testing.assert_array_equal(np.take(s[...,1],[1],axis=3).sum(3),s[...,1,1])
out={'passed':True,'checks':['own-only response','opposite-only response','combined response','nonlinear exact coherent closure','selection-dependent denominator','core/complement additivity','reducer population-axis indexing']}
# Selection changes at fixed shapes and shape changes at fixed selection must
# be assigned to different terms of the exact finite-difference ratio identity.
ea=np.array([.2,-.1,.4]);eb=np.array([.1,-.2,.3])
for sa,sb in [(np.array([1.,1.,0.]),np.array([1.,1.,0.])),(np.array([1.,1.,0.]),np.array([0.,1.,1.]))]:
    for a,b in [(ea,ea),(ea,eb)]:
        fa=sa@a/sa.sum();fb=sb@b/sb.sum();db=.5*(sa.sum()+sb.sum())
        shape=.5*(sa+sb)@(a-b)/db
        select=(.5*(a+b)@(sa-sb)-.5*(fa+fb)*(sa.sum()-sb.sum()))/db
        np.testing.assert_allclose(shape+select,fa-fb,atol=1e-14)
        if np.array_equal(sa,sb):assert select==0
        if np.array_equal(a,b):assert shape==0
out['checks'].append('exact shape/selection product and ratio decomposition')
(Path(__file__).parent/'contrast_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
