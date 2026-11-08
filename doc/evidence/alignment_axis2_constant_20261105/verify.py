import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('report.py');s=importlib.util.spec_from_file_location('r',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
x=np.zeros((3,2,5));sim=np.zeros((3,2,3));mass=np.array([[20,31],[40,19],[23,55]])
for c in range(3):
 for l,sgn in enumerate([1,-1]):
  x[c,l,0]=mass[c,l];x[c,l,1:3]=mass[c,l]*np.array([.01,.7])*sgn*.02
  x[c,l,3:]=mass[c,l]*np.array([.02,.1])*sgn*.02
  sim[c,l,0]=mass[c,l];sim[c,l,1:]=mass[c,l]*np.array([.04,.824])*sgn*.02
v=r.estimates(sim,x,np.ones((1,3)))
np.testing.assert_allclose(v['model_response'],[[.03,.8]])
np.testing.assert_allclose(v['m_percent'],[3.])
np.testing.assert_allclose(v['delta_response'],[[.01,.024]])
print('AXIS2_REDUCER_KNOWN_RESPONSE_PASSED')
