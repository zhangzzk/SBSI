"""Known factorial effects and draw-level membership covariance fixtures."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
e=load('factorial_test_eval',HERE/'evaluate.py');r=load('factorial_test_report',HERE/'report.py')
properties=torch.ones((2,64,5),dtype=torch.float64)
wg=torch.zeros((2,64,1),dtype=torch.float64);wg[0]=1
wm=1-wg
matrix=np.zeros((1,2,2))
for inp,values in enumerate([[1.,2.],[3.,5.]]):
    change=.04*torch.tensor(values,dtype=torch.float64)[:,None].expand(-1,64)
    for mask,weight in enumerate([wg,wm]):
        sums=e.weighted_statistics(weight,change,properties).sum(0)[0]
        matrix[0,inp,mask]=float(sums[1]/sums[0]/.04)
np.testing.assert_allclose(matrix,[[[1,2],[3,5]]],atol=1e-12,rtol=0)
effects=r.effects(matrix)
for name,expected in [('symmetric_photometry',2.5),('symmetric_membership',1.5),('interaction',1.),('diagonal_change',4.)]:
    np.testing.assert_allclose(effects[name],expected,atol=1e-12,rtol=0)
change=torch.zeros((1,64),dtype=torch.float64);change[:,::2]=.08
weight=(change>0).double()[...,None]
moment=e.weighted_statistics(weight,change,properties[:1]).sum(0)[0]
np.testing.assert_allclose(float(moment[1]/moment[0]/.04),2.,atol=1e-12,rtol=0)
(HERE/'verification.json').write_text(json.dumps({'passed':True,'definition':'Synthetic implementation fixtures, not scientific data',
    'checks':['Known two-target 2x2 response means and ordered effects','Correct symmetric effects and interaction','Selection remains paired with each individual draw']},indent=2)+'\n')
print('Factorial analytical fixtures PASSED')
