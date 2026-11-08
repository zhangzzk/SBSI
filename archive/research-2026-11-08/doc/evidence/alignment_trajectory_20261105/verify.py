"""Small deterministic checks of profile accounting and diagnostic weighting."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('learning_trace_verify',HERE/'trace.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
v=torch.tensor([[[-.2,0.,2.75,110.],[-.2,0.,2.75,110.]],[[.4,0.,3.05,60.],[.4,0.,3.05,60.]]],dtype=torch.float64)
i=torch.tensor([[.2,0.],[.4,0.]],dtype=torch.float64)
o=torch.zeros((2,64,4),dtype=torch.float64)
t.profile_add(o,v,i,torch.tensor([0,1]))
torch.testing.assert_close(o[0,8],torch.tensor([1.,.04,-.04,1.],dtype=torch.float64))
torch.testing.assert_close(o[1,32+11],torch.tensor([1.,.16,.16,0.],dtype=torch.float64))
torch.testing.assert_close(o.sum(1),torch.tensor([[1.,.04,-.04,1.],[2.,.20,.12,1.]],dtype=torch.float64))
# Duplicating draws must not change the average or the shape regression.
o2=torch.zeros_like(o);t.profile_add(o2,v.repeat_interleave(3,dim=1),i,torch.tensor([0,1]))
torch.testing.assert_close(o,o2)
print('Known signs, radius/magnitude bins, case totals and draw normalization passed.')
spec=importlib.util.spec_from_file_location('learning_report_verify',HERE/'report.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
assert r.safe(np.array([1.,np.nan]))==[1.,None]
profile=np.zeros((3,2,2,32,4))
profile[:,:,:,8,:]=[1.,.04,.02,0.]
density=np.zeros((2,2,32,3));density[:,:,8,:]=[1.,-2.,4.]
z={'profile':profile,'density':density}
summary=r.summarize(z)
np.testing.assert_allclose(np.array(summary['alignment'],dtype=float)[:,:,8],.5)
np.testing.assert_allclose(np.array(summary['alignment_error'],dtype=float)[:,8],0.)
np.testing.assert_allclose(np.array(summary['local_nll_by_measured_bin'],dtype=float)[:,8],2.)
# Identical fixed response across cases has exactly zero contrast/SE.
boot=r.bootstrap_error(z)
np.testing.assert_array_equal(boot['value'],[0.,0.]);np.testing.assert_array_equal(boot['case_se'],[0.,0.])
print('Reducer sign, case pooling, empty-bin handling and bootstrap identity passed.')
