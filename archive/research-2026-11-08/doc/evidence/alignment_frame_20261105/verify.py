"""Change-of-variables, support and persistence checks before experiment."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from sbsi.measurement_model import load_measurement_model
from sbsi.fixed_g0_domain import FLOW_FEATURES

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experimental_frame',HERE/'frame_model.py')
fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_f8c8_20261102')
torch.set_num_threads(1);torch.manual_seed(123)
old=load_measurement_model(ROOT/'base/flow/nll/selected.pt',device='cpu')
condition,target=old.condition_preprocessor,old.target_transform
config=json.loads((ROOT/'base/flow/protocol.json').read_text())['model_config']
scale=float(np.sqrt(np.mean(np.square(config['disk_coordinate_scales']))))
config['disk_coordinate_means']=[0.,0.];config['disk_coordinate_scales']=[scale,scale]
model=fm.FrameFlow(config,condition,'frame')
raw=np.tile(np.array([.2,.1,1.,24.,.2,.01,.02,.03]),(12,1));raw[:4,:2]=[[0,0],[.3,0],[0,.3],[-.2,.3]]
context=torch.as_tensor(condition.transform_frame(pd.DataFrame(raw,columns=FLOW_FEATURES)))
e=torch.randn(12,2,dtype=torch.float64)*.1
local=model.rotate(e,context)
torch.testing.assert_close(model.rotate(local,context,inverse=True),e,atol=1e-14,rtol=1e-14)
torch.testing.assert_close(local.square().sum(1),e.square().sum(1),atol=1e-14,rtol=1e-14)
for i in range(4):
    def rotate_standard(y):
        ep=y[:2]*model.base.disk_shape_scales+model.base.disk_shape_means
        p=model.rotate(ep[None,:],context[i:i+1])[0]
        return torch.cat([(p-model.base.disk_shape_means)/model.base.disk_shape_scales,y[2:]])
    jac=torch.autograd.functional.jacobian(rotate_standard,torch.tensor([.2,-.1,.4,1.],dtype=torch.float64))
    torch.testing.assert_close(torch.linalg.det(jac),torch.tensor(1.,dtype=torch.float64),atol=1e-12,rtol=1e-12)
# Nontrivial couplings exercise density/Jacobian consistency beyond identity.
with torch.no_grad():
    for layer in model.base.base.layers:
        layer.net[-1].weight.normal_(0,.001)
z=torch.randn(12,4)
physical,ld=model.forward_physical(z,context)
means=torch.as_tensor(target.means,dtype=torch.float64);scales=torch.as_tensor(target.scales,dtype=torch.float64)
lp=model.log_prob((physical-means)/scales,context)
expected=-.5*(z.double().square()+math.log(2*math.pi)).sum(1)-ld+scales.log().sum()
torch.testing.assert_close(lp,expected,atol=3e-5,rtol=3e-6)
assert bool((physical[:,:2].square().sum(1)<1).all() and (physical[:,2:]>0).all())
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'candidate.pt';fm.save(path,model,condition,target,config,{})
    loaded,_,_,_=fm.load(path,device='cpu')
    torch.testing.assert_close(loaded.forward_physical(z,context)[0],physical,rtol=0,atol=0)
receipt={'rotation_inverse_and_norm':'passed','standardized_rotation_determinant':'1, including zero-intrinsic fallback',
         'density_change_of_variables_max_error':float((lp-expected).abs().max()),'physical_support':'passed','save_load':'exact'}
(HERE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
