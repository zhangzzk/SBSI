"""Known responses under unequal per-leg counts, and paired-bootstrap algebra."""
import importlib.util
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('constant_report',HERE/'report.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
n=4
mass=np.array([[101,105],[55,61],[88,84],[40,35]],dtype=float)
sign=np.array([1,-1]);flow=sign*.02*.7;emu=sign*.02*.1
model=np.zeros((n,2,5));model[:,:,0]=mass;model[:,:,1]=mass*flow;model[:,:,3]=mass*emu
sim=np.zeros((n,2,3));sim[:,:,0]=mass;sim[:,:,1]=mass*(sign*.02*.824)
w=np.vstack([np.ones(n),np.random.default_rng(7).multinomial(n,np.full(n,1/n),100)])
v=r.estimates(sim,model,w)
np.testing.assert_allclose(v['m_percent'],3,atol=1e-12,rtol=0)
np.testing.assert_allclose(v['delta_response'][:,0],.024,atol=1e-12,rtol=0)
np.testing.assert_allclose(v['model_flow_response'][:,0],.7,atol=1e-12,rtol=0)
np.testing.assert_allclose(v['model_emulator_response'][:,0],.1,atol=1e-12,rtol=0)
np.testing.assert_array_equal(v['selected_count_ratio'],np.ones((101,2)))
np.testing.assert_array_equal(v['m_percent']-v['m_percent'],np.zeros(101))
receipt={'unequal_leg_count_response':'passed','known_m_percent':3,'known_delta_R11':.024,'additive_components':'passed','paired_identical_difference':'exact zero'}
(HERE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
