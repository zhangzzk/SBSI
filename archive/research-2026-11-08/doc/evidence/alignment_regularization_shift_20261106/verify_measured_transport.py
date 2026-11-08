"""Known four-state transport fixture checks inversion and edge orientation."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('anchor',HERE/'evaluate_measured_transport.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(2026110617)
base=torch.as_tensor(rng.uniform(-.4,.4,(31,2)),dtype=torch.float64)
b=torch.as_tensor(rng.uniform(-.2,.8,(31,2)),dtype=torch.float64)
states=[]
for signs in m.SIGNS:
    v=.02*(b*torch.as_tensor(signs)).sum(-1)
    states.append(m.disk_response_transport(base,torch.stack([v,v*0],-1))[0])
for a,z in m.EDGES:
    torch.testing.assert_close(m.transport_edge(states[a],b,a,z),states[z],atol=2e-12,rtol=0)
    torch.testing.assert_close(m.transport_edge(states[z],b,z,a),states[a],atol=2e-12,rtol=0)
    torch.testing.assert_close(m.transport_edge(base,b*0,a,z),base,atol=2e-12,rtol=0)
(HERE/'measured_transport_verification.json').write_text(json.dumps({
    'passed':True,'definition':'Synthetic implementation check, not scientific evidence',
    'checks':['Known fixed coefficients reproduce both directions of all four observed edges',
              'Zero neighbour coefficients imply zero change','Forward-inverse roundtrip']},indent=2)+'\n')
print('Measured transport analytical fixture PASSED')
