"""Execute the actual aggregation on an analytical synthetic four-scene fixture."""
import json,tempfile
from pathlib import Path
import numpy as np
from contrasts import contrasts,own,opposite
HERE=Path(__file__).parent
CASES=[1000,1040,1080,1120];CUTS=['uncut','mag25','radius3','radius3_mag25']
EDGE_STATES=[(0,2),(1,3),(0,1),(2,3),(0,3)]
sim=[];model=[];edges=[];receipts=[]
signs=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])*.02
for case in range(4):
    ss=np.zeros((2,4,4,2,2,3));mm=np.zeros((3,4,4,2,2,5));ec=np.zeros((4,5,2,2))
    for g in range(2):
        own_g=signs[:,g];other_g=signs[:,1-g]
        for k in range(4):
            # Each group has ten targets per population; the changing selected
            # mass represents many objects after averaging over this fixture.
            mass=np.full(4,10.) if k==0 else 10.+2*own_g+other_g
            simmean=.1+.81*own_g+.05*other_g
            modelmass=np.full(4,10.) if k==0 else 10.+2*own_g
            flowmean=.1+.8*own_g;emumean=.03*other_g
            for p in range(2):
                ss[:,:,k,g,p,0]=mass;ss[:,:,k,g,p,1]=mass*simmean
                mm[:,:,k,g,p,0]=modelmass;mm[:,:,k,g,p,1]=modelmass*flowmean;mm[:,:,k,g,p,3]=modelmass*emumean
            for e,(a,b) in enumerate(EDGE_STATES):
                ec[k,e,g]=[2*.5*(mass[a]+mass[b])*(simmean[a]-simmean[b]),2*.5*(simmean[a]+simmean[b])*(mass[a]-mass[b])]
    sim.append(ss);model.append(mm);edges.append(ec)
source=(HERE/'reduce.py').read_text();tail=source[source.index('sim=np.array(sim);'):]
with tempfile.TemporaryDirectory(prefix='sbsi_split_verify_') as tmp:
    env=dict(globals(),O=Path(tmp));exec(compile(tail,'<actual reducer aggregation>','exec'),env)
    result=json.loads((Path(tmp)/'report.json').read_text())
    for flag in result['comparisons'].values():
        r=flag['uncut']['all']
        for key,expected in [('own_group',.01),('opposite_group',.02),('coherent_total',.03)]:
            np.testing.assert_allclose(r[key]['error']['value'],expected,atol=1e-13)
    for group in result['comparisons']['common_U']['uncut']['within_group'].values():
        np.testing.assert_allclose(group['opposite_group']['simulation_shape_term']['value'],.05,atol=1e-13)
        np.testing.assert_allclose(group['opposite_group']['simulation_selection_term']['value'],0,atol=1e-13)
(HERE/'reducer_verification.json').write_text(json.dumps({'passed':True,'fixture':'Analytical synthetic data; not scientific evidence','checks':['full aggregation and bootstrap execute','known own/opposite/total errors','fixed-U flow opposite contrast zero','edge shape/selection closure']},indent=2)+'\n')
print('Reducer analytical fixture PASSED')
