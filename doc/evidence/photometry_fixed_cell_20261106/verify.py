"""Exercise the actual pure numerical helpers without loading GPU runtimes."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
def extract(path,names,namespace):
    tree=ast.parse(path.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

ns={'np':np}
extract(HERE.parent/'photometry_response_bins_20261106/evaluate.py',{'RADIUS','MAG','NR','NM','histogram'},ns)
ns['b']=SimpleNamespace(RADIUS=ns['RADIUS'],MAG=ns['MAG'])
extract(HERE/'evaluate.py',{'R_CENTRE','M_CENTRE','cells','centre_draws'},ns)
v=np.zeros((1,2,4));v[...,2]=[[3.02,3.08]];m=np.array([[24.52,24.72]]);v[...,3]=10.**(.4*(30-m))
props=np.zeros((1,2,5));props[...,0]=np.log(v[...,2]);props[...,1]=m
original=v.copy();centred=ns['centre_draws'](v,props)
np.testing.assert_array_equal(v,original)
np.testing.assert_array_equal(centred[:,0],centred[:,1])
np.testing.assert_allclose(centred[...,2],np.sqrt(3*3.1),rtol=0,atol=1e-12)
props[...,2]=[[2.,6.]];weight=np.array([[1.,.5]]);response=np.array([[3.,7.]])
h=ns['histogram'](np.array([2]),weight,response,props)
np.testing.assert_allclose(h.sum((0,1,2)),[.75,3.25,2.5],rtol=0,atol=1e-12)
extract(HERE/'selection.py',{'reduce_rows'},ns)
mom=ns['reduce_rows'](np.array([0,1]),np.array([.25,.75]),np.array([.5,.5]),np.array([0.,2.]),2)
total=mom.sum(0)
np.testing.assert_allclose(total[[0,1,2,3,4]],[2,1,1,1.5,1],rtol=0,atol=1e-12)
np.testing.assert_allclose(total[4]/total[2]-total[3]/total[1],-.5,rtol=0,atol=1e-12)
for p in HERE.glob('*.py'):ast.parse(p.read_text())
(HERE/'verification.json').write_text(json.dumps({'passed':True,'checks':['Original arrays preserved',
    'Different sub-bin photometry becomes exactly identical','Representative remains in original bin',
    'Paired selected mass and both coefficient moments conserved','Known selection-weighted reference coefficient contrast',
    'All Python sources parse']},indent=2)+'\n')
print('Fixed-photometry numerical fixtures PASSED')
