"""Check input ablation, identical starts, and four-way reduction conventions."""
import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
def main():
    ns={'__file__':str(H/'common.py'),'__name__':'factorial_test_setup'};exec(compile((H/'common.py').read_bytes(),str(H/'common.py'),'exec'),ns);s=type('Setup',(),ns)
    r=s.load('factorial_report_fixture',H/'report.py');t=s.load('factorial_training_fixture',H/'train.py')
    base=np.array([2.,3.,7.]);loss=np.array([.2,-.1,.3]);info=np.array([-.4,.2,.1]);interaction=np.array([.05,.06,-.02])
    values=[base,base+loss,base+info,base+loss+info+interaction]
    for name,truth in [('loss_only',loss),('input_only',info),('loss_with_input',loss+interaction),('input_with_loss',info+interaction),('both_vs_control',loss+info+interaction),('interaction',interaction)]:
        np.testing.assert_allclose(r.combine(values,r.CONTRASTS[name]),truth,rtol=0,atol=2e-15)
    assert not t.trainable('base.base.shape.layers.0.net.0.original.weight',True)
    assert not t.trainable('base.base.photometry.layers.0.net.0.extra_weight',False)
    assert t.trainable('base.base.photometry.layers.0.net.0.extra_weight',True)
    assert t.trainable('base.base.photometry.layers.0.net.0.original.weight',False)
    c=s.manifest();assert list(c['arm_definition'])==c['arms']==['control','weighted','neighbour','both']
    assert [(v['neighbour_input'],v['probability_loss']) for v in c['arm_definition'].values()]==[(False,False),(False,True),(True,False),(True,True)]
    # Four-arm probability reducer agrees with the known unequal-case fixture.
    import tempfile
    p=s.load('factorial_probability_fixture',H/'probability_report.py')
    with tempfile.TemporaryDirectory() as folder:
        p.H=p.ROOT=Path(folder);sizes=np.array([10.,20.,40.]);raw=np.zeros((4,4,3,2,9,10));factor=np.zeros((4,3,2,3))
        for a,q in enumerate([.5,.6,.7,.75]):
            score=.6*(q-1)**2+.4*q**2
            for i,n in enumerate(sizes):raw[a,:,i]=n*np.array([1,.6,q,score,0,.06,-.12,q*.1,-q*.2,score]);factor[a,i]=n*np.array([1,2+a*.1,3])
        for split in ['train','validation']:np.savez(p.ROOT/f'own_{split}.npz',probability_sums=raw,probability_kinds=['hard','soft_0.05','soft_0.025','soft_0.1'],cuts=[f'cut{i}' for i in range(9)],cases=[1,2,3],arms=c['arms'],factor_density=factor)
        p.main();result=json.loads((p.ROOT/'probability_report.json').read_text())['splits']['validation']['scores']['hard']['pooled']['nine_cut_mean_score']
        expect=(.6*(.75-1)**2+.4*.75**2)-(.6*(.7-1)**2+.4*.7**2)-(.6*(.6-1)**2+.4*.6**2)+(.6*(.5-1)**2+.4*.5**2)
        np.testing.assert_allclose(result['interaction']['value'],expect,rtol=0,atol=1e-15)
    (H/'factorial_verification.json').write_text(json.dumps(dict(four_way_contrasts_exact=True,input_off_weights_frozen=True,shape_parameters_frozen=True,unequal_case_reducer_interaction_exact=True),indent=2)+'\n')
    print('FACTORIAL_CHECKS_PASSED',flush=True)
if __name__=='__main__':main()
