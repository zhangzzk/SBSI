from pathlib import Path
import importlib.util,json
import numpy as np
import torch
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('decomposition_test',H/'decompose.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main():
 b=torch.tensor([[[0.1,0.2,4.,110.],[.2,.1,2.,110.],[.1,-.1,4.,90.],[.3,.2,2.,90.],[.1,.2,1.,50.]]])
 a=torch.tensor([[[0.2,0.3,4.,110.],[.4,.2,4.,110.],[.2,-.2,4.,110.],[.5,.3,4.,110.],[.2,.3,1.,50.]]])
 eb=b[...,:2]*.01;ea=a[...,:2]*.02
 event,stats=m.per_draw(b,a,eb,ea,3.,100.)
 np.testing.assert_array_equal(event,[[0,1,2,3,4]])
 np.testing.assert_allclose((stats[...,10:14]+stats[...,14:18]).numpy(),(stats[...,6:10]-stats[...,2:6]).numpy(),atol=1e-15)
 ev2,rev=m.per_draw(a,b,ea,eb,3.,100.)
 np.testing.assert_allclose(rev[...,10:].numpy(),-stats[...,10:].numpy(),atol=1e-15)
 raw=np.zeros((2,1,2,5,18));raw[0,0,0]=stats.numpy()[0]*np.arange(1,6)[:,None];raw[0,0,1]=stats.numpy()[0]*2;raw[1]=raw[0]*.7
 terms,direct=m.terms(raw)
 np.testing.assert_allclose(terms.sum(-2),direct,atol=1e-14)
 # Large term cancellation and distinct denominators must be reconstructed.
 np.testing.assert_allclose(terms.sum((-4,-3,-2)),direct.sum((-3,-2)),atol=1e-14)
 # A zero change returns zero in every term.
 _,same=m.per_draw(b,b,eb,eb,3.,100.);z=np.zeros_like(raw);z[:]=same.numpy()[0]
 t,d=m.terms(z);np.testing.assert_array_equal(t,0)
 result={'passed':True,'event_assignment':True,'per_draw_identity':True,'arm_swap_antisymmetry':True,'unequal_normalization':True,'no_change_zero':True}
 (H/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('VERIFIED',result,flush=True)
if __name__=='__main__':main()
