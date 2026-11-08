"""Algebra and finite-draw rank calibration prerequisites."""
import types
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
c=types.ModuleType('photo_verify_common');c.__file__=str(H/'common.py')
exec(compile((H/'common.py').read_bytes(),c.__file__,'exec'),c.__dict__)
def main():
    c.verify();r=c.load('photo_verify_report',H/'report.py');rng=np.random.default_rng(202611082210)
    p=rng.uniform(.2,.8,(13,9,3));o=rng.uniform(.2,.8,(13,9,3))
    p[:,:,2]=p[:,:,0]*p[:,:,1];o[:,:,2]=o[:,:,0]*o[:,:,1]
    d=r.decompose(p,o);np.testing.assert_allclose(d[:,:,-1],0,atol=1e-15)
    p[:,:,2]+=.01;d=r.decompose(p,o);np.testing.assert_allclose(d[:,:,-1],.01,atol=1e-15)
    np.testing.assert_allclose(r.decompose(o,p),-d,atol=1e-15)
    # Unequal selection masses and arbitrary scene partition must close exactly.
    B=rng.uniform(0,2,1001);a=rng.integers(0,2,1001);b=rng.random(1001);cell=np.arange(1001)%17
    ma=a.sum();mb=b.sum();ba=(B*a).sum()/ma;bb=(B*b).sum()/mb
    pieces=[((B[cell==k]-.5*(ba+bb))*(a[cell==k]-b[cell==k])).sum()/(.5*(ma+mb)) for k in range(17)]
    np.testing.assert_allclose(sum(pieces),ba-bb,rtol=1e-12,atol=1e-14)
    # Holding out every member of an exchangeable continuous 65-member sample
    # gives each rank 0,...,64 exactly once, including 1/65 in each extreme bin.
    v=rng.normal(size=65);ranks=np.array([(np.delete(v,i)<v[i]).sum() for i in range(65)])
    np.testing.assert_array_equal(np.sort(ranks),np.arange(65));assert np.mean(ranks==0)==1/65
    c.write(H/'verification.json',dict(marginal_association_identity=True,swap_antisymmetry=True,
           unequal_mass_scene_contribution_identity=True,iid_rank_null_65=True))
    print('DISTRIBUTION_FIXTURES_PASSED',flush=True)
if __name__=='__main__':main()
