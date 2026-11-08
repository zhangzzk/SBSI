import numpy as np
import pytest

from scripts.audit_flow_grid_sampling import summarize_groups


def test_group_decomposition_and_localization_of_one_extreme_draw():
    counts=np.array([4,6])
    cells=np.repeat(np.arange(2),counts)
    target=np.zeros((2,4))
    precision=np.tile(np.eye(4),(2,1,1))
    a=np.zeros((10,4)); c=np.zeros_like(a)
    a[7,0]=120.; c[7,0]=-60.
    d=summarize_groups([a,c],cells,counts,target,precision)
    assert d['top_rows'][0]==7
    assert d['concentration']['1']==pytest.approx(1.)
    assert d['cross'][1]<0
    assert d['average'][1]>0
    np.testing.assert_allclose(d['average'],d['cross']+d['disagreement']/4)
    assert d['cross'][0]==0
    np.testing.assert_allclose(d['mean_a'][1],[20.,0,0,0])


def test_full_population_count_mismatch_is_rejected():
    with pytest.raises(ValueError,match='full-population'):
        summarize_groups([np.zeros((10,4))]*2,np.zeros(10,dtype=int),np.array([11]),
                         np.zeros((1,4)),np.eye(4)[None])
