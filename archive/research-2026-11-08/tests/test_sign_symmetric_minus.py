import itertools
import json

import numpy as np
import pandas as pd
import pytest

from _script_loader import load_script_module

study = load_script_module('run_sign_symmetric_minus.py')
pair = load_script_module('paired_sign_symmetric_estimate.py')
adapter = load_script_module('prepare_constgold_inference_input.py')


def test_minus_only_and_shared_objects_have_equal_inclusion_probability():
    counts = np.zeros(5)
    trials = 0
    for plus_sample in itertools.combinations(range(4), 2):
        for uniform in [.25, .75]:
            take, shared = study.coupled_selection(np.arange(5), np.arange(4), plus_sample,
                                                   [0,0,0,0,uniform], .5)
            np.testing.assert_array_equal(take[:4],np.isin(np.arange(4),plus_sample))
            np.testing.assert_array_equal(shared,[True,True,True,True,False])
            counts += take
            trials += 1
    np.testing.assert_array_equal(counts/trials,np.full(5,.5))


def test_inconsistent_plus_replay_is_rejected():
    with pytest.raises(ValueError,match='outside'):
        study.coupled_selection([1,2],[1],[2],[.2,.4],.5)


def test_observation_frame_retains_shapes_and_encodes_disk_gap():
    frame=pd.DataFrame(dict(measured_ngmix_g1=[.3,0], measured_ngmix_g2=[.4,0]))
    columns=[*frame.columns,'measured_shape_log_disk_gap']
    observed=study.observation_frame(frame,columns)
    pd.testing.assert_frame_equal(observed[frame.columns],frame)
    np.testing.assert_array_equal(observed.measured_shape_log_disk_gap,np.log1p(-np.array([.25,0])))
    frame.loc[0,'measured_ngmix_g1']=1.
    with pytest.raises(ValueError,match='physical disk'):
        study.observation_frame(frame,columns)


def test_paired_parameters_separate_slope_offset_and_cross_response():
    m,c1,c2,cross=.03,.0004,-.0002,.005
    g=[(1+m)*.02+c1,cross*.02+c2,-(1+m)*.02+c1,-cross*.02+c2]
    np.testing.assert_allclose(pair.parameters(g),[m,c1,c2,cross],atol=1e-14)


def test_shared_normalizer_cancels_in_difference_when_information_matches():
    response=np.concatenate([-np.eye(2),-np.eye(2)])
    mapped=pair.parameter_map()@response
    np.testing.assert_array_equal(mapped[[0,3]],np.zeros((2,2)))
    np.testing.assert_array_equal(mapped[[1,2]],-np.eye(2))


def test_paired_reduction_preserves_shared_case_covariance(tmp_path):
    plus, minus, norm = [tmp_path/name for name in ['plus','minus','norm']]
    norm.mkdir()
    (norm/'study.py').write_text('import numpy as np\ndef load_selection(root):\n    return np.ones((20,5))\n')
    cases = np.arange(40,140)
    noise = np.column_stack([np.linspace(-.01,.01,100), np.linspace(-.02,.02,100)])
    expected = [.02,.0004,-.0002,.005]
    for folder, sign in [(plus,1), (minus,-1)]:
        (folder/'reference').mkdir(parents=True)
        (folder/'c2').mkdir()
        g = np.array([sign*.02*(1+expected[0])+expected[1], sign*.02*expected[3]+expected[2]])
        pd.DataFrame(dict(source_case=cases,source_input_index=np.zeros(100,dtype=int),
                          injected_g1=np.full(100,sign*.02),injected_g2=np.zeros(100))).to_parquet(folder/'reference/truth.parquet')
        np.savez(folder/'c2/pergal.npz',obs=np.arange(100),case=cases,v=g+noise,
                 Ii=np.tile(np.eye(2),(100,1,1)),e=np.zeros((100,2)))
        (folder/'c2/estimate.json').write_text(json.dumps(dict(left_out=[],g=g.tolist(),
            stuck_rhat_log_z_ge_1p1=0,c2_stuck_rhat_log_z_ge_1p1=0)))
    (minus/'rule.json').write_text(json.dumps(dict(plus_root=str(plus),normalization=str(norm),bootstrap_seed=17)))
    pair.run(minus)
    result=json.loads((minus/'paired_estimate.json').read_text())
    np.testing.assert_allclose(result['values'],expected,atol=1e-14)
    assert result['matched_selected']==100
    assert max(np.array(result['quoted_errors'])[[0,3]]) < 1e-8
    assert min(np.array(result['quoted_errors'])[[1,2]]) > 0


def test_minus_adapter_reads_actual_negative_shear_and_independent_measurements(tmp_path):
    base=tmp_path/'case40_-0.02/real0/catalogues'
    for name in ['input','CrossMatch','Shapes']:
        (base/name).mkdir(parents=True)
    truth=base/'input/gals_info_tile180.0_-0.5.feather'
    pd.DataFrame(dict(index_input=[1,2],r_input=[24.,24.],Re_input=[.8,.8],
                      gamma1_input=[-.02,-.02],gamma2_input=[0.,0.])).to_feather(truth)
    pd.DataFrame(dict(id_input=[1,2],id_detec=[11,12])).to_feather(base/'CrossMatch/tile180.0_-0.5_rot0_matched.feather')
    pd.DataFrame(dict(NUMBER=[11,12],NGMIX_G1=[-.1,-.2],NGMIX_G2=[.03,.04],
        MAG_AUTO=[24.,25.9],FLUX_RADIUS=[3.5,4.])).to_feather(base/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather')
    args=dict(pixel_size=.2,zero_point=30.,primary_mag=None,primary_re=None,
              measured_radius_min=.6,strict_radius=True,physical_disk_usability=True,shear=-.02)
    frame,_=adapter.read_case(tmp_path,40,**args)
    assert frame.source_input_index.tolist()==[1]
    assert frame.measured_ngmix_g1.tolist()==[-.1]
    changed=pd.read_feather(truth);changed.gamma1_input=.02;changed.to_feather(truth)
    with pytest.raises(ValueError,match='declared'):
        adapter.read_case(tmp_path,40,**args)
