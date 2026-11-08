import numpy as np
import pytest

from sbsi.flow_grid_case_validation import (
    paired_case_response_statistics, case_cross_from_global_contributions,
    summarize_paired_grid_predictions,
)


def simulated_rows(outputs=2):
    rng = np.random.default_rng(532)
    case = np.repeat(np.arange(12),13)
    cell = rng.integers(0,2,len(case))
    gamma = rng.normal(size=(len(case),2))
    # Case-dependent shears make refitting each deleted design necessary.
    gamma[:,0] *= 1+case/5
    measured = rng.normal(size=(len(case),outputs))
    predicted = measured*.8+rng.normal(size=(len(case),outputs))*.2
    return case,cell,gamma,measured,predicted


def statistics(case,cell,gamma,measured,predicted):
    normal = np.zeros((12,2,2,2))
    cross = [np.zeros((12,2,measured.shape[1],2)) for _ in range(2)]
    for c,b,g,m,p in zip(case,cell,gamma,measured,predicted):
        normal[c,b] += np.outer(g,g)
        cross[0][c,b] += np.outer(m,g)
        cross[1][c,b] += np.outer(p,g)
    return normal,*cross


@pytest.mark.parametrize("outputs",[2,4])
def test_paired_jackknife_matches_direct_refits_for_shape_and_four_outputs(outputs):
    case,cell,gamma,measured,predicted = simulated_rows(outputs)
    result = paired_case_response_statistics(*statistics(case,cell,gamma,measured,predicted))
    refits = []
    for deleted in range(12):
        responses = []
        for b in range(2):
            keep = (case != deleted)&(cell == b)
            g = gamma[keep]
            residual = predicted[keep]-measured[keep]
            responses.append((residual.T@g)@np.linalg.inv(g.T@g))
        refits.append(np.asarray(responses).reshape(2,-1))
    refits = np.asarray(refits)
    np.testing.assert_allclose(result['residual_jackknife'],refits,atol=1e-14)
    centered = refits-refits.mean(0)
    expected = 11/12*np.einsum('cbi,cbj->bij',centered,centered)
    np.testing.assert_allclose(result['residual_covariance'],expected,atol=1e-14)
    np.testing.assert_allclose(result['residual'],result['model_response']-result['measured_response'],atol=1e-14)


def test_shared_measurement_variation_cancels_in_identical_model_residual():
    case,cell,gamma,measured,_ = simulated_rows()
    result = paired_case_response_statistics(*statistics(case,cell,gamma,measured,measured))
    assert np.linalg.norm(result['measured_jackknife'].std(0)) > .01
    np.testing.assert_array_equal(result['residual'],0)
    np.testing.assert_array_equal(result['residual_covariance'],0)


def test_global_design_contributions_recover_case_raw_cross_statistics():
    case,cell,gamma,measured,predicted = simulated_rows()
    normal,_,model_cross = statistics(case,cell,gamma,measured,predicted)
    counts = np.bincount(cell,minlength=2)
    h = normal.sum(0)/counts[:,None,None]
    sums = np.zeros((12,2,4))
    for c,b,g,p in zip(case,cell,gamma,predicted):
        sums[c,b] += np.outer(p,g@np.linalg.inv(h[b])).reshape(4)
    recovered = case_cross_from_global_contributions(sums,normal,counts)
    np.testing.assert_allclose(recovered,model_cross,atol=1e-13)


def test_invalid_or_degenerate_design_rejected():
    args = statistics(*simulated_rows())
    bad = args[0].copy()
    bad[...,1,1] = 0
    with pytest.raises(ValueError,match='unstable'):
        paired_case_response_statistics(bad,*args[1:])
    with pytest.raises(ValueError,match='eight cases'):
        paired_case_response_statistics(*(a[:7] for a in args))


def test_complete_prediction_summary_matches_direct_cross_scores_and_integration_covariance():
    case,cell,gamma,measured,predicted = simulated_rows()
    normal,_,_ = statistics(case,cell,gamma,measured,predicted)
    counts = np.bincount(cell,minlength=2)
    design = np.einsum('ni,nij->nj',gamma,np.linalg.inv(normal.sum(0)/counts[:,None,None])[cell])
    rng = np.random.default_rng(551)
    groups = [((predicted+rng.normal(size=predicted.shape)*.3)[:,:,None]*design[:,None,:]).reshape(-1,4)
              for _ in range(4)]
    precision = np.array([np.diag([1,2,3,4]),np.diag([5,6,7,8])])
    result = summarize_paired_grid_predictions(gamma,cell,case,measured,groups,precision)
    means = np.stack([np.stack([g[cell==b].mean(0) for b in range(2)]) for g in groups])
    np.testing.assert_allclose(result['group_model_responses'].reshape(4,2,4),means,atol=1e-14)
    target = result['measured_response'].reshape(2,4)
    scores = [.5*np.einsum('bi,bij,bj->b',means[a]-target,precision,means[c]-target)/4
              for a in range(4) for c in range(4) if a != c]
    np.testing.assert_allclose(result['cross_score_per_cell'],np.mean(scores,axis=0),atol=1e-14)
    for b in range(2):
        np.testing.assert_allclose(result['integration_covariance_of_mean'][b],np.cov(means[:,b].T)/4,atol=1e-14)
    assert result['normalized_cross_score'] == pytest.approx(np.mean(scores))


def test_complete_prediction_summary_rejects_missing_rows():
    case,cell,gamma,measured,_ = simulated_rows()
    with pytest.raises(ValueError,match='every matched row'):
        summarize_paired_grid_predictions(gamma,cell,case,measured,
                                          [np.zeros((len(case)-1,4))]*2,np.tile(np.eye(4),(2,1,1)))
