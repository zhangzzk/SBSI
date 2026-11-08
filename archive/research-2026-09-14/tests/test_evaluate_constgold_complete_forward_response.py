import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_constgold_complete_forward_response import (
    _experiment_comparison,
    _flow_seed_sem,
    _measured_leg,
    _prediction_run_metadata,
    _prediction_run_diagnostics,
    _qmc_confidence_multiplier,
    _raw_case_response_sufficient,
    _with_circularized_condition,
    _with_flow_conditions,
    _with_truth_crowding,
    parse_args,
)


def test_qmc_confidence_multiplier_uses_student_t_for_four_seeds():
    assert _qmc_confidence_multiplier(4) == pytest.approx(3.182446305284263)
    assert _qmc_confidence_multiplier(4) > 1.96
    assert _qmc_confidence_multiplier(1) is None


def test_flow_seed_sem_is_null_for_single_checkpoint():
    assert _flow_seed_sem(np.array([[0.9, 0.0]])) is None
    assert np.allclose(
        _flow_seed_sem(np.array([[0.8, -0.1], [1.0, 0.1]])),
        [0.1, 0.1],
    )


def test_prediction_run_diagnostics_separates_model_and_qmc_variation():
    # Canonical ordering is model-major, then QMC seed.
    responses = np.array(
        [[0.8, 0.0], [1.0, 0.2], [1.2, -0.2], [1.4, 0.0]]
    )
    result = _prediction_run_diagnostics(
        responses, model_count=2, sampling_seed_count=2
    )
    np.testing.assert_allclose(result["response_by_flow_seed"], [[0.9, 0.1], [1.3, -0.1]])
    np.testing.assert_allclose(result["response_by_sampling_seed"], [[1.0, -0.1], [1.2, 0.1]])
    np.testing.assert_allclose(result["flow_seed_sem"], [0.2, 0.1])
    np.testing.assert_allclose(result["sampling_seed_sem"], [0.1, 0.1])
    assert len(result["response_by_prediction_run"]) == 4


def test_response_cli_accepts_repeated_classifier_and_qmc_seeds():
    args = parse_args(
        [
            "--constgold-root", "root",
            "--rblend-lookup", "rblend.feather",
            "--measurement-model", "flow.pt",
            "--detection-classifier", "detector1.pt",
            "--detection-classifier", "detector2.pt",
            "--sampling-seed", "7301",
            "--sampling-seed", "7302",
            "--shear-component", "g2",
            "--probability-factorization", "direct_selected",
            "--case", "1000",
            "--output", "result.json",
        ]
    )
    assert [path.name for path in args.detection_classifier] == [
        "detector1.pt", "detector2.pt"
    ]
    assert args.sampling_seed == [7301, 7302]
    assert args.shear_component == "g2"
    assert args.probability_factorization == "direct_selected"
    assert args.crowd_lookup is None


def test_truth_crowding_removes_the_required_external_lookup():
    from types import SimpleNamespace

    expected = pd.DataFrame(
        {
            "input_index": [11, 12],
            "nbr_flux_near": [0.1, 0.2],
            "nbr_flux_far": [0.3, 0.4],
            "nbr_flux_max": [0.5, 0.6],
        }
    )
    sheared = SimpleNamespace(object_view=lambda **kwargs: expected)
    shear_calls = []

    def shear(g1, g2):
        shear_calls.append((g1, g2))
        return sheared

    prior = SimpleNamespace(shear=shear)
    support = pd.DataFrame(
        {"input_index": [11, 12], "R_blend": [0.7, 0.8]}
    )

    result = _with_truth_crowding(
        support, prior=prior, active=np.array([0, 1], dtype=np.int64)
    )

    assert result["R_blend"].tolist() == [0.7, 0.8]
    np.testing.assert_allclose(
        result[["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]],
        expected[["nbr_flux_near", "nbr_flux_far", "nbr_flux_max"]],
    )
    assert shear_calls == [(0.0, 0.0)]


def test_response_circularized_size_uses_aligned_preshear_truth():
    baseline = pd.DataFrame(
        {
            "index": [11, 12],
            "axis_ratio": [0.25, 0.81],
            "position_angle": [0.2, 1.1],
            "Re": [0.8, 1.2],
        }
    )
    plus = pd.DataFrame({"input_index": [11, 12], "e1_input_p": [0.4, -0.2]})
    minus = pd.DataFrame({"input_index": [11, 12], "e1_input_p": [-0.3, 0.1]})

    plus_result = _with_circularized_condition(plus, baseline)
    minus_result = _with_circularized_condition(minus, baseline)
    expected = baseline["Re"].to_numpy() * np.sqrt(
        baseline["axis_ratio"].to_numpy()
    )

    np.testing.assert_allclose(plus_result["circularized_Re_input_p"], expected)
    np.testing.assert_array_equal(
        plus_result["circularized_Re_input_p"],
        minus_result["circularized_Re_input_p"],
    )
    assert not np.array_equal(plus_result["e1_input_p"], minus_result["e1_input_p"])


def test_response_attaches_rblend_when_the_flow_declares_it():
    from types import SimpleNamespace

    frame = pd.DataFrame({"e1_input_p": [0.1, -0.2]})
    bundle = SimpleNamespace(
        condition_preprocessor=SimpleNamespace(
            feature_names=("e1_input_p", "R_blend")
        )
    )

    result = _with_flow_conditions(
        frame, [bundle], rblend=np.array([0.3, 0.7])
    )

    assert "R_blend" not in frame
    np.testing.assert_array_equal(result["R_blend"], [0.3, 0.7])


def test_response_rejects_an_unavailable_flow_condition():
    from types import SimpleNamespace

    bundle = SimpleNamespace(
        condition_preprocessor=SimpleNamespace(
            feature_names=("e1_input_p", "not_available")
        )
    )
    with pytest.raises(KeyError, match="not_available"):
        _with_flow_conditions(
            pd.DataFrame({"e1_input_p": [0.1]}),
            [bundle],
            rblend=np.array([0.3]),
        )


def test_raw_per_model_sufficient_statistics_are_json_serializable():
    import json
    from scripts.evaluate_constgold_complete_forward_response import _serialize_sufficient
    raw = {"predicted": [{"1": {"denominator": 2., "numerator": np.array([.1, .2])}}]}
    restored = json.loads(json.dumps(_serialize_sufficient(raw), allow_nan=False))
    assert restored == {"predicted": [{"1": {"denominator": 2., "numerator": [.1, .2]}}]}


def test_raw_case_sufficient_keeps_every_model_by_qmc_run():
    measured = {
        1: {1000: {"count": 2, "sum": [0.1, 0.0]}},
        -1: {1000: {"count": 2, "sum": [-0.1, 0.0]}},
    }
    predicted = []
    for value in (0.08, 0.09, 0.10):
        predicted.append(
            {
                1: {
                    1000: {
                        "complete": {"denominator": 2.0, "numerator": [value, 0.0]},
                        "matched_detected_selected": {
                            "denominator": 2.0,
                            "numerator": [value, 0.0],
                        },
                    }
                },
                -1: {
                    1000: {
                        "complete": {"denominator": 2.0, "numerator": [-value, 0.0]},
                        "matched_detected_selected": {
                            "denominator": 2.0,
                            "numerator": [-value, 0.0],
                        },
                    }
                },
            }
        )

    response, matched = _raw_case_response_sufficient(
        case=1000,
        measured_matched_selected=measured,
        measured_matched_detected=measured,
        measured_matched_detected_selected=measured,
        measured_complete=measured,
        measured_detected=measured,
        truth_actual_detected=measured,
        truth_actual_selected=measured,
        predicted=predicted,
    )

    assert len(response["predicted"]) == 3
    assert len(matched["predicted"]) == 3
    assert set(response["measured"]) == {
        "matched_selected",
        "matched_detected",
        "matched_detected_selected",
        "complete",
        "detection_only",
    }
    assert set(response["truth_actual"]) == {"detection", "detection_and_selection"}
    assert [run["1"]["complete"]["numerator"][0] for run in response["predicted"]] == [
        0.08,
        0.09,
        0.10,
    ]


def test_prediction_run_metadata_uses_bundle_path_seed_order(tmp_path):
    from types import SimpleNamespace

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen model")
    bundle = SimpleNamespace(metadata={"seed": 501})

    result = _prediction_run_metadata([(bundle, checkpoint, 7301)])

    assert result == [
        {
            "path": str(checkpoint),
            "sha256": "e53d989a57dfed763fd28993b27f15ef304bc2ed1d384f2fff91afbe889c5fab",
            "training_seed": 501,
            "sampling_seed": 7301,
        }
    ]


def test_matched_detection_selection_cuts_draws_without_classifier_weights():
    from types import SimpleNamespace
    from scripts.evaluate_constgold_complete_forward_response import _prediction_sufficient, TARGET_NAMES
    values=np.zeros((3,4,4),float)
    values[:,:,2]=20.
    values[:,:,3]=np.log(2.)
    values[0,:,0]=[.1,.3,.4,.5]
    values[1,:,0]=[-.2,.2,.3,.4]
    values[2,:,0]=.55
    values[0,:2,3]=np.log(4.)
    values[1,0,3]=np.log(4.)
    values[2,:,3]=np.log(4.)
    bundle=SimpleNamespace(target_transform=SimpleNamespace(target_names=TARGET_NAMES),
                           sample=lambda *args,**kwargs:values.copy())
    kwargs=dict(blend_shift=np.zeros((3,2)),draws=4,seed=1,batch_size=3,abs_shape_max=.6,
                fixed_population_masks={'matched_detected':np.array([True,True,False]),
                                        'matched_detected_selected':np.array([True,True,False])})
    first,_=_prediction_sufficient(bundle,pd.DataFrame({'id':[1,2,3]}),
                                  detection_probability=np.array([.01,.99,.2]),**kwargs)
    second,_=_prediction_sufficient(bundle,pd.DataFrame({'id':[1,2,3]}),
                                   detection_probability=np.array([.9,.1,.8]),**kwargs)
    for result in [first,second]:
        selected=result['matched_detected_selected']
        assert np.isclose(selected['denominator'],3/4)
        assert np.allclose(selected['numerator'],[(.1+.3-.2)/4,0])
        assert result['matched_detected']['denominator']==2
        assert np.isclose(result['matched_detected']['numerator'][0],.5)
    assert first['complete']['denominator'] != second['complete']['denominator']


def test_both_detected_parent_keeps_leg_specific_measured_selection():
    from scripts.evaluate_constgold_complete_forward_response import _restrict_measured_population
    parent=np.array([1,2])
    plus=_restrict_measured_population({'ids':np.array([1,3]),'values':np.array([[.1,0],[.5,0]])},parent)
    minus=_restrict_measured_population({'ids':np.array([2,3]),'values':np.array([[-.2,0],[.5,0]])},parent)
    assert plus['count']==minus['count']==1
    assert np.allclose(plus['sum'],[.1,0])
    assert np.allclose(minus['sum'],[-.2,0])


def test_matched_selection_bootstrap_pools_leg_denominators_before_ratio():
    from scripts.evaluate_constgold_complete_forward_response import _matched_detection_selection_comparison
    measured={1:{0:{'count':2,'sum':[.04,0]},1:{'count':8,'sum':[.08,0]}},
              -1:{0:{'count':4,'sum':[-.04,0]},1:{'count':2,'sum':[-.02,0]}}}
    # Model has different selected weights per leg, but exactly 0.8 times
    # the measured mean for each leg/case. Preserve this after pooling by
    # using a leg-specific common weight multiplier across cases.
    pred={1:{},-1:{}}
    for d,multiplier in [(1,3.),(-1,2.)]:
        for c in [0,1]:
            item=measured[d][c]
            pred[d][c]={'matched_detected_selected':{'denominator':multiplier*item['count'],
                      'numerator':(multiplier*.8*np.array(item['sum'])).tolist()}}
    result=_matched_detection_selection_comparison(cases=(0,1),measured=measured,predicted=[pred],
                                                   h=.02,replicates=100,seed=3)
    assert np.isclose(result['R11_sim'],(.12/10+.06/6)/.04)
    assert np.isclose(result['R11_model'],.8*result['R11_sim'])
    assert np.isclose(result['m_percent'],25.)
    assert result['m_case_se_percent'] < 1e-12
    assert np.allclose(result['m_case_ci95_percent'],[25.,25.])


def test_matched_selection_uses_r22_for_g2_catalogues():
    from scripts.evaluate_constgold_complete_forward_response import _matched_detection_selection_comparison
    measured = {
        1: {0: {"count": 2, "sum": [0.0, 0.04]}},
        -1: {0: {"count": 2, "sum": [0.0, -0.04]}},
    }
    predicted = [{1: {}, -1: {}}]
    predicted[0][1][0] = {
        "matched_detected_selected": {"denominator": 2, "numerator": [0.0, 0.032]}
    }
    predicted[0][-1][0] = {
        "matched_detected_selected": {"denominator": 2, "numerator": [0.0, -0.032]}
    }

    result = _matched_detection_selection_comparison(
        cases=(0,), measured=measured, predicted=predicted, h=0.02,
        replicates=20, seed=3, component_index=1,
    )

    assert result["response_component"] == "R22"
    assert result["R_sim"] == 1.0
    assert result["R_model"] == 0.8
    assert result["m_percent"] == 25.0
    assert "R11_sim" not in result


def test_case_bootstrap_reports_complete_multiplicative_ratio():
    from scripts.evaluate_constgold_complete_forward_response import _case_bootstrap

    measured = {
        1: {0: {"count": 1, "sum": [0.02, 0.0]}},
        -1: {0: {"count": 1, "sum": [-0.02, 0.0]}},
    }
    predicted = [{1: {0: {}}, -1: {0: {}}}]
    for direction, numerator in ((1, [0.016, 0.0]), (-1, [-0.016, 0.0])):
        predicted[0][direction][0] = {
            branch: {"denominator": 1.0, "numerator": numerator}
            for branch in (
                "matched_selected", "matched_detected", "detection_only", "complete"
            )
        }

    result = _case_bootstrap(
        cases=(0,),
        measured_selected=measured,
        measured_detected=measured,
        measured_matched_selected=measured,
        measured_matched_detected=measured,
        predicted=predicted,
        h=0.02,
        replicates=20,
        seed=4,
    )

    summary = result["multiplicative_bias_percent"]["complete"]
    assert summary["standard_error"] == 0.0
    assert summary["ci95"] == [25.0, 25.0]


def test_vectorized_case_bootstrap_matches_literal_cluster_resampling():
    from scripts.evaluate_constgold_complete_forward_response import _case_bootstrap

    cases = (0, 1, 2)
    measured = {1: {}, -1: {}}
    predicted = [{1: {}, -1: {}}, {1: {}, -1: {}}]
    branches = (
        "matched_selected",
        "matched_detected",
        "detection_only",
        "complete",
    )
    for direction in (1, -1):
        for case in cases:
            measured[direction][case] = {
                "count": case + 2,
                "sum": [direction * (case + 1) * 0.02, direction * case * 0.001],
            }
            for model_index, model in enumerate(predicted):
                numerator = [
                    direction * (case + 1) * (0.016 + 0.001 * model_index),
                    direction * case * 0.0008,
                ]
                model[direction][case] = {
                    branch: {"denominator": case + 2.5, "numerator": numerator}
                    for branch in branches
                }

    result = _case_bootstrap(
        cases=cases,
        measured_selected=measured,
        measured_detected=measured,
        measured_matched_selected=measured,
        measured_matched_detected=measured,
        predicted=predicted,
        h=0.02,
        replicates=100,
        seed=13,
    )

    sampled = np.random.default_rng(13).integers(0, len(cases), size=(100, len(cases)))
    literal = []
    for indices in sampled:
        selected = [cases[index] for index in indices]
        measured_means = []
        model_responses = []
        for direction in (1, -1):
            count = sum(measured[direction][case]["count"] for case in selected)
            total = sum(measured[direction][case]["sum"][0] for case in selected)
            measured_means.append(total / count)
        for model in predicted:
            model_means = []
            for direction in (1, -1):
                denominator = sum(
                    model[direction][case]["complete"]["denominator"]
                    for case in selected
                )
                numerator = sum(
                    model[direction][case]["complete"]["numerator"][0]
                    for case in selected
                )
                model_means.append(numerator / denominator)
            model_responses.append((model_means[0] - model_means[1]) / 0.04)
        measured_response = (measured_means[0] - measured_means[1]) / 0.04
        literal.append(100.0 * (measured_response / np.mean(model_responses) - 1.0))
    literal = np.asarray(literal)
    summary = result["multiplicative_bias_percent"]["complete"]
    assert summary["standard_error"] == pytest.approx(literal.std(ddof=1))
    np.testing.assert_allclose(summary["ci95"], np.quantile(literal, [0.025, 0.975]))


def test_experiment_comparison_records_residual_and_bootstrap():
    measured = {"response_first_column": [0.9, -0.1], "plus_count": 12}
    predicted = {"response_first_column": [0.95, -0.08]}
    result = _experiment_comparison(
        measured=measured,
        predicted=predicted,
        bootstrap={
            "standard_error": [0.01, 0.02],
            "ci95": [[0.03, -0.02], [0.07, 0.06]],
        },
        definition="fixed population",
        model_components=("flow",),
    )

    assert result["measured"] is measured
    assert result["predicted"] is predicted
    assert result["model_components"] == ["flow"]
    assert np.allclose(
        result["predicted_minus_measured"]["response_first_column"],
        [0.05, 0.02],
    )
    assert result["predicted_minus_measured"][
        "case_bootstrap_standard_error"
    ] == [0.01, 0.02]


def test_measured_leg_reports_inner_join_exclusions(tmp_path):
    catalogue = tmp_path / "case0_0.02" / "real0" / "catalogues"
    cross_dir = catalogue / "CrossMatch"
    shape_dir = catalogue / "Shapes"
    cross_dir.mkdir(parents=True)
    shape_dir.mkdir(parents=True)
    pd.DataFrame(
        {"id_detec": [1, 2], "id_input": [10, 20]}
    ).to_feather(cross_dir / "tile180.0_-0.5_rot0_matched.feather")
    pd.DataFrame(
        {
            "NUMBER": [1, 3],
            "NGMIX_G1": [0.1, 0.2],
            "NGMIX_G2": [0.0, 0.0],
            "MAG_AUTO": [25.0, 25.0],
            "FLUX_RADIUS": [5.0, 5.0],
        }
    ).to_feather(
        shape_dir / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    )

    _, _, report, _, _ = _measured_leg(
        tmp_path,
        case=0,
        sign=0.02,
        active_ids=np.array([10]),
        abs_shape_max=0.6,
    )

    assert report["crossmatch_rows_without_shape"] == 1
    assert report["shape_rows_without_crossmatch"] == 1
    assert report["crossmatched_rows_outside_finite_prior_support"] == 0
    assert report["actual_detected_count"] == 1


def test_measured_leg_accepts_an_explicit_no_cut_control(tmp_path):
    catalogue = tmp_path / "case0_0.02" / "real0" / "catalogues"
    cross_dir = catalogue / "CrossMatch"
    shape_dir = catalogue / "Shapes"
    cross_dir.mkdir(parents=True)
    shape_dir.mkdir(parents=True)
    pd.DataFrame({"id_detec": [1], "id_input": [10]}).to_feather(
        cross_dir / "tile180.0_-0.5_rot0_matched.feather"
    )
    pd.DataFrame(
        {
            "NUMBER": [1],
            "NGMIX_G1": [0.95],
            "NGMIX_G2": [0.0],
            "MAG_AUTO": [30.0],
            "FLUX_RADIUS": [0.1],
        }
    ).to_feather(
        shape_dir / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    )

    _, selected, report, _, _ = _measured_leg(
        tmp_path,
        case=0,
        sign=0.02,
        active_ids=np.array([10]),
        abs_shape_max=None,
        mag_max=None,
        radius_min_arcsec=None,
    )

    assert selected["count"] == 1
    assert report["dropped_measured_cut"] == 0


def test_physical_and_engineered_samples_have_identical_selection_sufficient_statistics():
    from types import SimpleNamespace
    from scripts.evaluate_constgold_complete_forward_response import _prediction_sufficient,TARGET_NAMES
    engineered=np.array([[[.1,.2,24.,np.log(4.)],[.2,.1,28.,np.log(.1)]]])
    physical=np.stack((engineered[...,0],engineered[...,1],np.exp(engineered[...,3]),10**(.4*(30-engineered[...,2]))),axis=-1)
    names=('measured_ngmix_g1','measured_ngmix_g2','measured_flux_radius','measured_flux_from_mag_auto')
    kwargs=dict(detection_probability=np.array([.7]),blend_shift=np.zeros((1,2)),draws=2,seed=2,batch_size=1,abs_shape_max=.8)
    results=[]
    for labels,values in [(TARGET_NAMES,engineered),(names,physical)]:
        bundle=SimpleNamespace(target_transform=SimpleNamespace(target_names=labels),sample=lambda *args,**kw:values.copy())
        results.append(_prediction_sufficient(bundle,pd.DataFrame({'id':[1]}),**kwargs)[0])
    for key in ['complete','detection_only','selection_only','unweighted']:
        assert results[0][key]['denominator']==results[1][key]['denominator']
        np.testing.assert_allclose(results[0][key]['numerator'],results[1][key]['numerator'])


def test_direct_selected_probability_conditions_flow_shape_on_the_exact_cut():
    from types import SimpleNamespace
    from scripts.evaluate_constgold_complete_forward_response import _prediction_sufficient

    physical_names = (
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_flux_radius",
        "measured_flux_from_mag_auto",
    )
    samples = np.array(
        [
            [[0.2, 0.1, 4.0, 100.0], [0.6, 0.2, 2.0, 100.0]],
            [[-0.4, 0.3, 2.0, 100.0], [-0.2, 0.5, 2.0, 100.0]],
        ]
    )
    bundle = SimpleNamespace(
        model=type("ConditionalPhysicalDiskFlow", (), {})(),
        target_transform=SimpleNamespace(target_names=physical_names),
        sample=lambda *args, **kwargs: samples.copy(),
    )
    probability = np.array([0.8, 0.2])
    shift = np.array([[0.1, -0.05], [0.2, 0.3]])

    result, _ = _prediction_sufficient(
        bundle,
        pd.DataFrame({"id": [1, 2]}),
        detection_probability=probability,
        blend_shift=shift,
        draws=2,
        seed=1,
        batch_size=2,
        abs_shape_max=None,
        mag_max=25.8,
        radius_min_arcsec=0.75,
        truth_shape=np.array([[0.3, 0.0], [-0.1, 0.2]]),
        probability_factorization="direct_selected",
    )

    # Only object 1 has a passing radius draw.  Its selected conditional shape
    # is (0.2, 0.1) plus the deterministic shift; object 2 is conservatively
    # imputed at zero and its possible contribution is bounded, not hidden.
    assert result["complete"]["denominator"] == pytest.approx(1.0)
    np.testing.assert_allclose(result["complete"]["numerator"], [0.24, 0.04])
    np.testing.assert_allclose(result["truth_complete"]["numerator"], [0.22, 0.04])
    support = result["direct_selected_conditional_support"]
    assert support["zero_pass_row_count"] == 1
    assert support["zero_pass_event_probability_fraction"] == pytest.approx(0.2)
    np.testing.assert_allclose(support["leg_mean_abs_bound"], [0.24, 0.26])


def test_direct_selected_probability_rejects_a_non_disk_flow():
    from types import SimpleNamespace
    from scripts.evaluate_constgold_complete_forward_response import (
        TARGET_NAMES,
        _prediction_sufficient,
    )

    bundle = SimpleNamespace(
        target_transform=SimpleNamespace(target_names=TARGET_NAMES),
        sample=lambda *args, **kwargs: np.zeros((1, 2, 4)),
    )
    with pytest.raises(ValueError, match="physical disk flow"):
        _prediction_sufficient(
            bundle,
            pd.DataFrame({"id": [1]}),
            detection_probability=np.array([0.5]),
            blend_shift=np.zeros((1, 2)),
            draws=2,
            seed=1,
            batch_size=1,
            abs_shape_max=None,
            probability_factorization="direct_selected",
        )


def test_direct_selected_support_summary_pools_mass_and_bounds_each_case():
    from scripts.evaluate_constgold_complete_forward_response import (
        _direct_selected_support_summary,
    )

    predicted = []
    for multiplier in (1.0, 2.0):
        run = {1: {}, -1: {}}
        for direction in (1, -1):
            for case in (10, 11):
                run[direction][case] = {
                    "direct_selected_conditional_support": {
                        "total_event_probability_mass": 10.0,
                        "missing_numerator_abs_bound": np.array(
                            [multiplier * case * 1.0e-6, 0.0]
                        ),
                    }
                }
        predicted.append(run)

    result = _direct_selected_support_summary(
        predicted, cases=(10, 11), h=0.05, component_index=0
    )

    # For run 1, each leg has pooled mean bound 21e-6/20; summing
    # legs and dividing by 2h gives 21e-6. Run 2 doubles it.
    np.testing.assert_allclose(
        [entry["response_abs_bound"][0] for entry in result["per_prediction_run"]],
        [21.0e-6, 42.0e-6],
    )
    assert result["pooled_ensemble_response_abs_bound"][0] == pytest.approx(
        31.5e-6
    )
    # Worst case is case 11: mean over runs of 2 legs * bound/(2h).
    assert result["maximum_case_ensemble_response_abs_bound"] == pytest.approx(
        33.0e-6
    )
