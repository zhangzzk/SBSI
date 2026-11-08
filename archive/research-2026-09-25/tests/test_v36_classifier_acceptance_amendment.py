from scripts.amend_v36_classifier_acceptance import passes


def check(point, lo, hi):
    return dict(point_degradation=point, ci95_lower=lo, ci95_upper=hi, margin=.05)


def test_small_uncertain_worsening_passes():
    assert passes(check(.01, -.01, .03))


def test_wide_uncertainty_is_not_acceptance():
    assert not passes(check(.01, -.1, .12))


def test_resolved_worsening_fails():
    assert not passes(check(.01, .005, .015))


def test_improvement_and_equality_pass():
    assert passes(check(-.02, -.03, -.01))
    assert passes(check(0, 0, 0))


def test_all_endpoints_and_finite_values_required():
    assert not passes(check([.01, .01], [-.01, .001], [.03, .02]))
    assert not passes(check(float('nan'), -.01, .01))
