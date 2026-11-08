"""Ordered marked-Poisson augmentation density and support accounting."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from sbsi.poisson_augmentation import point_log_augmentation_ratio, poisson_augmentation_log_tilt


RATE = 3.2
BASE_TYPE = np.array([.7, .3])
AUG_TYPE = np.array([.2, .8])


def toy_log_densities(radius_squared, mark, kind):
    # Cartesian unit-disk density times a continuous mark and counting type.
    # Base: uniform position, mark density 2m. h: position 2(1-r²)/pi,
    # mark density 6m(1-m), and explicitly normalized type probabilities.
    intensity = RATE/np.pi*BASE_TYPE[kind]*2*mark
    h = 2*(1-radius_squared)/np.pi*AUG_TYPE[kind]*6*mark*(1-mark)
    return np.log(h), np.log(intensity)


def draw_toy_scenes(count, rng, augmentation_probability=0.):
    base_count = rng.poisson(RATE, count)
    base_ptr = np.r_[0, np.cumsum(base_count)]
    owner = np.repeat(np.arange(count), base_count)
    radius2 = rng.random(base_ptr[-1])
    mark = np.sqrt(rng.random(base_ptr[-1]))
    kind = rng.choice(2, base_ptr[-1], p=BASE_TYPE)
    augmented = rng.random(count) < augmentation_probability
    total = base_count+augmented
    ptr = np.r_[0, np.cumsum(total)]
    slots = np.zeros(count, dtype=int)
    slots[augmented] = rng.integers(total[augmented])
    local = np.arange(base_ptr[-1])-base_ptr[owner]
    destinations = ptr[owner]+local+(augmented[owner] & (local >= slots[owner]))
    result = [np.empty(ptr[-1], dtype=v.dtype) for v in (radius2, mark, kind)]
    for output, original in zip(result, (radius2, mark, kind)):
        output[destinations] = original
    inserted = ptr[:-1][augmented]+slots[augmented]
    size = augmented.sum()
    result[0][inserted] = 1-np.sqrt(rng.random(size))
    result[1][inserted] = rng.beta(2., 2., size)
    result[2][inserted] = rng.choice(2, size, p=AUG_TYPE)
    return ptr, *result


def assert_monte_carlo_mean(values, expected):
    values = np.asarray(values)
    error = values.std(axis=0, ddof=1)/np.sqrt(len(values))
    assert np.all(np.abs(values.mean(axis=0)-expected) < 5*error)


def test_uniform_insertion_and_ordered_factorial_give_sum_without_count_factor():
    radius2, marks, types = np.array([.1, .6, .8]), np.array([.2, .7, .4]), np.array([0, 1, 0])
    logh, loglam = toy_log_densities(radius2, marks, types)
    n, a = len(marks), .4
    logbase = -RATE-gammaln(n+1)+loglam.sum()
    # N alternative insertion histories, each with probability 1/N, starting
    # from an ordered (N-1)-point Poisson configuration.
    histories = [-RATE-gammaln(n)+np.delete(loglam, j).sum()+logh[j]-np.log(n)
                 for j in range(n)]
    logaug = logsumexp(histories)
    expected = np.logaddexp(np.log1p(-a)+logbase, np.log(a)+logaug)-logbase
    actual = poisson_augmentation_log_tilt(point_log_augmentation_ratio(logh, loglam), [0, n], a)
    np.testing.assert_allclose(actual, [expected], atol=2e-15, rtol=0)
    np.testing.assert_allclose(logaug-logbase, logsumexp(logh-loglam), atol=2e-15, rtol=0)


def test_spatial_mark_normalization_and_poisson_count_identity_by_integration():
    spatial, _ = quad(lambda r: 2*(1-r*r)/np.pi*2*np.pi*r, 0, 1)
    mark, _ = quad(lambda m: 6*m*(1-m), 0, 1)
    np.testing.assert_allclose(spatial*mark*AUG_TYPE.sum(), 1., atol=1e-14)
    # Special case h=lambda/RATE makes the density ratio N/RATE. This
    # verifies the count-factorial cancellation against shifted Poisson PMFs.
    counts = np.arange(60)
    ptr = np.r_[0, np.cumsum(counts)]
    a = .45
    tilt = np.exp(poisson_augmentation_log_tilt(np.full(ptr[-1], -np.log(RATE)), ptr, a))
    base = poisson.pmf(counts, RATE)
    np.testing.assert_allclose(base*tilt, (1-a)*base+a*poisson.pmf(counts-1, RATE), atol=1e-16)
    np.testing.assert_allclose(base@tilt, 1., atol=2e-15)


def test_base_poisson_expectation_of_nonuniform_marked_tilt_is_one():
    ptr, radius2, marks, types = draw_toy_scenes(160000, np.random.default_rng(774101))
    logh, loglam = toy_log_densities(radius2, marks, types)
    tilt = np.exp(poisson_augmentation_log_tilt(point_log_augmentation_ratio(logh, loglam), ptr, .45))
    assert_monte_carlo_mean(tilt, 1.)
    assert np.any(np.diff(ptr) == 0)


def test_importance_recovers_base_counts_marks_regions_and_empty_scenes_from_augmented_mixture():
    size, a = 180000, .45
    ptr, radius2, marks, types = draw_toy_scenes(size, np.random.default_rng(774102), a)
    counts = np.diff(ptr)
    owner = np.repeat(np.arange(size), counts)
    logh, loglam = toy_log_densities(radius2, marks, types)
    inverse = np.exp(-poisson_augmentation_log_tilt(point_log_augmentation_ratio(logh, loglam), ptr, a))
    rare_count = np.bincount(owner[(types == 1) & (radius2 < .04) & (marks < .25)], minlength=size)
    values = np.column_stack((np.ones(size), counts,
        np.bincount(owner, weights=marks, minlength=size),
        np.bincount(owner, weights=radius2, minlength=size),
        np.bincount(owner, weights=(types == 1), minlength=size), rare_count > 0, counts == 0))
    expected = np.array([1., RATE, RATE*2/3, RATE/2, RATE*BASE_TYPE[1],
                          -np.expm1(-RATE*BASE_TYPE[1]*.04*.25**2), np.exp(-RATE)])
    assert_monte_carlo_mean(inverse[:, None]*values, expected)
    assert_monte_carlo_mean(counts, RATE+a)
    # The added point has uniform insertion rank. In the N=1 stratum, the
    # empty original scene is retained rather than being rejected/refilled.
    assert np.any(counts == 0) and np.any(counts == 1)


def test_order_invariance_empty_segments_zero_h_and_extreme_log_ratios():
    ratios = np.array([np.log(2.), -np.inf, np.log(3.), np.log(.1), -np.inf, 10000., -10000.])
    pointers = np.array([0, 0, 3, 3, 5, 7, 7], dtype=np.uint64)
    result = poisson_augmentation_log_tilt(ratios, pointers, .3)
    swapped = ratios[[2, 1, 0, 4, 3, 6, 5]]
    np.testing.assert_allclose(result, poisson_augmentation_log_tilt(swapped, pointers, .3), atol=2e-15)
    np.testing.assert_array_equal(result[[0, 2, 5]], np.full(3, np.log1p(-.3)))
    np.testing.assert_allclose(result[4], 10000.+np.log(.3), atol=1e-12, rtol=0)
    assert poisson_augmentation_log_tilt([], [0], .3).shape == (0,)
    np.testing.assert_array_equal(point_log_augmentation_ratio([-np.inf, 0.], [0., np.log(2.)]),
                                  [-np.inf, -np.log(2.)])
    np.testing.assert_array_equal(poisson_augmentation_log_tilt([-np.inf], [0, 1], .3),
                                  [np.log1p(-.3)])


def test_outside_support_mass_is_not_misrepresented_as_normalized_base_tilt():
    # h is normalized globally, with b mass outside the base aperture. On the
    # aperture, h=(1-b)*lambda/RATE. The base expectation is 1-a*b, not one.
    counts, a, b = np.arange(60), .45, .2
    ptr = np.r_[0, np.cumsum(counts)]
    ratio = np.full(ptr[-1], np.log1p(-b)-np.log(RATE))
    base_expectation = poisson.pmf(counts, RATE)@np.exp(poisson_augmentation_log_tilt(ratio, ptr, a))
    np.testing.assert_allclose(base_expectation, 1-a*b, atol=2e-15)
    with pytest.raises(ValueError, match='positive finite base intensity'):
        point_log_augmentation_ratio([0.], [-np.inf])
    with pytest.raises(ValueError, match='positive finite base intensity'):
        point_log_augmentation_ratio([-np.inf], [-np.inf])
    with pytest.raises(ValueError, match='singular mass'):
        poisson_augmentation_log_tilt([np.inf], [0, 1], a)


@pytest.mark.parametrize('pointers', [[1, 1], [0, 2], [0, 1, 0], [0., 1.], []])
def test_invalid_csr_is_rejected(pointers):
    with pytest.raises(ValueError, match='CSR'):
        poisson_augmentation_log_tilt([0.], pointers, .5)


@pytest.mark.parametrize('probability', [0., 1., -1., np.nan, np.inf])
def test_invalid_mixture_probability_is_rejected(probability):
    with pytest.raises(ValueError, match='probability'):
        poisson_augmentation_log_tilt([], [0, 0], probability)
