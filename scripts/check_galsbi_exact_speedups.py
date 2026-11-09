"""Bitwise checks of the exact sampler speed-ups (CPU, a few thousand rows; seconds on a login node).

1. Deferred neighbour Sersic: a defer_sersic draw, filled in shuffled pieces, equals the eager draw.
2. Photometry reuse: log_mixture_prior with decode's photometry equals per-type recomputation.
3. Scene refresh with deferred neighbours equals the eager refresh after filling (same random stream).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from check_galsbi_continuous_prior import build_prior  # noqa: E402
from sbsi.galsbi.coordinates import GalSBITargetCoordinates  # noqa: E402
from sbsi.galsbi.model_inputs import SceneMarks  # noqa: E402
from sbsi.galsbi.prior import ExactMemo  # noqa: E402
from sbsi.galsbi.scenes import GalSBIScenePrior  # noqa: E402
from sbsi.models import get_model  # noqa: E402


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def main():
    galaxy, cosmo, _ = build_prior()
    SceneMarks(get_model('like-1.0-galsbi').scene_inputs, galaxy)
    failures = []

    eager, _ = galaxy.sample(20000, 11)
    lazy, _ = galaxy.sample(20000, 11, defer_sersic=True)
    if not np.isnan(lazy['n']).all():
        failures.append('deferred draw has finite n')
    order = np.random.default_rng(5).permutation(len(lazy['n']))
    for piece in np.array_split(order, 7):
        galaxy.fill_sersic(lazy, piece)
    for k, v in eager.items():
        if not same(v, lazy[k]):
            failures.append(f'deferred draw differs in {k}')
    print('1 deferred Sersic:', len(eager['n']), 'galaxies')

    distance = ExactMemo(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))
    coords = GalSBITargetCoordinates(galaxy, distance)
    x = coords.encode(eager)
    x = np.concatenate((x, x+np.random.default_rng(6).normal(0, .1, x.shape)))  # also off-support rows
    new_total, new_data, new_p = coords.log_mixture_prior(x)
    data, logj = coords.decode(x, np.zeros(len(x), dtype=int))
    terms = []
    for kind in range(len(galaxy.types)):
        data['type'] = np.full(len(x), kind, dtype=int)
        terms.append(galaxy.log_intensity(data))
    from scipy.special import logsumexp
    old_total = logsumexp(np.column_stack(terms), axis=1)+logj
    if not same(old_total, new_total):
        failures.append(f'log_mixture_prior differs: max {np.nanmax(np.abs(old_total-new_total))}')
    print('2 photometry reuse:', len(x), 'rows,', int(np.isfinite(new_total).sum()), 'on support')

    targets = {k: v[:3000] for k, v in eager.items() if k != 'proposal_index'}
    for fraction, seed in ((.1, 21), (1., 22)):
        scenes = {}
        for defer in (False, True):
            sp = GalSBIScenePrior(galaxy, defer_neighbour_sersic=defer)
            base, _ = sp.sample_given_targets(targets, 31)
            out = sp.refresh(base, fraction, seed)
            sp.fill_neighbour_sersic(out, np.arange(0, 3000, 2))
            sp.fill_neighbour_sersic(out)
            scenes[defer] = out
        a, b = scenes[False], scenes[True]
        for k in a.neighbours:
            if not same(a.neighbours[k], b.neighbours[k]):
                failures.append(f'refresh {fraction} differs in neighbour {k}')
        if not (same(a.indptr, b.indptr) and same(a.offset_arcsec, b.offset_arcsec)):
            failures.append(f'refresh {fraction} differs in layout')
        print(f'3 refresh fraction {fraction}:', a.indptr[-1], 'neighbours')
    print('FAILURES', failures if failures else 'none')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
