"""Exact measured-radius partition of cached constant-shear moments.

All contributions use each side's own full selected denominator per shear leg.
The symmetric fraction/conditional-mean split is algebraic, not causal.
"""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
MODES = ['predicted_usability_generated_selection', 'true_usability_generated_selection']


def entry(x):
    return {'value': x[0].tolist(), 'case_se': np.std(x[1:], axis=0, ddof=1).tolist()}


def leg_difference(x):
    return (x[:, 0] - x[:, 1]) / .04


def decompose(sim, model, weights):
    # Input dimensions: case, leg, radius region, intrinsic-mag bin, moment.
    s = np.einsum('bc,clrkq->blrkq', weights, sim)
    x = np.einsum('bc,clrkq->blrkq', weights, model)
    ns, nm = s[..., 0], x[..., 0]
    fs = ns / ns.sum(axis=(-1, -2), keepdims=True)
    fm = nm / nm.sum(axis=(-1, -2), keepdims=True)
    es = s[..., 1:] / ns[..., None]
    em = (x[..., 1:3] + x[..., 3:5]) / nm[..., None]
    assert np.all(ns > 0) and np.all(nm > 0)
    conditional = leg_difference(.5 * (fs + fm)[..., None] * (es - em))
    fractions = leg_difference(.5 * (es + em) * (fs - fm)[..., None])
    delta = leg_difference(fs[..., None] * es - fm[..., None] * em)
    np.testing.assert_allclose(conditional + fractions, delta, atol=1e-12, rtol=1e-12)
    rs = leg_difference((s[..., 1:].sum(axis=(2, 3))) / ns.sum(axis=(2, 3))[..., None])
    rm = leg_difference((x[..., 1:3] + x[..., 3:5]).sum(axis=(2, 3)) / nm.sum(axis=(2, 3))[..., None])
    np.testing.assert_allclose(delta.sum(axis=(1, 2)), rs - rm, atol=1e-12, rtol=1e-12)
    return {
        'total_delta_response': entry(rs - rm),
        'radius_region_delta_contribution': entry(delta.sum(axis=2)),
        'intrinsic_mag_delta_contribution': entry(delta.sum(axis=1)),
        'radius_and_intrinsic_mag_delta_contribution': entry(delta),
        'conditional_mean_contribution': entry(conditional.sum(axis=(1, 2))),
        'fraction_contribution': entry(fractions.sum(axis=(1, 2))),
        'radius_conditional_mean_contribution': entry(conditional.sum(axis=2)),
        'radius_fraction_contribution': entry(fractions.sum(axis=2)),
        'simulation_selected_fractions_per_leg': entry(fs.sum(axis=-1)),
        'model_selected_fractions_per_leg': entry(fm.sum(axis=-1)),
    }


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    cases = json.loads((HERE / 'manifest.json').read_text())['cases']
    weights = np.vstack([np.ones(len(cases)), np.random.default_rng(202611067001).multinomial(
        len(cases), np.full(len(cases), 1 / len(cases)), 10000)])
    sources = {
        'axis1_retained': ROOT / 'galsbi_dev_constant_grid_20261104/sums',
        'axis1_cartesian': ROOT / 'alignment_factorized_constant_20261105/cartesian',
        'axis1_seed612': ROOT / 'alignment_cartesian_seeds_20261105/constant/seed612',
        'axis1_seed613': ROOT / 'alignment_cartesian_seeds_20261105/constant/seed613',
        'axis2_retained': ROOT / 'alignment_axis2_constant_20261105/retained',
        'axis2_cartesian': ROOT / 'alignment_axis2_constant_20261105/cartesian',
    }
    out = {'cases': cases, 'definition': 'DeltaR=simulation-model; exact partition of selected ensemble. '
           'Own per-leg denominator on both sides. Symmetric fraction/mean split is descriptive, '
           'not a physical attribution; 10000 paired whole-case bootstraps, fixed models/catalogues.',
           'intrinsic_magnitude_bins': ['<24', '24..25.5', '25.5..27.5', '>=27.5'], 'arms': {}}
    for arm, directory in sources.items():
        models, sims = [], []
        names = ['mag25.8', 'mag25.8_radius_gt0.60', 'mag25.8_radius_ge0.75']
        for case in cases:
            with np.load(directory / f'case{case}.npz') as z:
                ci = [list(z['cuts']).index(n) for n in names]
                models.append(z['model'][:2, :, ci, :, :5])
                sims.append(z['sim'][:, ci, :, :3])
        models, sims = np.array(models), np.array(sims)
        out['arms'][arm] = {}
        for mode, label in enumerate(MODES):
            m = models[:, mode]
            # Disjoint bins: r<=.6, .6<r<.75, r>=.75.
            sb = np.stack([sims[:, :, 0] - sims[:, :, 1], sims[:, :, 1] - sims[:, :, 2], sims[:, :, 2]], axis=2)
            mb = np.stack([m[:, :, 0] - m[:, :, 1], m[:, :, 1] - m[:, :, 2], m[:, :, 2]], axis=2)
            for a in (sb, mb):
                assert np.all(a[..., 0] >= 0), 'Nested selection lost mass'
            out['arms'][arm][label] = {
                'mag25.8': decompose(sb, mb, weights),
                'mag25.8_radius_gt0.60': decompose(sb[:, :, 1:], mb[:, :, 1:], weights),
                'mag25.8_radius_ge0.75': decompose(sb[:, :, 2:], mb[:, :, 2:], weights),
            }
        print('DONE', arm, flush=True)
    path = ROOT / 'alignment_axis2_constant_20261105/nested_selection.json'
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(path, flush=True)


if __name__ == '__main__':
    main()
