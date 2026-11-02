"""Build the F8/C8 training domain from the GalSBI re-simulation (training family, legs 0.0 and 0.05).

Writes the format the frozen F8 trainer (/project/ls-gruen/users/zekang.zhang/sbsi_flow_f8_20260924) reads:
``flow/{g0,g05}/caseNNN.npz`` (usable parents) and ``usability/{g0,g05}/caseNNN.npz`` (all parents), then
``manifest.json`` and ``completed_projection.json`` with ``--finalize``. One directory serves as both the base
and the harmonized domain: the new sims are measured once, with centred 65 px stamps.

Parent = the sheared half of the scene (index >= n/2), as rendered. Rendered rows the renderer omitted (edge
galaxies) are dropped and reported. Inputs per leg: Mobius-sheared intrinsic e1, e2 at that leg's shear, rendered
Sersic n, LSST-i magnitude, circularized Re = Re*sqrt(q) (arcsec), and the three close-crowding sums of all
rendered objects (sbsi.v39_inference.physical_close_crowding). Targets: ngmix g1, g2 of the secondaries shape
catalogue, FLUX_RADIUS (px), flux 10**(-0.4*(MAG_AUTO-30)). Usable = detected, finite, g1^2+g2^2 < 1, positive
radius and flux; unusable parents stay in the usability files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.v39_inference import physical_close_crowding

LEGS = {'g0': '0.0', 'g05': '0.05'}
FEATURES = ['e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p', 'circularized_Re_input_p',
            'nbr_close_050', 'nbr_close_100', 'nbr_close_200']
TARGETS = ['measured_ngmix_g1', 'measured_ngmix_g2', 'measured_flux_radius', 'measured_flux_from_mag_auto']
TILE = 'tile180.0_-0.5'
TRUTH_MAX = 30.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stat(path):
    s = os.stat(path)
    return {'path': str(path), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def split_cases(cases, seed, validation_size):
    """Grouped ShuffleSplit convention of the frozen sbsi.fixed_g0_domain.split_group_values."""
    groups = np.asarray(sorted(set(cases)), dtype=np.int64)
    n_validation = int(np.ceil(validation_size*groups.size))
    permutation = np.random.RandomState(seed).permutation(groups.size)
    return groups[permutation[n_validation:]].tolist(), groups[permutation[:n_validation]].tolist()


def measured(base):
    """Detections of one leg joined one-to-one: truth id -> ngmix shape, radius, flux."""
    cross = pd.read_feather(base/f'CrossMatch/{TILE}_rot0_matched.feather', columns=['id_input', 'id_detec'])
    shapes = pd.read_feather(base/f'Shapes/shape_catalogue_detect_position_secondaries_{TILE}.feather',
                             columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO'])
    for frame, column in ((cross, 'id_input'), (cross, 'id_detec'), (shapes, 'NUMBER')):
        if frame[column].duplicated().any():
            raise ValueError(f'non-unique {column} in {base}')
    joined = cross.merge(shapes, left_on='id_detec', right_on='NUMBER', how='inner', validate='one_to_one')
    report = {'crossmatch_rows': len(cross), 'shape_rows': len(shapes), 'joined_rows': len(joined),
              'crossmatch_without_shape': int(len(cross)-len(joined))}
    return joined.set_index('id_input'), report


def one_case(sims, out, case):
    record = {'case': case, 'legs': {}}
    generated = pd.read_feather(sims/f'gals{case}_0.05.feather', columns=['index'])
    if not np.array_equal(generated['index'].to_numpy(), np.arange(len(generated))):
        raise ValueError(f'case {case}: generated index not contiguous')
    boundary = len(generated)//2
    truth = {}
    for leg, h in LEGS.items():
        truth[leg] = pd.read_feather(sims/f'case{case}_{h}/real0/catalogues/input/gals_info_{TILE}.feather')
        truth[leg] = truth[leg].sort_values('index_input', kind='stable').reset_index(drop=True)
    fixed = [c for c in truth['g0'].columns if c not in ('gamma1_input', 'gamma2_input')]
    for c in fixed:
        if not np.array_equal(truth['g0'][c].to_numpy(), truth['g05'][c].to_numpy()):
            raise ValueError(f'case {case}: truth column {c} differs between legs')
    t = truth['g0']
    ids = t['index_input'].to_numpy(np.int64)
    rendered_parent = ids >= boundary
    unrendered = np.setdiff1d(np.arange(boundary, len(generated)), ids[rendered_parent])
    record.update(generated_rows=len(generated), parent_boundary=boundary,
                  rendered_rows=len(t), rendered_parents=int(rendered_parent.sum()),
                  unrendered_parents_dropped=unrendered.tolist())
    scene = pd.DataFrame({'RA': t['RA_input'], 'DEC': t['DEC_input'], 'r': t['r_input']})
    crowd = physical_close_crowding(scene, np.flatnonzero(rendered_parent))
    parent = t.loc[rendered_parent]
    if (parent['r_input'] >= TRUTH_MAX).any():
        raise ValueError(f'case {case}: parent beyond the declared truth bound')
    q = parent['axis_ratio_input'].to_numpy(float)
    static = np.column_stack([parent['sersic_n_input'], parent['r_input'], parent['Re_input'].to_numpy(float)*np.sqrt(q),
                              crowd])
    parent_ids = parent['index_input'].to_numpy(np.int64)
    for leg, h in LEGS.items():
        p = truth[leg].loc[rendered_parent]
        gamma = p[['gamma1_input', 'gamma2_input']].to_numpy(np.float64)
        amplitude = np.hypot(gamma[:, 0], gamma[:, 1])
        if leg == 'g0' and np.any(gamma != 0):
            raise ValueError(f'case {case}: nonzero shear in the zero leg')
        if leg == 'g05' and np.abs(amplitude-.05).max() > 1e-12:
            raise ValueError(f'case {case}: parent |gamma| != 0.05 ({np.abs(amplitude-.05).max():.3g})')
        e1, e2 = apply_shear_to_ellipticity(p['e1_input_rot0'], p['e2_input_rot0'], gamma[:, 0], gamma[:, 1])
        context = np.column_stack([e1, e2, static]).astype(np.float32)
        joined, report = measured(sims/f'case{case}_{h}/real0/catalogues')
        detected = np.isin(parent_ids, joined.index.to_numpy())
        target = np.full((len(parent_ids), 4), np.nan)
        m = joined.reindex(parent_ids[detected])
        target[detected] = np.column_stack([m['NGMIX_G1'], m['NGMIX_G2'], m['FLUX_RADIUS'],
                                            10.**(-.4*(m['MAG_AUTO'].to_numpy(float)-30.))])
        finite = np.isfinite(target).all(1) & np.isfinite(context).all(1) & np.isfinite(gamma).all(1)
        with np.errstate(invalid='ignore'):
            interior = target[:, 0]**2+target[:, 1]**2 < 1
            positive = (target[:, 2] > 0) & (target[:, 3] > 0)
        usable = detected & finite & interior & positive
        for name, arrays in (('usability', dict(input_index=parent_ids, context=context, gamma=gamma,
                                                usable=usable, detected=detected)),
                             ('flow', dict(case=np.full(usable.sum(), case, np.int32), input_index=parent_ids[usable],
                                           context=context[usable], target=target[usable], gamma=gamma[usable]))):
            path = out/f'{name}/{leg}/case{case:03d}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays)
        flagged = detected & (m['MAG_AUTO'].reindex(parent_ids).to_numpy(float) >= 90)
        record['legs'][leg] = report | {
            'parents_detected': int(detected.sum()), 'parents_usable': int(usable.sum()),
            'detected_not_finite': int((detected & ~finite).sum()),
            'detected_not_interior': int((detected & finite & ~interior).sum()),
            'detected_not_positive': int((detected & finite & interior & ~positive).sum()),
            'detected_mag_auto_ge_90': int(flagged.sum()),
            'flow_rows': int(usable.sum()), 'flow_sha256': sha256(out/f'flow/{leg}/case{case:03d}.npz'),
            'usability_sha256': sha256(out/f'usability/{leg}/case{case:03d}.npz'),
            'sources': [stat(sims/f'case{case}_{h}/real0/catalogues/{x}') for x in
                        (f'input/gals_info_{TILE}.feather', f'CrossMatch/{TILE}_rot0_matched.feather',
                         f'Shapes/shape_catalogue_detect_position_secondaries_{TILE}.feather')]}
    (out/'records').mkdir(exist_ok=True)
    (out/f'records/case{case:03d}.json').write_text(json.dumps(record, indent=1)+'\n')
    print(f'CASE {case} parents={len(parent_ids)} unrendered={len(unrendered)} '
          f'usable g0={record["legs"]["g0"]["parents_usable"]} g05={record["legs"]["g05"]["parents_usable"]}', flush=True)


def finalize(sims, out, cases, seed, validation_size):
    records = [json.loads((out/f'records/case{c:03d}.json').read_text()) for c in cases]
    train, validation = split_cases(cases, seed, validation_size)
    counts = {leg: {'truth_parent_rows': sum(r['rendered_parents'] for r in records),
                    'usable_rows': sum(r['legs'][leg]['parents_usable'] for r in records),
                    'detected_rows': sum(r['legs'][leg]['parents_detected'] for r in records)} for leg in LEGS}
    for leg in LEGS:
        counts[leg]['invalid_rows_after_truth_cut'] = counts[leg]['truth_parent_rows']-counts[leg]['usable_rows']
    unrendered = {r['case']: r['unrendered_parents_dropped'] for r in records if r['unrendered_parents_dropped']}
    manifest = {
        'format_version': 1,
        'population': {'name': 'galsbi_resim_full_valid_measurement_domain_true_mag_lt30_centred65_v1',
                       'anchor_leg': None, 'predicate': None, 'measured_selection_cut': None,
                       'sheared_leg_recut': None, 'independent_per_leg_validity': True,
                       'inherited_parent_support_only': False,
                       'truth_analysis_cut': {'column': 'r_input_p', 'operator': '<', 'value': TRUTH_MAX}},
        'flow': {'features': FEATURES, 'targets': TARGETS,
                 'per_leg_validity': 'detected; finite inputs/targets/gamma; open shape disk; positive radius and flux',
                 **{f'{leg}_cases': {str(r['case']): {'rows': r['legs'][leg]['flow_rows'],
                                                      'sha256': r['legs'][leg]['flow_sha256']} for r in records}
                    for leg in LEGS}},
        'split': {'seed': seed, 'validation_size': validation_size, 'train_cases': train,
                  'validation_cases': validation},
        'counts': counts,
        'unrendered_parents_dropped': {'rows': sum(map(len, unrendered.values())), 'by_case': unrendered},
        'measurement': {'stamp_size': 65, 'centering': 'centred', 'pixel_scale': 0.2, 'band': 'LSST-i',
                        'shape_catalogue': 'shape_catalogue_detect_position_secondaries',
                        'noise': 'per-case noise field shared by the two legs'},
        'sources': {'sims': str(sims), 'legs': LEGS, 'script_sha256': sha256(__file__)},
        'limitations': ['GalSBI training family 0-199 only; no calibration acceptance',
                        'renderer-omitted parents at the patch edge are dropped and listed',
                        'usability records cover rendered parents (all have true r < 30)'],
    }
    path = out/'manifest.json'
    path.write_text(json.dumps(manifest, indent=1)+'\n')
    files = {f'{name}/{leg}/case{r["case"]:03d}.npz': r['legs'][leg][f'{name}_sha256']
             for r in records for leg in LEGS for name in ('flow', 'usability')}
    (out/'completed_projection.json').write_text(json.dumps(
        {'files': files, 'manifest_sha256': sha256(path), 'altered_labels': 0, 'removed_rows': 0,
         'note': 'built directly from the sims; unrendered parents are listed in manifest.json'}, indent=1)+'\n')
    print('FINALIZED', json.dumps(counts), 'unrendered', manifest['unrendered_parents_dropped']['rows'], flush=True)


def write_c8(out, c8, cases):
    """C8 prepared cases (frozen format scene_classifier_physical_f8_v1): per-leg inputs and labels are the
    usability arrays unchanged. The frozen trainer fixes cases 40-99 (train) and 100-119 (tune)."""
    domain_sha = sha256(out/'manifest.json')
    (c8/'cases').mkdir(parents=True)
    records = {}
    for case in cases:
        arrays = {}
        for i, leg in enumerate(LEGS):
            with np.load(out/f'usability/{leg}/case{case:03d}.npz') as z:
                if i == 0:
                    arrays['ids'] = z['input_index']
                elif not np.array_equal(arrays['ids'], z['input_index']):
                    raise ValueError(f'case {case}: legs disagree on parents')
                arrays |= {f'x{i}': z['context'], f'u{i}': z['usable'], f'd{i}': z['detected']}
        path = c8/f'cases/case{case:03d}.npz'
        np.savez(path, **arrays)
        records[str(case)] = {'prepared_sha256': sha256(path), 'parents': len(arrays['ids']),
                              'usable': [int(arrays['u0'].sum()), int(arrays['u1'].sum())]}
    manifest = {'format_version': 'scene_classifier_physical_f8_v1', 'coordinate_convention': 'physical_arcsec',
                'with_corrected_labels': True, 'cases': cases, 'features': FEATURES, 'case_records': records,
                'population': 'rendered GalSBI parents (sheared half); labels = per-leg usable; no cuts',
                'truth_analysis_cut': None, 'kernel_scales': [.5, 1., 2.], 'distance_unit': 'arcsec',
                'source_domain_manifest_sha256': domain_sha, 'script_sha256': sha256(__file__)}
    (c8/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n')
    smoke = c8.parent/'smoke_prepared'
    (smoke/'cases').mkdir(parents=True)
    for case in (40, 100):
        (smoke/f'cases/case{case:03d}.npz').symlink_to(c8/f'cases/case{case:03d}.npz')
    (smoke/'manifest.json').write_text(json.dumps(manifest | {'cases': [40, 100]}, indent=1)+'\n')
    print('C8_PREPARED cases', len(cases), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sims', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', type=int, nargs='+', required=True)
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--c8-out', type=Path, help='with --finalize: also write the C8 prepared cases 40-119 here')
    ap.add_argument('--split-seed', type=int, default=501)
    ap.add_argument('--validation-size', type=float, default=0.2)
    a = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('run under Slurm')
    if a.finalize:
        finalize(a.sims, a.out, a.cases, a.split_seed, a.validation_size)
        if a.c8_out is not None:
            write_c8(a.out, a.c8_out, list(range(40, 120)))
    else:
        for case in a.cases:
            one_case(a.sims, a.out, case)


if __name__ == '__main__':
    main()
