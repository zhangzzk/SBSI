#!/usr/bin/env python
"""Freeze measured GalSBI constant-shear image catalogues as paired inference references.

One shared random subset of input identities (case, index_input) is drawn,
independently of any measurement, and each shear leg then applies its own
measured usability and selection cuts (the cuts of the selection
normalization). A galaxy may therefore be selected on one leg only; it is not
required on both. Writes <output>/plus and <output>/minus as
galsbi_population_v1 references for scripts/run_galsbi_inference.py. Rows
that do not match between catalogues are dropped and counted.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.cli import _model_payload
from sbsi.disk.precision import LOG_GAP_COLUMN
from sbsi.models import get_model

TILE = 'tile180.0_-0.5'
MEASURED = ['measured_ngmix_g1', 'measured_ngmix_g2', 'measured_flux_radius', 'measured_flux_from_mag_auto']
# Unsheared input properties; they must be identical between the two legs of a case.
INTRINSIC = ['RA_input', 'DEC_input', 'redshift_input', 'Re_input', 'axis_ratio_input',
             'position_angle_input', 'sersic_n_input', 'r_input']
LEGS = (('plus', .02), ('minus', -.02))
# Protocol entries the inference and selection stages read; values as in the earlier image references.
TEMPLATE = dict(chains=4, draws_per_chain=2048, density_batch_size=4096,
    response_pairing=dict(radius_arcsec=10., nearest_k=20, truth_cuts=None),
    mapping='installed UFig zero-flexion rendered ellipse',
    code_gates=dict(renderer_geometry_max_error=1e-12, log_kernel_replay_max_error=1e-6),
    exploration_gates=dict(max_rank_rhat=1.01, min_bulk_ess=400, min_prior_importance_ess_per_chain=100),
    reference_kind='galsbi_population_v1', shear=None, no_shear_estimate=True, shear_known_to_sampler=False,
    mock_kind='image')


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def leg_paths(root, case, shear):
    base = Path(root)/f'case{case}_{shear:g}'/'real0'/'catalogues'
    return dict(truth=base/f'input/gals_info_{TILE}.feather',
                crossmatch=base/f'CrossMatch/{TILE}_rot0_matched.feather',
                shapes=base/f'Shapes/shape_catalogue_detect_position_all_{TILE}.feather')


def cut_mask(values, cuts, zero_point):
    """values columns NGMIX_G1, NGMIX_G2, MAG_AUTO, FLUX_RADIUS. Returns (usable, selected, flux)."""
    finite = np.isfinite(values).all(1)
    e2 = np.square(values[:, :2]).sum(1)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        flux = 10.**(.4*(zero_point-values[:, 2]))
    usable = finite & (values[:, 3] > 0) & (e2 < 1) & np.isfinite(flux) & (flux > 0)
    # Same comparisons as the selection normalization (run_galsbi_bright.selection), in pixels and flux.
    selected = (usable & (values[:, 3] > cuts['min_radius_pixels']) & (values[:, 3] < cuts['max_radius_pixels'])
                & (flux > 10**(.4*(zero_point-cuts['magnitude_limit'])))
                & (np.sqrt(e2) < cuts['max_measured_ellipticity']))
    return usable, selected, flux


def read_leg(root, case, shear, keep_ids, cuts, zero_point):
    paths = leg_paths(root, case, shear)
    truth = pd.read_feather(paths['truth'], columns=['index_input', *INTRINSIC, 'gamma1_input', 'gamma2_input'])
    cross = pd.read_feather(paths['crossmatch'], columns=['id_detec', 'id_input'])
    shape = pd.read_feather(paths['shapes'], columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS'])
    for frame, keys in ((truth, ['index_input']), (cross, ['id_detec', 'id_input']), (shape, ['NUMBER'])):
        for key in keys:
            if frame[key].isna().any() or frame[key].duplicated().any():
                raise ValueError(f'case {case} leg {shear:g}: missing or duplicate {key}')
    if not np.array_equal(truth[['gamma1_input', 'gamma2_input']].to_numpy(),
                          np.broadcast_to([shear, 0.], (len(truth), 2))):
        raise ValueError(f'case {case}: input shear is not the declared ({shear:g}, 0)')
    joined = cross.merge(shape, left_on='id_detec', right_on='NUMBER', validate='one_to_one')
    matched = joined.merge(truth[['index_input', 'r_input', 'Re_input', 'redshift_input', 'sersic_n_input']],
                           left_on='id_input', right_on='index_input', validate='one_to_one')
    values = matched[['NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS']].to_numpy(float)
    usable, selected, flux = cut_mask(values, cuts, zero_point)
    drawn = matched.id_input.isin(keep_ids).to_numpy()
    chosen = matched.loc[selected & drawn]
    g1, g2 = chosen.NGMIX_G1.to_numpy(float), chosen.NGMIX_G2.to_numpy(float)
    rows = pd.DataFrame({
        'source_case': np.full(len(chosen), case, np.int64),
        'source_input_index': chosen.id_input.to_numpy(np.int64),
        'source_detection_id': chosen.NUMBER.to_numpy(np.int64),
        'measured_ngmix_g1': g1, 'measured_ngmix_g2': g2,
        'measured_flux_radius': chosen.FLUX_RADIUS.to_numpy(float),
        'measured_flux_from_mag_auto': flux[selected & drawn],
        LOG_GAP_COLUMN: np.log1p(-(g1**2+g2**2)),
        'measured_mag_auto': chosen.MAG_AUTO.to_numpy(float),
        'true_r': chosen.r_input.to_numpy(float), 'true_re': chosen.Re_input.to_numpy(float),
        'true_redshift': chosen.redshift_input.to_numpy(float),
        'true_sersic_n': chosen.sersic_n_input.to_numpy(float),
        'injected_g1': np.full(len(chosen), shear), 'injected_g2': np.zeros(len(chosen))})
    report = dict(truth_rows=len(truth), crossmatch_rows=len(cross), shape_rows=len(shape),
        crossmatch_without_shape=len(cross)-len(joined), shape_without_crossmatch=len(shape)-len(joined),
        matched_without_truth=len(joined)-len(matched), unusable=int((~usable).sum()),
        usable=int(usable.sum()), selected=int(selected.sum()), drawn_and_selected=len(rows),
        sources={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()})
    return truth, rows, report


def build(root, cases, output, *, fraction, seed, model_name, cuts):
    if not 0 < fraction <= 1 or not len(cases) or len(set(cases)) != len(cases):
        raise ValueError('fraction in (0, 1] and distinct cases required')
    model = get_model(model_name)
    conditions = model.observing_conditions
    if conditions is None:
        raise ValueError('model must pin its training observing conditions')
    if Path(output).exists():
        raise ValueError('fresh output required')
    frames, reports = {name: [] for name, _ in LEGS}, []
    for case in cases:
        legs, case_report = {}, dict(case=case)
        for name, shear in LEGS:
            truth = pd.read_feather(leg_paths(root, case, shear)['truth'], columns=['index_input'])
            legs[name] = np.sort(truth.index_input.to_numpy(np.int64))
        if not np.array_equal(legs['plus'], legs['minus']):
            raise ValueError(f'case {case}: legs have different input identities')
        # Shared identity draw from (seed, case), independent of every measured quantity.
        ids = legs['plus']
        keep = ids[np.random.default_rng([seed, case]).random(len(ids)) < fraction]
        intrinsic = {}
        for name, shear in LEGS:
            truth, rows, report = read_leg(root, case, shear, keep, cuts, conditions['zero_point'])
            intrinsic[name] = truth.sort_values('index_input')[INTRINSIC].to_numpy()
            frames[name].append(rows)
            case_report[name] = report
        if not np.array_equal(intrinsic['plus'], intrinsic['minus']):
            raise ValueError(f'case {case}: unsheared input properties differ between legs')
        both = np.intersect1d(frames['plus'][-1].source_input_index, frames['minus'][-1].source_input_index)
        case_report.update(drawn_identities=len(keep), selected_both_legs=len(both))
        reports.append(case_report)
        print('CASE', case, len(keep), {n: case_report[n]['drawn_and_selected'] for n, _ in LEGS}, flush=True)
    output = Path(output)
    output.mkdir(parents=True)
    selection_report = dict(cases=reports, cuts=cuts, fraction=fraction, seed=seed,
        unmatched_policy='crossmatch rows without shapes, shapes without crossmatch and matches without truth '
                         'are dropped and counted per case and leg')
    (output/'selection_report.json').write_text(json.dumps(selection_report, indent=1)+'\n')
    summary = {}
    for name, shear in LEGS:
        rows = pd.concat(frames[name], ignore_index=True)
        if rows[['source_case', 'source_input_index']].duplicated().any():
            raise ValueError('duplicate catalogue identities')
        leg = output/name
        leg.mkdir()
        rows[[*MEASURED, LOG_GAP_COLUMN]].to_parquet(leg/'observations.parquet', index=False)
        rows.drop(columns=[*MEASURED, LOG_GAP_COLUMN]).to_parquet(leg/'truth.parquet', index=False)
        protocol = dict(TEMPLATE, conditions=dict(conditions), model_identity=_model_payload(model),
            scene_inputs=model.scene_inputs, injected_shear=[shear, 0.], leg=name, source=str(Path(root).resolve()),
            cases=list(map(int, cases)), selection=(
                f'Shared Bernoulli({fraction:g}) draw of input identities per (seed {seed}, case), independent of '
                'measurement; each leg applies its own measured cuts: finite, radius>0, |e|<1, flux>0, '
                f"radius>{cuts['min_radius_pixels']:g} and <{cuts['max_radius_pixels']:g} pixels, "
                f"MAG_AUTO<{cuts['magnitude_limit']:g} (as flux), measured |e|<{cuts['max_measured_ellipticity']:g}. "
                'Selection on both legs is not required.'),
            cuts=cuts, preparation_sha256=digest(__file__),
            selection_report_sha256=digest(output/'selection_report.json'))
        (leg/'protocol.json').write_text(json.dumps(protocol, indent=1)+'\n')
        result = dict(status='prepared_image_reference', preparation_checks_pass=True,
            observation_count=len(rows), cases=len(cases), observations_sha256=digest(leg/'observations.parquet'),
            truth_sha256=digest(leg/'truth.parquet'), protocol_sha256=digest(leg/'protocol.json'),
            selection_report_sha256=digest(output/'selection_report.json'),
            limitation='Known-shear development check on GalSBI image simulations; shared noise seed between legs')
        (leg/'result.json').write_text(json.dumps(result, indent=1)+'\n')
        summary[name] = len(rows)
    print('IMAGE_REFERENCE_COMPLETE', summary, output, flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sims-root', type=Path,
        default=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_g002/sims'))
    parser.add_argument('--cases', type=int, nargs=2, default=(1000, 1140), help='first and exclusive last case')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fraction', type=float, required=True, help='shared identity sampling probability')
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--model', default='like-1.0-galsbi')
    parser.add_argument('--magnitude-limit', type=float, default=25.)
    parser.add_argument('--min-radius-pixels', type=float, default=3.)
    parser.add_argument('--max-radius-pixels', type=float, default=100.)
    parser.add_argument('--max-measured-ellipticity', type=float, default=.99)
    args = parser.parse_args(argv)
    cuts = dict(magnitude_limit=args.magnitude_limit, min_radius_pixels=args.min_radius_pixels,
                max_radius_pixels=args.max_radius_pixels, max_measured_ellipticity=args.max_measured_ellipticity)
    build(args.sims_root, list(range(*args.cases)), args.output, fraction=args.fraction, seed=args.seed,
          model_name=args.model, cuts=cuts)


if __name__ == '__main__':
    main()
