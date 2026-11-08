#!/usr/bin/env python
"""Matched-truth conditional V3.6-like output marginals, not a prior mixture."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch

from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.crowding import FLOW_FEATURES
from sbsi.disk_inference_store import file_hash, write_json
from sbsi.disk_response_transport import disk_response_transport
from sbsi.flow_size_condition import circularized_radius
from sbsi.forward_catalogue import make_pair_catalogue, prepare_flow_inputs, rescale_emulator_pairs
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE, load_disk_response
from scripts.plot_disk_faint_corners import contour_levels, coordinates
from scripts.probe_disk_derivative_reference import shear_context64


ROWS = (239507, 440648, 274111, 10642)
CONDITIONS = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73,
                  moffat_beta=2.224, pixel_rms=.312)
SCENE_COLUMNS = ('index', 'RA', 'DEC', 'redshift', 'Re', 'axis_ratio',
                 'position_angle', 'sersic_n', 'r')


def build_context(scene_degrees, input_id):
    """No truth cuts; all rendered neighbours for crowding; 20 for response."""
    if scene_degrees['index'].duplicated().any():
        raise ValueError('unique scene identities required')
    scene = scene_degrees.copy()
    # Simulation angles have explicit degree units. Avoid single-row heuristics.
    scene['position_angle'] = np.deg2rad(scene['position_angle'])
    primary = scene.loc[scene['index'] == input_id].reset_index(drop=True)
    if len(primary) != 1:
        raise ValueError('exactly one matching truth object required')
    frame = prepare_flow_inputs(primary, scene, conditions=CONDITIONS)
    frame['e1_input_p'] = frame.e1_input_rot0_p
    frame['e2_input_p'] = frame.e2_input_rot0_p
    frame['circularized_Re_input_p'] = circularized_radius(frame.Re_input_p, frame.axis_ratio_input_p)
    pairs = make_pair_catalogue(primary, scene, r_max_arcsec=10., k=21)
    pairs = pairs.loc[(pairs.distance > 0) & (pairs.distance < 10.)]
    pairs = pairs.sort_values('distance', kind='stable').head(20).reset_index(drop=True)
    pairs = rescale_emulator_pairs(pairs, CONDITIONS)
    return frame.loc[:, FLOW_FEATURES].to_numpy(float), pairs


def validity(values):
    """Count invalid outputs explicitly; never clip, repair, or apply analysis cuts."""
    finite = np.isfinite(values).all(1)
    positive = (values[:, 2:] > 0).all(1)
    disk = np.square(values[:, :2]).sum(1) < 1
    return finite & positive & disk, dict(nonfinite=int((~finite).sum()),
        nonpositive_auxiliary=int((~positive).sum()), outside_open_disk=int((~disk).sum()))


def plot_corner(output, record, x):
    observed = np.array(record['observed_plot_coordinates'])
    labels = [r'Measured $e_1$', r'Measured $e_2$', 'Radius [arcsec]', 'MAG_AUTO']
    edges = []
    for k in range(4):
        low, high = np.quantile(x[:, k], [.001, .999])
        low, high = min(low, observed[k]), max(high, observed[k])
        margin = max((high-low)*.04, 1e-6)
        edges.append(np.linspace(low-margin, high+margin, 65))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
    fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.87, wspace=.11, hspace=.12)
    enclosed = []
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.set_xlim(edges[j][0], edges[j][-1])
            if i == j:
                hist, _ = np.histogram(x[:, i], bins=edges[i])
                ax.stairs(hist/(len(x)*np.diff(edges[i])), edges[i], color='#187b9b', lw=1.4)
                ax.axvline(observed[i], color='#cc2870', lw=1.6)
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
            else:
                ax.set_ylim(edges[i][0], edges[i][-1])
                hist, _, _ = np.histogram2d(x[:, j], x[:, i], bins=(edges[j], edges[i]))
                hist /= len(x)
                levels = contour_levels(hist)
                ax.pcolormesh(edges[j], edges[i], hist.T, cmap='Blues', shading='flat', rasterized=True)
                ax.contour((edges[j][1:]+edges[j][:-1])/2, (edges[i][1:]+edges[i][:-1])/2,
                           hist.T, levels=levels, colors='#187b9b', linewidths=1.2)
                ax.axvline(observed[j], color='#cc2870', lw=.7, alpha=.65)
                ax.axhline(observed[i], color='#cc2870', lw=.7, alpha=.65)
                ax.scatter([observed[j]], [observed[i]], marker='+', color='#cc2870', s=80, linewidth=1.8, zorder=10)
                enclosed.append(dict(pair=[j, i], window_mass=float(hist.sum()),
                    levels=levels.tolist(), mass_above_levels=[float(hist[hist >= v].sum()) for v in levels]))
            if i == 3:
                ax.set_xlabel(labels[j])
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0:
                ax.tick_params(labelleft=False)
    truth = record['truth_metadata']
    fig.suptitle(f'V3.6-like prediction at the actual simulation truth — row {record["row"]}\n'
        f'MAG_AUTO = {observed[3]:.3f}; radius = {observed[2]:.3f} arcsec; '
        f'e = ({observed[0]:.4f}, {observed[1]:.4f})\n'
        f'Case {truth["source_case"]}, input {truth["source_input_index"]}; actual shear g = (0.02, 0)',
        fontsize=12, y=.985)
    invalid = record['n_drawn']-record['n_valid']
    fig.text(.59, .845, 'Diagonal: 1D marginal density\nLower triangle: 2D marginals\nContours: approximately 68% and 95%\n\n'
        'Own truth + own rendered neighbours\nFlow + output-conditioned disk response\nConditional on usable output\nNo prior mixture; no measured analysis cuts\nNo KDE or contour smoothing\n\n'
        f'{record["n_drawn"]:,} forward draws\n{invalid:,} invalid draw(s) excluded and recorded',
        fontsize=10, ha='left', va='top', linespacing=1.5)
    handles = [Line2D([], [], color='#187b9b', label='Matched-truth prediction'),
        Line2D([], [], color='#cc2870', marker='+', ls='', markersize=9, label='Actual measurement')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=2, frameon=False)
    fig.text(.5, .018, 'Examples chosen from the worst-50 approximate-inference curvature list before plotting.\n'
        'Pairwise marginals, not fixed-output slices or an independent calibration test.', ha='center', fontsize=10)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'row_{record["row"]}_truth_corner.{extension}', dpi=170, facecolor='white')
    plt.close(fig)
    return enclosed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--render-from', type=Path, help='Replot verified saved draws without new model evaluations')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=262144)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 1:
        raise ValueError('new output and positive draw count required')
    args.output.mkdir(parents=True)
    if args.render_from:
        report = json.loads((args.render_from/'report.json').read_text())
        for record in report['rows']:
            path = args.render_from/f'row_{record["row"]}_truth_draws.npz'
            if file_hash(path) != record['samples_sha256']:
                raise ValueError('saved samples changed')
            with np.load(path) as arrays:
                record['histogram_contours'] = plot_corner(args.output, record, arrays['samples'])
            record['samples_path'] = str(path.resolve())
        report['render_source'] = str(args.render_from.resolve())
        report['render_script_sha256'] = file_hash(__file__)
        write_json(args.output/'report.json', report)
        return
    if args.run is None:
        raise ValueError('--run required for new draws')
    start = time.monotonic()
    torch.set_num_threads(4)
    audit_path = args.run/'curvature_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    manifest_path = args.run/'input/image_mock_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    measurements = pd.read_parquet(args.run/'input/measurements.parquet')
    metadata = pd.read_parquet(args.run/'input/truth.parquet')
    V36_LIKE.validate()
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device='cuda')
    flow.model.double()
    flow.model.eval()
    response_model = load_disk_response(V36_LIKE, device='cuda', nthread=4)
    assert tuple(flow.condition_preprocessor.feature_names) == tuple(FLOW_FEATURES)
    records = []
    print('TRUTH_CORNER_MODELS_READY', time.monotonic()-start, flush=True)
    for row in ROWS:
        selected = next(r for r in audit['worst'] if r['row'] == row)
        truth = selected['truth']
        for key, value in truth.items():
            if metadata.iloc[row][key] != value:
                raise ValueError(f'frozen row metadata mismatch: {key}')
        case, input_id = truth['source_case'], truth['source_input_index']
        sources = next(c['sources'] for c in manifest['per_case'] if c['case'] == case)
        for spec in sources.values():
            if file_hash(spec['path']) != spec['sha256']:
                raise ValueError('simulation source changed')
        scene = pd.read_feather(sources['truth']['path'])
        scene = scene.rename(columns={f'{c}_input': c for c in SCENE_COLUMNS})
        own = scene.loc[scene['index'] == input_id]
        if len(own) != 1:
            raise ValueError('missing/duplicate own truth; refusing unmatched row')
        g = own[['gamma1_input', 'gamma2_input']].to_numpy(float)[0]
        np.testing.assert_allclose(g, [.02, 0.], rtol=0, atol=1e-14)
        generated_path = Path(sources['truth']['path']).parents[4]/f'gals{case}_0.02.feather'
        generated = pd.read_feather(generated_path)
        original = generated.loc[generated['index'] == input_id]
        if len(original) != 1:
            raise ValueError('missing/duplicate original unsheared truth')
        intrinsic = ['Re', 'axis_ratio', 'position_angle', 'sersic_n', 'r']
        np.testing.assert_allclose(own[intrinsic], original[intrinsic], rtol=2e-6, atol=2e-6)
        cross = pd.read_feather(sources['crossmatch']['path'], columns=['id_detec', 'id_input'])
        match = cross.loc[cross.id_detec == truth['source_detection_id']]
        if len(match) != 1 or int(match.iloc[0].id_input) != input_id:
            raise ValueError('detection-to-truth match differs')
        shapes = pd.read_feather(sources['shapes']['path'], columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO'])
        actual = shapes.loc[shapes.NUMBER == truth['source_detection_id']]
        if len(actual) != 1:
            raise ValueError('missing/duplicate actual measurement')
        actual = actual[['NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO']].to_numpy(float)[0]
        observed = measurements.iloc[row][list(flow.target_transform.target_names)].to_numpy(float)
        expected = actual.copy()
        expected[3] = 10**(.4*(30-expected[3]))
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
        raw, pairs = build_context(scene.loc[:, SCENE_COLUMNS], input_id)
        response = CatalogueDiskResponse.from_pairs([input_id], pairs, response_model)
        record = dict(row=row, truth_metadata=truth, own_truth=own.iloc[0][list(SCENE_COLUMNS)].to_dict(),
            sources=sources, generated_source=dict(path=str(generated_path), sha256=file_hash(generated_path)),
            raw_flow_feature_names=list(FLOW_FEATURES), raw_zero_shear_context=raw.tolist(),
            pair_ids=pairs.index_input_s.tolist(), pair_distances_arcsec=pairs.distance.tolist(),
            observed_physical=observed.tolist(), observed_plot_coordinates=coordinates(observed).tolist(),
            shear=g.tolist(), seed=20260926+row, n_drawn=args.draws, invalid_batches=[])
        draws = []
        torch.manual_seed(record['seed'])
        with torch.no_grad():
            context = shear_context64(torch.tensor(raw, device=flow.device),
                                      torch.tensor(g, device=flow.device), flow.condition_preprocessor)
            record['standardized_sheared_context'] = context.cpu().tolist()
            for lo in range(0, args.draws, 8192):
                n = min(8192, args.draws-lo)
                physical = flow.model.sample_physical(context, n_samples=n)[0]
                valid, counts = validity(physical.cpu().numpy())
                physical = physical[torch.tensor(valid, device=flow.device)]
                if len(physical):
                    values = physical.cpu().numpy()
                    b, _ = response.predict(np.zeros(len(values), dtype=np.int64), values[:, 2], values[:, 3])
                    physical[:, :2], _ = disk_response_transport(physical[:, :2],
                        torch.tensor(b, device=flow.device)[:, None]*physical.new_tensor(g))
                values = physical.cpu().numpy()
                post_valid, post_counts = validity(values)
                draws.append(values[post_valid])
                record['invalid_batches'].append(dict(start=lo, n=n, pre_transport=counts,
                    pre_transport_invalid=int((~valid).sum()), post_transport=post_counts,
                    post_transport_invalid=int((~post_valid).sum())))
        physical = np.concatenate(draws)
        if len(physical) < 1000:
            raise ValueError('too few valid draws for diagnostic contours')
        x = coordinates(physical)
        record['n_valid'] = len(x)
        record['measured_selection_pass_fraction_of_valid'] = float(((physical[:, 2] > 3.) & (x[:, 3] < 25.8)).mean())
        record['observed_marginal_cdf'] = (x <= coordinates(observed)).mean(0).tolist()
        record['predictive_quantiles_001_16_50_84_999'] = np.quantile(x, [.001, .16, .5, .84, .999], axis=0).tolist()
        record['histogram_contours'] = plot_corner(args.output, record, x)
        path = args.output/f'row_{row}_truth_draws.npz'
        np.savez_compressed(path, samples=x, observed=coordinates(observed), shear=g, raw_context=raw)
        record['samples_sha256'] = file_hash(path)
        records.append(record)
        write_json(args.output/'report.json', dict(status='diagnostic_only', rows=records,
            script_sha256=file_hash(__file__), audit_sha256=file_hash(audit_path),
            manifest_sha256=file_hash(manifest_path), model_preset='V3.6-like',
            flow_sha256=V36_LIKE.flow_sha256s[0], response_sha256=V36_LIKE.emulator_sha256,
            seconds=time.monotonic()-start, conditions=CONDITIONS,
            selection='Closest measured magnitude to each bin midpoint 17.5/18.5/19.5/20.5 among frozen worst50 approximate-curvature cases; chosen before predictions.',
            interpretation='Own intrinsic simulation truth and rendered scene; actual g=(.02,0), flow plus disk response conditional on U. No prior atoms or usability weighting or measured cuts. FP64 neural evaluation; original tree model. Invalid numerical draws counted explicitly, excluded without clipping. No KDE. Development cases, not representative validation.'))
        print(f'TRUTH_CORNER_COMPLETE row={row} valid={len(x)}/{args.draws} cdf={record["observed_marginal_cdf"]} seconds={time.monotonic()-start:.1f}', flush=True)


if __name__ == '__main__':
    main()
