#!/usr/bin/env python
"""Pairwise slices of a fixed diagnostic atom-union contribution to p(y|g).

Not marginal densities or a full-prior likelihood surface. The two unplotted
outputs stay at the observation. Original pi*p(U|atom,g) weights are retained;
posterior atom weights are NEVER reused as mixture weights. No KDE is used.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from scripts.probe_disk_curvature import load_runtime
from scripts.probe_disk_derivative_reference import shear_context64
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.disk_inference_store import file_hash, write_json
from sbsi.disk_response_transport import disk_response_transport, transported_physical_log_prob
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE


def compact_response(response, atoms):
    parts, counts = [], []
    for atom in atoms:
        shard = np.searchsorted(response.offsets, atom, side='right')-1
        source = response.responses[shard]
        local = atom-response.offsets[shard]
        part = source.features[source.indptr[local]:source.indptr[local+1]]
        parts.append(part)
        counts.append(len(part))
    return CatalogueDiskResponse(np.r_[0, np.cumsum(counts)],
        np.concatenate(parts), response.responses[0].model)


def slice_points(observed, pair, axes):
    xx, yy = np.meshgrid(axes[pair[0]], axes[pair[1]])
    points = np.broadcast_to(observed, (xx.size, 4)).copy()
    points[:, pair[0]], points[:, pair[1]] = xx.ravel(), yy.ravel()
    return points, xx, yy


def plot_row(output, row, axes, surfaces, observed, capture, atom_count, center, h):
    # OO layout and sequential colormap follow the Matplotlib skill template.
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, panels = plt.subplots(2, 3, figsize=(13, 8.3), layout='constrained')
    fig.set_layout_engine('constrained', rect=(0, .115, 1, .885))
    labels = [r'Measured $e_1$', r'Measured $e_2$',
              r'Measured radius [arcsec]', r'Flux / observed flux']
    factor = np.array([1., 1., .2, 1/observed[3]])
    measured = observed*factor
    for ax, pair, surface in zip(panels.flat, itertools.combinations(range(4), 2), surfaces):
        x, y = axes[pair[0]]*factor[pair[0]], axes[pair[1]]*factor[pair[1]]
        anchor = np.max(surface[0])
        relative = (surface-anchor)/np.log(10)
        im = ax.pcolormesh(x, y, relative[0], cmap='viridis', vmin=-4, vmax=0,
                           shading='auto', rasterized=True)
        for node, color, style in ((0, 'white', '-'), (1, '#ff8c42', '--')):
            ax.contour(x, y, relative[node], levels=[-3., -2., -1., np.log10(.5)],
                       colors=color, linestyles=style, linewidths=.9)
        ax.axvline(measured[pair[0]], color='#ed3e76', lw=.8, alpha=.7)
        ax.axhline(measured[pair[1]], color='#ed3e76', lw=.8, alpha=.7)
        ax.scatter([measured[pair[0]]], [measured[pair[1]]], s=115, marker='+',
                   color='#ff4081', linewidth=2.3, zorder=10)
        ax.set(xlabel=labels[pair[0]], ylabel=labels[pair[1]])
        ax.ticklabel_format(useOffset=False, style='plain')
    fig.colorbar(im, ax=panels, fraction=.025, pad=.02,
        label=r'$\log_{10}(L_S / L_{S,\mathrm{peak},g_0})$  [each slice]')
    mag = 30-2.5*np.log10(observed[3])
    title = (f'V3.6-like joint-density slices — observation {row}\n'
        f'e = ({observed[0]:.4f}, {observed[1]:.4f}); radius = {observed[2]*.2:.3f} arcsec; MAG_AUTO = {mag:.3f}\n'
        f'{atom_count} fixed atoms; capture at observed point: {capture[0]:.1%} at g₀, {capture[1]:.1%} at g₀ + Δg₁')
    fig.suptitle(title, fontsize=13)
    handles = [Line2D([], [], color='black', label=f'g₀ = ({center[0]:.5f}, {center[1]:.5f})'),
        Line2D([], [], color='#ff8c42', ls='--', label=f'g₀ + Δg₁; Δg₁ = {h:g}'),
        Line2D([], [], color='#ff4081', marker='+', ls='', markersize=10, label='Observed values')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .049),
               ncol=3, frameon=False)
    fig.text(.5, .006, 'Other two outputs fixed at observation. Truncated prior sum, not a marginal or credible region.\n'
        'Contours: 0.1%, 1%, 10%, 50% of the center-shear slice peak. Off-point atom coverage is unknown.',
        fontsize=10, ha='center', va='bottom')
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'row_{row}_joint_slices.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=97)
    parser.add_argument('--view', choices=('overview', 'local'), default='overview')
    parser.add_argument('--render-only', type=Path, help='Redraw saved grids without model evaluation')
    args = parser.parse_args()
    if args.output.exists() or args.grid < 5 or args.grid % 2 != 1:
        raise ValueError('new output directory and odd grid >=5 required')
    args.output.mkdir(parents=True)
    if args.render_only:
        report = json.loads((args.render_only/'report.json').read_text())
        points = np.array(report['shears'])
        for row in report['rows']:
            data = np.load(args.render_only/f'row_{row["row"]}_joint_slices.npz')
            plot_row(args.output, row['row'], data['axes'], data['log_density'],
                data['observed'], row['capture_at_observation'], row['n_atoms'],
                points[0], points[1, 0]-points[0, 0])
        report['render_source'] = str(args.render_only)
        report['render_script_sha256'] = file_hash(__file__)
        write_json(args.output/'report.json', report)
        return
    torch.set_num_threads(4)
    started = time.monotonic()
    likelihood, proposal, mock, prepared = load_runtime(args.run)
    del proposal
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device='cuda')
    flow.model.double()
    device = flow.device
    center = np.array(prepared['identity']['center'])
    h = prepared['identity']['h']
    points = np.array([center, center+[h, 0]])
    means = torch.tensor(flow.target_transform.means, dtype=torch.float64, device=device)
    scales = torch.tensor(flow.target_transform.scales, dtype=torch.float64, device=device)
    reports = []
    print('PLOT_RUNTIME_READY', time.monotonic()-started, flush=True)
    for worker, row in enumerate((142230, 3563)):
        source = args.diagnostics/f'worker_{worker}'/f'row_{row}_selected.json'
        selected = json.loads(source.read_text())
        if selected['protocol']['preparation_sha256'] != file_hash(args.run/'disk_assembled_v1/manifest.json'):
            raise ValueError('diagnostic preparation mismatch')
        info = selected['selected']
        atoms = np.array(info['atoms'])
        response = compact_response(likelihood.disk_response, atoms)
        observed = mock.measurements.iloc[row][list(flow.target_transform.target_names)].to_numpy(float)
        raw = torch.tensor(likelihood.cache.zero_flow.iloc[atoms].to_numpy(float), dtype=torch.float64, device=device)
        contexts = [shear_context64(raw, torch.tensor(g, device=device), flow.condition_preprocessor) for g in points]
        log_weights = [torch.tensor(np.log(likelihood.cache.get(*g).detection_probability[atoms])
            -np.log(likelihood.disk_response.n_atoms), device=device) for g in points]
        log_b = [likelihood.population_normalization.log_mass(*g) for g in points]

        def evaluate(values):
            result = np.full((2, len(values)), -np.inf)
            valid = ((values[:, :2]**2).sum(1) < 1) & (values[:, 2] > 3) & (values[:, 3] > 10**((30-25.8)/2.5))
            rows = np.flatnonzero(valid)
            # Each neural batch has at most ~4096 atom/output pairs.
            block = max(1, 4096//len(atoms))
            for lo in range(0, len(rows), block):
                ids = rows[lo:lo+block]
                batch = values[ids]
                local = np.tile(np.arange(len(atoms)), len(batch))
                physical = np.repeat(batch, len(atoms), axis=0)
                coeff, _ = response.predict(local, physical[:, 2], physical[:, 3])
                physical = torch.tensor(physical, device=device)
                coeff = torch.tensor(coeff, device=device)
                for node, g in enumerate(points):
                    context = contexts[node].repeat(len(batch), 1)
                    term = transported_physical_log_prob(physical, coeff[:, None]*physical.new_tensor(g),
                        lambda base: flow.model.log_prob((base-means)/scales, context))
                    # Convert standardized output density to physical density.
                    term = term.reshape(len(batch), len(atoms))+log_weights[node]
                    result[node, ids] = (torch.logsumexp(term, 1)-scales.log().sum()-log_b[node]).cpu().numpy()
            return result

        with torch.no_grad():
            # Forward draws choose plotting limits only; never alter likelihood.
            terms_path = source.parent/f'row_{row}_h0.0010000_atom_terms.npz'
            saved = np.load(terms_path)
            np.testing.assert_array_equal(atoms, saved['atoms'])
            leading = np.unique(np.argsort(saved['log_terms'][:2], axis=1)[:, -3:].ravel())
            draws = []
            for node, g in enumerate(points):
                torch.manual_seed(20260924+row)
                draw = flow.model.sample_physical(contexts[node][leading], n_samples=1024).reshape(-1, 4)
                a = np.repeat(leading, 1024)
                physical = draw.cpu().numpy()
                coeff, _ = response.predict(a, physical[:, 2], physical[:, 3])
                draw[:, :2], _ = disk_response_transport(draw[:, :2],
                    torch.tensor(coeff, device=device)[:, None]*draw.new_tensor(g))
                draws.append(draw.cpu().numpy())
            quantile = np.quantile(np.concatenate(draws), [.16, .84], axis=0)
            width = np.maximum(np.max(np.abs(quantile-observed), axis=0)*1.5,
                               [.01, .01, .1, .01*observed[3]])
            if args.view == 'local':
                # A display window, not a population cut or likelihood modification.
                width = np.array([.015, .01, .5, .15*observed[3]])
            axes = np.array([np.linspace(v-w, v+w, args.grid) for v, w in zip(observed, width)])
            axes[:, args.grid//2] = observed
            observed_log = evaluate(observed[None])[:, 0]
            surfaces = []
            for pair in itertools.combinations(range(4), 2):
                values, xx, yy = slice_points(observed, pair, axes)
                surface = evaluate(values).reshape(2, *xx.shape)
                np.testing.assert_allclose(surface[:, args.grid//2, args.grid//2], observed_log, atol=1e-9)
                surfaces.append(surface)
                print(f'PLOT_SLICE row={row} pair={pair} seconds={time.monotonic()-started:.1f}', flush=True)
        surfaces = np.array(surfaces)
        capture = np.array(info['capture_at_original_nodes'])[:2]
        plot_row(args.output, row, axes, surfaces, observed, capture, len(atoms), center, h)
        np.savez_compressed(args.output/f'row_{row}_joint_slices.npz', axes=axes,
            log_density=surfaces, observed=observed, atoms=atoms, shears=points)
        report = dict(row=row, source=str(source), source_sha256=file_hash(source),
            terms_sha256=file_hash(terms_path), observed=observed.tolist(),
            target_names=flow.target_transform.target_names, n_atoms=len(atoms),
            capture_at_observation=capture.tolist(), log_density_at_observation=observed_log.tolist(),
            log_density_change=float(observed_log[1]-observed_log[0]))
        reports.append(report)
        print('PLOT_ROW_COMPLETE', json.dumps(report), flush=True)
    write_json(args.output/'report.json', dict(status='diagnostic_only', rows=reports,
        script_sha256=file_hash(__file__), preparation_sha256=file_hash(args.run/'disk_assembled_v1/manifest.json'),
        grid=args.grid, view=args.view, shears=points.tolist(), seconds=time.monotonic()-started,
        precision='Full FP64 input shear, preprocessing and neural evaluation; cached FP32 classifier.',
        interpretation='Pairwise joint-density slices, other outputs fixed at observed values. Original uniform prior and usability weights, full-prior cached selection normalizer. Fixed diagnostic atom union, not full-prior density; off-observation captured mass unknown. Contours are density levels, not probability regions. Limits from forward samples only; no density smoothing or empirical tuning.'))


if __name__ == '__main__':
    main()
