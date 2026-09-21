#!/usr/bin/env python
"""Sample-based marginals of the selected predictive law on diagnostic atoms.

Each atom has equal prior mass, multiplied by cached p(U|atom,g). Forward
draws are selected by the actual measured cuts, then the restricted mixture
is normalized. It is NOT a posterior-weighted predictive distribution, the
full24m predictive law, or a likelihood as a function of shear.
"""
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
import torch

from scripts.plot_disk_joint_likelihood import compact_response
from scripts.probe_disk_curvature import load_runtime
from scripts.probe_disk_derivative_reference import shear_context64
from sbsi.disk_inference_store import file_hash, write_json
from sbsi.disk_response_transport import disk_response_transport
from sbsi.measurement_model import load_measurement_model
from sbsi.models import V36_LIKE


ROWS = (412689, 120835, 86325)
COLORS = ('#187b9b', '#df7c20')


def selected_weights(probability, atom_index, selected, draws_per_atom):
    unnormalized = np.asarray(probability)[atom_index]/draws_per_atom
    retained = unnormalized[selected]
    if not len(retained) or not np.isfinite(retained).all() or retained.sum() <= 0:
        raise ValueError('positive selected mass required')
    return retained/retained.sum(), float(retained.sum()/len(probability))


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(quantiles, cumulative/cumulative[-1], values[order])


def contour_levels(hist, masses=(.95, .68)):
    """Uniform-bin density thresholds enclosing requested total sample mass."""
    ordered = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    if cumulative[-1] < max(masses):
        raise ValueError('plot window omits too much mass for requested contours')
    return np.unique([ordered[min(np.searchsorted(cumulative, q), len(ordered)-1)] for q in masses])


def coordinates(physical):
    result = np.asarray(physical).copy()
    result[..., 2] *= .2
    result[..., 3] = 30-2.5*np.log10(result[..., 3])
    return result


def plot_corner(output, record, samples, weights):
    observed = np.array(record['observed_plot_coordinates'])
    labels = [r'Measured $e_1$', r'Measured $e_2$', 'Radius [arcsec]', 'MAG_AUTO']
    edges = []
    for k in range(4):
        quantiles = np.array([weighted_quantile(x[:, k], w, [.001, .999]) for x, w in zip(samples, weights)])
        low, high = min(quantiles[:, 0].min(), observed[k]), max(quantiles[:, 1].max(), observed[k])
        margin = max((high-low)*.04, 1e-6)
        edges.append(np.linspace(low-margin, high+margin, 65))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
    fig.subplots_adjust(left=.085, right=.98, bottom=.13, top=.87, wspace=.11, hspace=.12)
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
                for node, (x, w) in enumerate(zip(samples, weights)):
                    hist, _ = np.histogram(x[:, i], bins=edges[i], weights=w)
                    ax.stairs(hist/np.diff(edges[i]), edges[i], color=COLORS[node],
                              lw=1.4, linestyle='-' if node == 0 else '--')
                ax.axvline(observed[i], color='#cc2870', lw=1.6)
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
            else:
                ax.set_ylim(edges[i][0], edges[i][-1])
                cx = (edges[j][1:]+edges[j][:-1])/2
                cy = (edges[i][1:]+edges[i][:-1])/2
                for node, (x, w) in enumerate(zip(samples, weights)):
                    hist, _, _ = np.histogram2d(x[:, j], x[:, i], bins=(edges[j], edges[i]), weights=w)
                    levels = contour_levels(hist)
                    if node == 0:
                        ax.pcolormesh(edges[j], edges[i], hist.T, cmap='Blues',
                                      shading='flat', rasterized=True)
                    ax.contour(cx, cy, hist.T, levels=levels, colors=COLORS[node],
                               linestyles='-' if node == 0 else '--', linewidths=1.2)
                    enclosed.append(dict(pair=[j, i], node=node, window_mass=float(hist.sum()),
                        levels=levels.tolist(), mass_above_levels=[float(hist[hist >= v].sum()) for v in levels]))
                ax.axvline(observed[j], color='#cc2870', lw=.7, alpha=.65)
                ax.axhline(observed[i], color='#cc2870', lw=.7, alpha=.65)
                ax.scatter([observed[j]], [observed[i]], marker='+', color='#cc2870', s=80, linewidth=1.8, zorder=10)
            if i == 3:
                ax.set_xlabel(labels[j])
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0:
                ax.tick_params(labelleft=False)
    g = record['shears'][0]
    fig.suptitle(f'V3.6-like marginalized corner — observation {record["row"]}\n'
        f'MAG_AUTO = {observed[3]:.3f}; radius = {observed[2]:.3f} arcsec; '
        f'e = ({observed[0]:.4f}, {observed[1]:.4f})\n'
        f'{record["n_atoms"]} diagnostic atoms; center top-32 account for '
        f'{record["center_top32_point_mass"]:.1%} of likelihood at the observation', fontsize=13, y=.985)
    fig.text(.59, .845, 'Diagonal: 1D marginal density\nLower triangle: 2D marginals\nContours: approximately 68% and 95%\n\n'
        'Original prior × usability weights\nMeasured cuts applied to forward draws\nNo posterior atom reweighting\nNo KDE or contour smoothing\n\n'
        'Conditional on the retained atom subset,\nnot the full 24-million-atom population.', fontsize=11,
        ha='left', va='top', linespacing=1.55)
    handles = [Line2D([], [], color=COLORS[0], label=f'g₀ = ({g[0]:.5f}, {g[1]:.5f})'),
        Line2D([], [], color=COLORS[1], ls='--', label='g₀ + (0.001, 0)'),
        Line2D([], [], color='#cc2870', marker='+', ls='', markersize=9, label='Observed')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .065), ncol=3, frameon=False)
    fig.text(.5, .018, 'True marginals over unplotted outputs, unlike the previous fixed-output slices.\n'
        'Atoms were chosen using this observation; these are diagnostic plots, not goodness-of-fit tests.',
        ha='center', fontsize=10)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'row_{record["row"]}_corner.{extension}', dpi=170, facecolor='white')
    plt.close(fig)
    return enclosed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 1:
        raise ValueError('new output and positive draws required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    likelihood, proposal, mock, prepared = load_runtime(args.run)
    del proposal
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device='cuda')
    flow.model.double()
    center = np.array(prepared['identity']['center'])
    points = np.array([center, center+[.001, 0.]])
    records = []
    print('CORNER_RUNTIME_READY', time.monotonic()-start, flush=True)
    for row in ROWS:
        matches = list((args.run/'tail_exact_16617850').glob(f'worker_*/row_{row}_exact.json'))
        if len(matches) != 1:
            raise ValueError('one prior exact result required')
        source = matches[0]
        reference = json.loads(source.read_text())
        if reference['identity']['preparation_sha256'] != file_hash(args.run/'disk_assembled_v1/manifest.json'):
            raise ValueError('reference preparation differs')
        exact = reference['exact']
        atoms = np.unique(np.array(exact['top_atoms']).ravel())
        response = compact_response(likelihood.disk_response, atoms)
        raw = torch.tensor(likelihood.cache.zero_flow.iloc[atoms].to_numpy(float), device=flow.device)
        observed = mock.measurements.iloc[row][list(flow.target_transform.target_names)].to_numpy(float)
        record = dict(row=row, n_atoms=len(atoms), atoms=atoms.tolist(), shears=points.tolist(),
            observed_physical=observed.tolist(), observed_plot_coordinates=coordinates(observed).tolist(),
            center_top32_point_mass=float(np.sum(exact['top_posterior_weights'][0])),
            reference=str(source), reference_sha256=file_hash(source), nodes=[])
        samples, weights, arrays = [], [], {}
        with torch.no_grad():
            for node, g in enumerate(points):
                context = shear_context64(raw, torch.tensor(g, device=flow.device), flow.condition_preprocessor)
                probability = likelihood.cache.get(*g).detection_probability[atoms]
                draws = []
                for local, atom in enumerate(atoms):
                    # CRN at both shear states for each original atom identity.
                    torch.manual_seed((20260925+int(atom)) % 2**32)
                    physical = flow.model.sample_physical(context[local:local+1], n_samples=args.draws)[0]
                    values = physical.cpu().numpy()
                    b, _ = response.predict(np.full(args.draws, local, dtype=np.int64), values[:, 2], values[:, 3])
                    physical[:, :2], _ = disk_response_transport(physical[:, :2],
                        torch.tensor(b, device=flow.device)[:, None]*physical.new_tensor(g))
                    draws.append(physical.cpu().numpy())
                physical = np.concatenate(draws)
                valid = np.isfinite(physical).all(1) & ((physical[:, :2]**2).sum(1) < 1) & (physical[:, 2:] > 0).all(1)
                if not valid.all():
                    raise ValueError('invalid forward samples; refusing silent drops')
                selected = (physical[:, 2] > 3.) & (physical[:, 3] > 10**((30-25.8)/2.5))
                indices = np.repeat(np.arange(len(atoms)), args.draws)
                w, selected_mass = selected_weights(probability, indices, selected, args.draws)
                x = coordinates(physical[selected])
                samples.append(x)
                weights.append(w)
                arrays[f'samples_{node}'], arrays[f'weights_{node}'] = x, w
                record['nodes'].append(dict(n_drawn=len(physical), n_selected=int(selected.sum()),
                    restricted_prior_selected_mass=selected_mass,
                    weight_ess=float(1/np.sum(w*w)),
                    observed_marginal_cdf=[float(w[x[:, k] <= coordinates(observed)[k]].sum()) for k in range(4)]))
                print(f'CORNER_DRAWS row={row} node={node} selected={selected.sum()} seconds={time.monotonic()-start:.1f}', flush=True)
        record['histogram_contours'] = plot_corner(args.output, record, samples, weights)
        path = args.output/f'row_{row}_draws.npz'
        np.savez_compressed(path, **arrays, atoms=atoms, observed=coordinates(observed), shears=points)
        record['samples_sha256'] = file_hash(path)
        records.append(record)
        print(f'CORNER_ROW_COMPLETE row={row} seconds={time.monotonic()-start:.1f}', flush=True)
    write_json(args.output/'report.json', dict(status='diagnostic_only', rows=records,
        draws_per_atom=args.draws, script_sha256=file_hash(__file__),
        preparation_sha256=file_hash(args.run/'disk_assembled_v1/manifest.json'),
        seconds=time.monotonic()-start,
        selection='Worst production curvature row in 19-19.5 and 20-20.3, plus faintest in exact worst32; all previously inspected.',
        interpretation='Finite-sample marginals of normalized selected predictive law on a data-chosen fixed atom subset; uniform prior times usability, not posterior weights. Full24m predictive coverage unknown. Existing cuts unchanged. No KDE. Full FP64 flow/input shear; cached FP32 usability. Top32 point mass comes from original full24m likelihood evaluation and is NOT integrated predictive coverage.'))


if __name__ == '__main__':
    main()
