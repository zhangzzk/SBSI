#!/usr/bin/env python
"""Stronger-guard test for the GalSBI-dev F8 flow (rule: $DATA_DIR/sbsi_caches/galsbi_guard_strength_20261103).

  train --steps N   rerun only the guard phase of the frozen recipe (scripts.train_full_domain_guard_flow) from the
                    saved GalSBI density-only checkpoint with GUARD_STEPS_PER_EPOCH = N; every other constant, seed
                    and the selection rule are unchanged. Output <root>/steps<N>/.
  train --bands     the same with the guard on soft windows tiling measured radius x magnitude (make_bands);
                    output <root>/bands<N>/
  train --and-only  the same with the guard on the 9 joint cuts only (radius 2.8 / 3.0 / 3.2 AND magnitude
                    AND_MAGNITUDE); output <root>/steps<N>_and25/
  train --grid      the September grid response term (6x6x6 truth cells, every NLL step, 12 cells x 128 pairs x 16
                    draws, cell error scale) with each cell's response taken through the soft measured GRID_CUT on
                    both legs; output <root>/grid_cut/
  map            shear response per measured (radius, magnitude) bin, simulation and flows (validation cases)
  check             response mismatch of each flow on all matched pairs of the 40 validation cases, at the 16
                    guarded cuts and 15 held-out cuts (plus any --extra cuts, reported but outside the held-out
                    RMS), plus validation NLL; per-case sums, case bootstrap.

Runs under the frozen F8 venv (python -I, cwd /tmp) in a Slurm GPU allocation, like run_galsbi_f8_stage.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

DATA = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
ROOT = DATA/'galsbi_guard_strength_20261103'
EXPERIMENT = DATA/'galsbi_f8c8_20261102'
SOURCE = EXPERIMENT/'base/flow'
DOMAIN = EXPERIMENT/'base/domain'
DRAWS, BLOCK, CHECK_SEED = 16, 2048, 614_000_000_001
BOOTSTRAP, BOOTSTRAP_SEED = 10_000, 20260925
# Held-out cuts (never in training): (radius_min px, magnitude_max)
HELD_OUT = ((2.5, None), (3.5, None), (3.75, None), (4.0, None), (5.0, None),
            (None, 24.5), (None, 25.0), (None, 25.4),
            (2.5, 25.8), (3.5, 25.8), (3.75, 25.8), (4.0, 25.8), (3.0, 25.0), (3.75, 25.0), (3.0, 25.4))
MAIN = 'radius_gt_3_magnitude_lt_25.8'
# Band guard (rule addendum "band guard"): soft windows tiling measured FLUX_RADIUS x MAG_AUTO, open at both ends
BAND_RADIUS = tuple(float(x) for x in np.round(np.arange(2.2, 5.001, 0.2), 2))
BAND_MAGNITUDE = (24.0, 24.6, 25.0, 25.4, 25.8, 26.4)
BAND_TOLERANCE = 0.003   # band scale: 0.3% of the main-cut response, divided by the band's share of the main-cut mass
MAIN_RESPONSE = 0.6035   # simulated main-cut (R11 + R22)/2 per galaxy, validation cases (check_mag25.json)
# AND-only guard (rule addendum "AND-only guard"): magnitude limits around the target MAG_AUTO < 25
AND_MAGNITUDE = (24.8, 25.0, 25.2)
# Grid arm (rule addendum "grid with measured cut"): truth cells, soft measured cut inside the response term only
GRID_CUT = {'name': 'radius_gt_3_magnitude_lt_25', 'radius_min': 3.0, 'magnitude_max': 25.0}
GRID_AXES = {'r_input_p': 3, 'circularized_Re_input_p': 4, 'nbr_close_100': 6}   # context_raw columns
GRID_BINS = 6
GRID_CELLS_PER_STEP, GRID_PAIRS_PER_CELL, GRID_DRAWS = 12, 128, 16
GRID_MIN_MASS, GRID_MIN_CASES = 200.0, 8   # summed soft pass weight of the g = 0 leg; cases with any pass weight
GRID_SEED = 615_000_000_001


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen():
    """Check the frozen checkout against the GalSBI experiment pins and return the trainer module."""
    experiment = json.loads((EXPERIMENT/'experiment.json').read_text())
    checkout = Path(experiment['checkout'])
    for name, expected in experiment['checkout_hashes'].items():
        if sha256(checkout/name) != expected:
            raise ValueError(f'frozen source changed: {name}')
    import sbsi
    import scripts.train_full_domain_guard_flow as trainer
    if Path(sbsi.__file__).resolve().parent != checkout/'sbsi' or \
            Path(trainer.__file__).resolve().parents[1] != checkout:
        raise ValueError(f'wrong installed SBSI: {sbsi.__file__}')
    return trainer


def windows(edges):
    bounds = [None, *edges, None]
    return list(zip(bounds[:-1], bounds[1:]))


def make_bands():
    return tuple({'name': f'band_r{rl}-{rh}_m{ml}-{mh}', 'radius_min': rl, 'radius_max': rh,
                  'magnitude_min': ml, 'magnitude_max': mh}
                 for rl, rh in windows(BAND_RADIUS) for ml, mh in windows(BAND_MAGNITUDE))


def allow_bands():
    """Let the frozen guard module take two-sided windows (extra keys radius_max, magnitude_min).

    A window [lo, hi) is sigmoid((x - lo)/s) - sigmoid((x - hi)/s), so adjacent windows sum exactly; one-sided cuts
    keep the frozen expressions. Only the band training calls this.
    """
    import torch
    import sbsi.flow_guard_response as guard
    plain = ('name', 'radius_min', 'magnitude_max')
    original = guard._validate_cuts

    def validate(cuts):
        cuts = tuple(dict(c) for c in cuts)
        original([{k: c[k] for k in plain} for c in cuts])
        for c in cuts:
            if not set(c) - set(plain) <= {'radius_max', 'magnitude_min'}:
                raise ValueError(f'unknown guard keys: {sorted(c)}')
        return cuts

    def window(sigmoid, x, lo, hi, softness):
        if hi is None:
            return sigmoid((x - float(lo)) / softness)
        if lo is None:
            return sigmoid((float(hi) - x) / softness)
        return sigmoid((x - float(lo)) / softness) - sigmoid((x - float(hi)) / softness)

    def weights(values, cuts, radius, magnitude, ones, sigmoid, radius_softness, magnitude_softness):
        result = []
        for c in validate(cuts):
            w = ones
            for x, lo, hi, s in ((radius, c['radius_min'], c.get('radius_max'), radius_softness),
                                 (magnitude, c.get('magnitude_min'), c['magnitude_max'], magnitude_softness)):
                if lo is not None or hi is not None:
                    w = w * window(sigmoid, x, lo, hi, s)
            result.append(w)
        return result

    def torch_weights(values, cuts, *, radius_softness, magnitude_softness, zero_point=30.0):
        flux = values[..., 3].clamp_min(torch.finfo(values.dtype).tiny)
        magnitude = float(zero_point) - 2.5 * torch.log10(flux)
        return torch.stack(weights(values, cuts, values[..., 2], magnitude, torch.ones_like(flux), torch.sigmoid,
                                   radius_softness, magnitude_softness), dim=-1)

    def numpy_weights(values, cuts, *, radius_softness, magnitude_softness, zero_point=30.0):
        values = np.asarray(values, dtype=np.float64)
        magnitude = float(zero_point) - 2.5 * np.log10(values[:, 3])
        return np.stack(weights(values, cuts, values[:, 2], magnitude, np.ones(len(values)), guard._numpy_sigmoid,
                                radius_softness, magnitude_softness), axis=-1)

    guard._validate_cuts, guard.torch_guard_weights, guard.numpy_guard_weights = validate, torch_weights, numpy_weights


def band_scales(trainer, bands):
    """BAND_TOLERANCE x MAIN_RESPONSE x D_main / D_band: the loss then weighs each band by its contribution to a
    cut's response. Masses from the zero-shear rows of the training cases, with the same soft windows."""
    import sbsi.flow_guard_response as guard
    cases = json.loads((DOMAIN/'manifest.json').read_text())['split']['train_cases']
    rows = np.concatenate([np.load(DOMAIN/f'flow/g0/case{int(c):03d}.npz')['target'] for c in cases])
    kw = dict(radius_softness=trainer.RADIUS_SOFTNESS, magnitude_softness=trainer.MAGNITUDE_SOFTNESS)
    main = {'name': 'main', 'radius_min': 3.0, 'magnitude_max': 25.8}
    mass = np.concatenate([guard.numpy_guard_weights(rows[i:i + 1_000_000], (main,) + bands, **kw).sum(0)[None]
                           for i in range(0, len(rows), 1_000_000)]).sum(0)
    scales = BAND_TOLERANCE * MAIN_RESPONSE * mass[0] / mass[1:]
    return scales, {'rows': int(len(rows)), 'main_mass': float(mass[0] / len(rows)),
                    'band_mass': (mass[1:] / len(rows)).tolist(), 'scales': scales.tolist()}


def grid_edges(features, weight):
    """Per-axis quantile edges weighted by the g = 0 pass weight, open ends (tails go to the end bins)."""
    edges = []
    for x in features.T:
        order = np.argsort(x)
        cumulative = np.cumsum(weight[order])
        inner = x[order][np.searchsorted(cumulative / cumulative[-1], np.arange(1, GRID_BINS) / GRID_BINS)]
        edges.append(np.concatenate([[-np.inf], inner, [np.inf]]))
    return np.asarray(edges)


def grid_cells(features, edges):
    index = [np.searchsorted(e[1:-1], x, side='right') for e, x in zip(edges, features.T)]
    return np.ravel_multi_index(index, (GRID_BINS,) * len(edges))


def grid_sums(data, edges, kw):
    """Per case and cell sums from which the soft-cut response and its case jackknife follow exactly:
    N, sum w0, sum w0 e0 (2), sum (w1 e1 - w0 e0) gamma^T (2x2), sum (w1 - w0) gamma (2), sum gamma gamma^T (2x2)."""
    import sbsi.flow_guard_response as guard
    zero, shear, gamma = (np.asarray(data[k], dtype=np.float64)
                          for k in ('pair_zero_target', 'pair_shear_target', 'gamma'))
    cell = grid_cells(np.asarray(data['context_raw'])[data['pairs'][:, 0]][:, list(GRID_AXES.values())], edges)
    cases, case = np.unique(data['pair_case'], return_inverse=True)
    w0, w1 = (guard.numpy_guard_weights(v, (GRID_CUT,), **kw)[:, 0] for v in (zero, shear))
    columns = np.column_stack([np.ones(len(zero)), w0, w0[:, None] * zero[:, :2],
                               np.einsum('no,ni->noi', w1[:, None] * shear[:, :2] - w0[:, None] * zero[:, :2],
                                         gamma).reshape(-1, 4),
                               (w1 - w0)[:, None] * gamma, np.einsum('no,ni->noi', gamma, gamma).reshape(-1, 4)])
    key = case * GRID_BINS ** 3 + cell
    sums = np.stack([np.bincount(key, c, minlength=len(cases) * GRID_BINS ** 3) for c in columns.T], axis=-1)
    return sums.reshape(len(cases), GRID_BINS ** 3, -1), cell


def grid_response(s):
    """(..., 14) sums -> (response (..., 4) as R11 R12 R21 R22, pass mass D, pass mean mu (..., 2), inverse normal)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mass = s[..., 1] / s[..., 0]
        mean = s[..., 2:4] / s[..., 1:2]
        cross = (s[..., 4:8].reshape(*s.shape[:-1], 2, 2) - mean[..., :, None] * s[..., None, 8:10]) / mass[..., None, None]
        normal = s[..., 10:14].reshape(*s.shape[:-1], 2, 2)
        inverse = np.linalg.inv(np.where(np.abs(np.linalg.det(normal))[..., None, None] > 0, normal, np.eye(2)))
        return (cross @ inverse).reshape(*s.shape[:-1], 4), mass, mean, inverse


class GridCutPopulation:
    """Soft-cut guard response of each truth cell: v = w(y) (e - mu_B) / D_B, R_B = sum_B (v1 - v0) gamma^T G_B^-1."""

    def __init__(self, context, data, edges, kw, scale=None, allowed=None):
        import torch
        sums, cell = grid_sums(data, edges, kw)
        total = sums.sum(0)
        response, mass, mean, inverse = grid_response(total)
        deleted, *_ = grid_response(total[None] - sums)
        n_cases = len(sums)
        error = np.sqrt((n_cases - 1) / n_cases * ((deleted - deleted.mean(0)) ** 2).sum(0))
        scale = error if scale is None else scale
        normal = total[:, 10:14].reshape(-1, 2, 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            condition = np.linalg.cond(normal)
        reasons = {'mass': total[:, 1] < GRID_MIN_MASS, 'cases': (sums[..., 1] > 0).sum(0) < GRID_MIN_CASES,
                   'design': ~(condition < 1e8),
                   'nonfinite': ~(np.isfinite(response).all(1) & np.isfinite(scale).all(1) & (scale > 0).all(1))}
        if allowed is not None:   # validation scores the training cells only
            reasons['not_in_training'] = ~np.isin(np.arange(GRID_BINS ** 3), allowed)
        keep = ~np.any(list(reasons.values()), axis=0)
        self.cells = np.flatnonzero(keep)
        slot = np.full(GRID_BINS ** 3, -1)
        slot[self.cells] = np.arange(len(self.cells))
        pair_slot = slot[cell]
        device = context.device
        self.context, self.kw = context, kw
        self.pairs = torch.as_tensor(data['pairs'], dtype=torch.long, device=device)
        self.pair_slot = torch.as_tensor(pair_slot, device=device)
        self.members = [torch.as_tensor(np.flatnonzero(pair_slot == k), device=device) for k in range(len(self.cells))]
        as64 = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)
        self.target, self.response_scale = as64(response[keep]), as64(scale[keep])
        self.mass, self.mean = as64(mass[keep]), as64(mean[keep])
        design = np.zeros((len(cell), 2))
        inside = pair_slot >= 0
        design[inside] = np.einsum('ni,nij->nj', np.asarray(data['gamma'], dtype=np.float64)[inside],
                                   inverse[cell[inside]]) * total[cell[inside], 0:1]
        self.design = as64(design)
        self.scale_all = scale
        self.statistics = {
            'cut': GRID_CUT, 'axes': list(GRID_AXES), 'inner_edges': edges[:, 1:-1].tolist(), 'cells_kept': self.cells.tolist(),
            'cells_dropped': {k: np.flatnonzero(v).tolist() for k, v in reasons.items()},
            'pairs_in_kept_cells': int(inside.sum()), 'pairs': int(len(cell)), 'cases': int(n_cases),
            'pairs_per_cell': total[keep, 0].tolist(), 'pass_mass': total[keep, 1].tolist(),
            'target': response[keep].tolist(), 'scale': scale[keep].tolist(), 'shape_normal': None}

    def contributions(self, model, indices, draws, seed):
        """Per pair (batch, 4): its cell's response numerator, scaled so the cell mean over its pairs is R_B."""
        import torch
        import sbsi.flow_guard_response as guard
        pair, slot = self.pairs[indices], self.pair_slot[indices]
        legs = guard.physical_context_draws(model, [self.context[pair[:, 0]], self.context[pair[:, 1]]], draws, seed)
        influence = []
        for values in legs:
            w = guard.torch_guard_weights(values, (GRID_CUT,), **self.kw)[..., 0]
            influence.append((w[..., None] * (values[..., :2] - self.mean[slot][:, None]) / self.mass[slot][:, None, None])
                             .mean(dim=1))
        return torch.einsum('bo,bi->boi', influence[1] - influence[0], self.design[indices]).reshape(len(indices), 4)


def grid_backward(model, population, seed, weight):
    """Independent-group cross score over GRID_CELLS_PER_STEP random cells (as the September grid term), replayed."""
    import torch
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(population.cells), GRID_CELLS_PER_STEP, replace=False)
    groups = [torch.cat([m[torch.as_tensor(rng.integers(len(m), size=GRID_PAIRS_PER_CELL), device=m.device)]
                         for m in (population.members[c] for c in chosen)]) for _ in range(2)]
    target, scale = population.target[chosen], population.response_scale[chosen]
    cell_mean = lambda v: v.reshape(GRID_CELLS_PER_STEP, GRID_PAIRS_PER_CELL, 4).mean(dim=1)
    with torch.no_grad():
        means = [cell_mean(population.contributions(model, g, GRID_DRAWS, seed + 100_000_000 * s))
                 for s, g in enumerate(groups)]
    score = 0.5 * ((means[0] - target) * (means[1] - target) / scale ** 2).mean()
    if not bool(torch.isfinite(score)):
        raise ValueError('nonfinite grid score')
    for s, g in enumerate(groups):
        coefficient = 0.5 * (means[1 - s] - target) / scale ** 2 / target.numel()
        ((cell_mean(population.contributions(model, g, GRID_DRAWS, seed + 100_000_000 * s)) * coefficient).sum()
         * weight).backward()
    return float(score)


def grid_evaluate(model, population, indices, *, draws, groups, batch_size, seed):
    """Fixed validation pairs, independent latent groups; cross score averaged over cells with pairs."""
    import torch
    with torch.no_grad():
        idx = torch.as_tensor(indices, dtype=torch.long, device=population.context.device)
        idx = idx[population.pair_slot[idx] >= 0]
        n = len(population.cells)
        count = torch.bincount(population.pair_slot[idx], minlength=n).double()
        means = []
        for group in range(groups):
            total = torch.zeros(n, 4, dtype=torch.float64, device=idx.device)
            for block, start in enumerate(range(0, len(idx), batch_size)):
                rows = idx[start:start + batch_size]
                total.index_add_(0, population.pair_slot[rows],
                                 population.contributions(model, rows, draws, seed + group * 1_000_000_000 + block).double())
            means.append(total / count.clamp(min=1)[:, None])
        scored = count > 0
        z = [((m - population.target) / population.response_scale)[scored] for m in means]
        loss = torch.stack([0.5 * (z[a] * z[b]).mean() for a in range(groups) for b in range(a + 1, groups)]).mean()
        mean = torch.stack(means).mean(0)
    return {'loss': float(loss), 'cells_scored': int(scored.sum()), 'pairs': int(len(idx)), 'draws': int(draws),
            'groups': int(groups), 'mean_residual': (mean - population.target).cpu().numpy().tolist(),
            'group_means': None}


def use_grid(trainer, out):
    """Swap the frozen guard phase for the grid term: populations, a combined NLL + grid step, evaluation."""
    base = trainer.base
    loaded, nll_pass, made = [], base.nll_pass, {}

    def load_split(*a, **k):
        loaded.append(base_load(*a, **k))
        return loaded[-1]

    def population(context, pairs, *a, radius_softness, magnitude_softness, **k):
        data = next(d for d in loaded if d['pairs'] is pairs)
        kw = dict(radius_softness=radius_softness, magnitude_softness=magnitude_softness)
        if 'train' not in made:   # the trainer builds the training population first
            p = data['pairs'][:, 0]
            features = np.asarray(data['context_raw'])[p][:, list(GRID_AXES.values())]
            import sbsi.flow_guard_response as guard
            weight = guard.numpy_guard_weights(np.asarray(data['pair_zero_target'], dtype=np.float64),
                                               (GRID_CUT,), **kw)[:, 0]
            made['edges'] = grid_edges(features, weight)
            made['train'] = GridCutPopulation(context, data, made['edges'], kw)
            pop = made['train']
        else:   # validation: training edges and training error scale
            pop = GridCutPopulation(context, data, made['edges'], kw, scale=made['train'].scale_all,
                                    allowed=made['train'].cells)
            made['validation'] = pop
        print(f"GRID_POPULATION cells_kept={len(pop.cells)} pairs_in_kept={pop.statistics['pairs_in_kept_cells']:,}/"
              f"{pop.statistics['pairs']:,} dropped={ {k: len(v) for k, v in pop.statistics['cells_dropped'].items()} }",
              flush=True)
        return pop

    def combined_pass(model, values, context, *, optimizer=None, seed=None):
        if optimizer is None:
            return nll_pass(model, values, context)
        import torch
        model.train()
        generator = torch.Generator(device=values.device).manual_seed(int(seed))
        order = torch.randperm(len(values), generator=generator, device=values.device)
        nll_total, scores, norms = 0.0, [], []
        for step, rows in enumerate(order.split(base.BATCH_SIZE)):
            optimizer.zero_grad(set_to_none=True)
            nll = -model.log_prob(values[rows], context[rows]).mean()
            if not bool(torch.isfinite(nll)):
                raise ValueError('non-finite grid-stage NLL')
            nll.backward()
            scores.append(grid_backward(model, made['train'], GRID_SEED + int(seed) * 1_000_003 + step * 100_003,
                                        trainer.GUARD_WEIGHT))
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.GRADIENT_CLIP,
                                                              error_if_nonfinite=True)))
            optimizer.step()
            nll_total += float(nll.detach()) * len(rows)
        return {'nll': nll_total / len(values), 'grid_loss_mean': float(np.mean(scores)), 'rows': int(len(values)),
                'optimizer_steps': len(norms), 'gradient_norm_mean': float(np.mean(norms)),
                'clipped_step_fraction': float(np.mean(np.asarray(norms) > trainer.GRADIENT_CLIP))}

    if not (out/'nll_completed.json').exists():
        raise ValueError('grid arm starts from the copied density checkpoint only')
    base_load = base.load_split
    base.load_split, base.nll_pass = load_split, combined_pass
    trainer.make_guard_cuts = lambda r, m: (GRID_CUT,)
    trainer.GuardResponsePopulation = population
    trainer.guard_training_epoch = lambda *a, **k: {'steps': 0, 'note': 'grid term is inside every NLL step'}
    trainer.evaluate_guard_response = grid_evaluate
    trainer.GUARD_STEPS_PER_EPOCH = 0
    return made


def train(steps, no_global=False, bands=False, and_only=False, grid=False):
    trainer = frozen()
    out = ROOT/'grid_cut' if grid else ROOT/(f'{"bands" if bands else "steps"}{steps}'
                                             + ('_noglobal' if no_global else '') + ('_and25' if and_only else ''))
    if not (out/'nll_completed.json').exists():
        (out/'nll').mkdir(parents=True, exist_ok=True)
        for name in ('selected.pt', 'selected_optimizer.pt', 'history.json'):
            shutil.copy2(SOURCE/'nll'/name, out/'nll'/name)
        receipt = json.loads((SOURCE/'nll_completed.json').read_text())
        if sha256(out/'nll/selected.pt') != receipt['selected_sha256']:
            raise ValueError('copied density checkpoint differs from its receipt')
        shutil.copy2(SOURCE/'nll_completed.json', out/'nll_completed.json')
    trainer.GUARD_STEPS_PER_EPOCH = steps
    if no_global:   # the trainer builds its cuts once, through this module-level name
        all_cuts = trainer.make_guard_cuts
        trainer.make_guard_cuts = lambda r, m: tuple(c for c in all_cuts(r, m) if c['name'] != 'global')
    if and_only:   # keep only the radius-and-magnitude cuts, at the AND_MAGNITUDE limits
        all_cuts = trainer.make_guard_cuts
        trainer.make_guard_cuts = lambda r, m: tuple(c for c in all_cuts(r, AND_MAGNITUDE)
                                                     if c['radius_min'] is not None and c['magnitude_max'] is not None)
    if bands:   # the trainer builds cuts and scales once, through these module-level names
        allow_bands()
        band_cuts = make_bands()
        scales, record = band_scales(trainer, band_cuts)
        (out/'band_scales.json').write_text(json.dumps({'bands': list(band_cuts), **record}, indent=1) + '\n')
        trainer.make_guard_cuts = lambda r, m: band_cuts
        trainer.guard_scales = lambda cuts: scales
    if grid:
        use_grid(trainer, out)
    print(f'GUARD_STRENGTH steps_per_epoch={trainer.GUARD_STEPS_PER_EPOCH} no_global={no_global} bands={bands} '
          f'and_only={and_only} grid={grid} out={out}', flush=True)
    trainer.train(argparse.Namespace(domain_root=DOMAIN, output_root=out, resume=True))


def cut_list(trainer, extra=()):
    from sbsi.flow_guard_response import make_guard_cuts
    guarded = make_guard_cuts(trainer.RADIUS_THRESHOLDS, trainer.MAGNITUDE_THRESHOLDS)
    held = tuple({'name': '_'.join(([f'radius_gt_{r:g}'] if r else []) + ([f'magnitude_lt_{m:g}'] if m else [])),
                  'radius_min': r, 'magnitude_max': m} for r, m in HELD_OUT + tuple(extra))
    return guarded + held, len(guarded)


def check(flows, tag='', extra=()):
    import pandas as pd
    import torch
    from sbsi.fixed_g0_domain import FLOW_FEATURES
    from sbsi.flow_guard_response import GuardResponsePopulation, torch_guard_weights
    from sbsi.measurement_model import load_measurement_model
    trainer = frozen()
    base = trainer.base
    device = torch.device('cuda')
    manifest = json.loads((DOMAIN/'manifest.json').read_text())
    cases = manifest['split']['validation_cases']
    data = base.load_split(DOMAIN, cases)
    cuts, n_guarded = cut_list(trainer, extra)
    n_held = n_guarded + len(HELD_OUT)
    case_of = {c: i for i, c in enumerate(cases)}
    pair_case = torch.as_tensor(np.vectorize(case_of.get)(data['pair_case']), device=device)
    row_case = np.concatenate([np.full(m[f'{leg}_rows'], case_of[m['case']]) for m in data['per_case_matching']
                               for leg in ('left', 'right')])
    if len(row_case) != len(data['target_raw']):
        raise ValueError('row-to-case map does not cover the validation rows')
    row_case = torch.as_tensor(row_case, device=device)
    n_pairs = len(data['pairs'])
    result = {'cases': cases, 'pairs': n_pairs, 'rows': len(row_case), 'draws': DRAWS, 'seed': CHECK_SEED,
              'cuts': list(cuts), 'guarded': n_guarded, 'held_out': len(HELD_OUT), 'flows': {}}
    sums = {}
    sim = None
    for name, path in flows.items():
        bundle = load_measurement_model(path, device=device)
        model = bundle.model.eval()
        context = bundle.context_tensor(pd.DataFrame(data['context_raw'], columns=FLOW_FEATURES))
        population = GuardResponsePopulation(context, data['pairs'], data['pair_zero_target'],
                                             data['pair_shear_target'], data['gamma'], cuts,
                                             radius_softness=trainer.RADIUS_SOFTNESS,
                                             magnitude_softness=trainer.MAGNITUDE_SOFTNESS, response_scale=1.0)
        model_sum = torch.zeros(len(cases), len(cuts), 4, dtype=torch.float64, device=device)
        if sim is None:   # measured side, identical for every flow
            measured = [torch.as_tensor(data[k], device=device) for k in ('pair_zero_target', 'pair_shear_target')]
            sim = torch.zeros_like(model_sum)
            count = torch.zeros(len(cases), dtype=torch.float64, device=device)
            for start in range(0, n_pairs, BLOCK):
                rows = torch.arange(start, min(n_pairs, start + BLOCK), device=device)
                influence = []
                for values in measured:
                    v = values[rows]
                    w = torch_guard_weights(v, population.cuts, radius_softness=population.radius_softness,
                                            magnitude_softness=population.magnitude_softness,
                                            zero_point=population.zero_point)
                    influence.append(w[..., None] * (v[:, None, :2] - population.baseline_mean[None])
                                     / population.baseline_mass[None, :, None])
                part = torch.einsum('bco,bi->bcoi', influence[1] - influence[0], population.design[rows])
                sim.index_add_(0, pair_case[rows], part.reshape(len(rows), len(cuts), 4))
                count.index_add_(0, pair_case[rows], torch.ones(len(rows), dtype=torch.float64, device=device))
            if not torch.allclose(sim.sum(0) / n_pairs, population.target, rtol=1e-9, atol=1e-12):
                raise ValueError('per-case measured sums do not reproduce the frozen guard target')
        with torch.no_grad():
            for block, start in enumerate(range(0, n_pairs, BLOCK)):
                rows = torch.arange(start, min(n_pairs, start + BLOCK), device=device)
                model_sum.index_add_(0, pair_case[rows], population.contributions(model, rows, DRAWS,
                                                                                   CHECK_SEED + block))
            values = torch.as_tensor(bundle.target_transform.transform_array(data['target_raw']),
                                     dtype=torch.float64, device=device)
            nll = torch.zeros(len(cases), dtype=torch.float64, device=device)
            for rows in torch.arange(len(values), device=device).split(8192):
                nll.index_add_(0, row_case[rows], -model.log_prob(values[rows], context[rows]).double())
        sums[name] = (model_sum.cpu().numpy(), nll.cpu().numpy())
        result['flows'][name] = {'path': str(path), 'sha256': sha256(path)}
        print(f'CHECK_FLOW_DONE {name}', flush=True)
        del bundle, model, context, population
        torch.cuda.empty_cache()
    sim, count = sim.cpu().numpy(), count.cpu().numpy()
    rows_per_case = np.bincount(row_case.cpu().numpy(), minlength=len(cases)).astype(float)
    np.savez(ROOT/f'check{tag}_sums.npz',sim=sim, count=count, rows=rows_per_case,
             **{f'model_{k}': v[0] for k, v in sums.items()}, **{f'nll_{k}': v[1] for k, v in sums.items()})
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(len(cases), np.full(len(cases), 1 / len(cases)),
                                                                size=BOOTSTRAP).astype(float)

    def m_eq(model_sum, w):   # (..., case, cut, 4) -> (..., cut); ratio of summed diagonals, pairs cancel
        t = np.einsum('...c,ckj->...kj', w, sim)
        m = np.einsum('...c,ckj->...kj', w, model_sum)
        return 100 * ((t[..., 0] + t[..., 3]) / (m[..., 0] + m[..., 3]) - 1)

    def nll_mean(n, w):
        return (w @ n) / (w @ rows_per_case)

    ones = np.ones(len(cases))
    point = {k: m_eq(v[0], ones) for k, v in sums.items()}
    boot = {k: m_eq(v[0], weights) for k, v in sums.items()}
    nll_point = {k: nll_mean(v[1], ones) for k, v in sums.items()}
    nll_boot = {k: nll_mean(v[1], weights) for k, v in sums.items()}
    result['target_diagonal'] = ((sim.sum(0)[:, 0] + sim.sum(0)[:, 3]) / 2 / count.sum()).tolist()
    for k in sums:
        result['flows'][k].update(
            m_eq=point[k].tolist(), m_eq_se=boot[k].std(0, ddof=1).tolist(),
            nll=float(nll_point[k]), nll_se=float(nll_boot[k].std(ddof=1)),
            rms_m_eq_guarded=float(np.sqrt(np.mean(point[k][:n_guarded] ** 2))),
            rms_m_eq_held_out=float(np.sqrt(np.mean(point[k][n_guarded:n_held] ** 2))))
        if 'steps4' in sums and k != 'steps4':
            result['flows'][k]['minus_steps4'] = {
                'm_eq': (point[k] - point['steps4']).tolist(),
                'm_eq_se': (boot[k] - boot['steps4']).std(0, ddof=1).tolist(),
                'abs_m_eq': (np.abs(point[k]) - np.abs(point['steps4'])).tolist(),
                'abs_m_eq_se': (np.abs(boot[k]) - np.abs(boot['steps4'])).std(0, ddof=1).tolist(),
                'nll': float(nll_point[k] - nll_point['steps4']),
                'nll_se': float((nll_boot[k] - nll_boot['steps4']).std(ddof=1))}
    (ROOT/f'check{tag}.json').write_text(json.dumps(result, indent=1) + '\n')
    names = [c['name'] for c in cuts]
    print(f"{'cut':38s}" + ''.join(f'{k:>20s}' for k in sums))
    for i, n in enumerate(names):
        tag = '' if i < n_guarded else ' (held out)' if i < n_held else ' (extra)'
        print(f'{n + tag:38s}' + ''.join(f"{point[k][i]:+8.2f} +- {boot[k][:, i].std(ddof=1):5.2f}  " for k in sums))
    for k in sums:
        f = result['flows'][k]
        print(f"{k}: NLL {f['nll']:.5f} +- {f['nll_se']:.5f}  RMS m_eq guarded {f['rms_m_eq_guarded']:.2f}"
              f"  held out {f['rms_m_eq_held_out']:.2f}")


R_EDGES, M_EDGES = np.round(np.arange(1.0, 6.001, 0.1), 2), np.round(np.arange(20.0, 27.001, 0.25), 2)


def response_map(flows, tag=''):
    """Shear response per measured (FLUX_RADIUS, MAG_AUTO) bin, simulation and flows, per validation case.

    Hard bins (np.digitize on R_EDGES x M_EDGES, under/overflow kept), each leg binned by its own measurement. Per
    case and bin: leg counts, the e response sum_pairs (1[bin(shear)] e_shear - 1[bin(zero)] e_zero) x design and
    the count response sum_pairs (1[bin(shear)] - 1[bin(zero)]) x design (design as the guard: gamma (G'G)^-1 N).
    Any hard cut's response follows by summing bins. Flows: 16 antithetic CRN draws per pair, the check's seeds.
    """
    import torch
    from sbsi.flow_guard_response import physical_context_draws
    from sbsi.measurement_model import load_measurement_model
    import pandas as pd
    from sbsi.fixed_g0_domain import FLOW_FEATURES
    trainer = frozen()
    device = torch.device('cuda')
    cases = json.loads((DOMAIN/'manifest.json').read_text())['split']['validation_cases']
    data = trainer.base.load_split(DOMAIN, cases)
    case_of = {c: i for i, c in enumerate(cases)}
    pair_case = torch.as_tensor(np.vectorize(case_of.get)(data['pair_case']), device=device)
    gamma = data['gamma']
    design = torch.as_tensor(gamma @ np.linalg.inv(gamma.T @ gamma) * len(gamma), device=device)
    pairs = torch.as_tensor(data['pairs'], device=device)
    n_pairs, nc = len(pairs), len(cases)
    r_edges, m_edges = (torch.as_tensor(e, device=device) for e in (R_EDGES, M_EDGES))
    nb = (len(R_EDGES) + 1) * (len(M_EDGES) + 1)

    def bins(values):   # (..., 4) physical -> flat bin index
        mag = 30 - 2.5 * torch.log10(values[..., 3].clamp_min(1e-30))
        return torch.bucketize(values[..., 2].contiguous(), r_edges, right=True) * (len(M_EDGES) + 1) + \
            torch.bucketize(mag.contiguous(), m_edges, right=True)

    def accumulate(legs, rows, weight):
        """legs: [(..., 4) zero, (..., 4) shear] with leading (batch[, draws]); adds into acc with this weight."""
        d = design[rows]
        c = pair_case[rows]
        for sign, (leg, values) in zip((-1, 1), enumerate(legs)):
            b = bins(values)
            flat_case = c.view(-1, *([1] * (b.ndim - 1))).expand_as(b)
            idx = (flat_case * nb + b).reshape(-1)
            dd = d.view(len(rows), *([1] * (b.ndim - 1)), 2).expand(*b.shape, 2).reshape(-1, 2)
            e = values[..., :2].reshape(-1, 2)
            acc['count'][leg].index_add_(0, idx, torch.full((len(idx),), weight, dtype=torch.float64, device=device))
            acc['shape'].index_add_(0, idx, sign * weight * torch.einsum('no,ni->noi', e, dd))
            acc['mass'].index_add_(0, idx, sign * weight * dd)

    out = {}
    for name, path in [('sim', None)] + list(flows.items()):
        acc = {'count': [torch.zeros(nc * nb, dtype=torch.float64, device=device) for _ in range(2)],
               'shape': torch.zeros(nc * nb, 2, 2, dtype=torch.float64, device=device),
               'mass': torch.zeros(nc * nb, 2, dtype=torch.float64, device=device)}
        if path is None:
            measured = [torch.as_tensor(data[k], device=device) for k in ('pair_zero_target', 'pair_shear_target')]
            for start in range(0, n_pairs, BLOCK):
                rows = torch.arange(start, min(n_pairs, start + BLOCK), device=device)
                accumulate([m[rows] for m in measured], rows, 1.0)
        else:
            bundle = load_measurement_model(path, device=device)
            model = bundle.model.eval()
            context = bundle.context_tensor(pd.DataFrame(data['context_raw'], columns=FLOW_FEATURES))
            with torch.no_grad():
                for block, start in enumerate(range(0, n_pairs, BLOCK)):
                    rows = torch.arange(start, min(n_pairs, start + BLOCK), device=device)
                    p = pairs[rows]
                    legs = physical_context_draws(model, [context[p[:, 0]], context[p[:, 1]]], DRAWS,
                                                  CHECK_SEED + block)
                    accumulate([v.double() for v in legs], rows, 1.0 / DRAWS)
            del bundle, model, context
            torch.cuda.empty_cache()
        shape = (nc, len(R_EDGES) + 1, len(M_EDGES) + 1)
        out[f'{name}_count'] = torch.stack(acc['count']).reshape(2, *shape).transpose(0, 1).cpu().numpy()
        out[f'{name}_shape'] = acc['shape'].reshape(*shape, 2, 2).cpu().numpy()
        out[f'{name}_mass'] = acc['mass'].reshape(*shape, 2).cpu().numpy()
        print(f'MAP_DONE {name}', flush=True)
    np.savez(ROOT/f'map{tag}.npz', r_edges=R_EDGES, m_edges=M_EDGES, cases=np.asarray(cases), pairs=n_pairs, **out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    mp = sub.add_parser('map')
    mp.add_argument('--flows', nargs='+', default=['nll_only', 'steps4', 'steps40'])
    mp.add_argument('--tag', default='')
    t = sub.add_parser('train')
    t.add_argument('--steps', type=int, required=True)
    t.add_argument('--no-global', action='store_true', help='drop the global (uncut) term from the guard')
    t.add_argument('--bands', action='store_true', help='guard on the band tiling instead of the 16 cuts')
    t.add_argument('--and-only', action='store_true', help='guard on the 9 radius-and-AND_MAGNITUDE cuts only')
    t.add_argument('--grid', action='store_true', help='grid response term with the measured GRID_CUT (--steps ignored)')
    c = sub.add_parser('check')
    c.add_argument('--flows', nargs='+', default=['nll_only', 'steps4', 'steps4_noglobal', 'steps40'])
    c.add_argument('--extra', nargs='+', default=[], metavar='RADIUS,MAG',
                   help='extra cuts outside the held-out RMS, e.g. 2.8,25 (empty side = no limit)')
    c.add_argument('--tag', default='', help='output suffix: check<tag>.json, check<tag>_sums.npz')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('scheduler allocation required')
    known = {'nll_only': SOURCE/'nll/selected.pt', 'steps4': SOURCE/'guard/selected.pt'}
    if a.stage == 'train':
        train(a.steps, a.no_global, a.bands, a.and_only, a.grid)
    elif a.stage == 'map':
        response_map({k: known.get(k, ROOT/k/'guard/selected.pt') for k in a.flows}, a.tag)
    else:
        extra = [tuple(float(x) if x else None for x in e.split(',')) for e in a.extra]
        check({k: known.get(k, ROOT/k/'guard/selected.pt') for k in a.flows}, a.tag, extra)


if __name__ == '__main__':
    main()
