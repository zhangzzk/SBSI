#!/usr/bin/env python
"""Stronger-guard test for the GalSBI-dev F8 flow (rule: $DATA_DIR/sbsi_caches/galsbi_guard_strength_20261103).

  train --steps N   rerun only the guard phase of the frozen recipe (scripts.train_full_domain_guard_flow) from the
                    saved GalSBI density-only checkpoint with GUARD_STEPS_PER_EPOCH = N; every other constant, seed
                    and the selection rule are unchanged. Output <root>/steps<N>/.
  map               shear response per measured (radius, magnitude) bin, simulation and flows (validation cases)
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


def train(steps, no_global=False):
    trainer = frozen()
    out = ROOT/(f'steps{steps}' + ('_noglobal' if no_global else ''))
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
    print(f'GUARD_STRENGTH steps_per_epoch={steps} no_global={no_global} out={out}', flush=True)
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
        train(a.steps, a.no_global)
    elif a.stage == 'map':
        response_map({k: known.get(k, ROOT/k/'guard/selected.pt') for k in a.flows}, a.tag)
    else:
        extra = [tuple(float(x) if x else None for x in e.split(',')) for e in a.extra]
        check({k: known.get(k, ROOT/k/'guard/selected.pt') for k in a.flows}, a.tag, extra)


if __name__ == '__main__':
    main()
