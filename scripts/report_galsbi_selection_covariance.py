"""Recompute two-component, cross-band selection MC covariance from saved chunks.

No sampling, data cuts, selection means or catalogue shear estimates change.
Independent chunks are the resampling units, with CRN correlations retained.
Run larger bootstrap checks through Slurm.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sbsi.inference_moments import selection_log_contrasts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(mass, nodes, centre, step):
    bands = mass.shape[2]
    contrasts = np.zeros((2*bands, len(nodes)*bands))
    for axis in range(2):
        for sign in [-1, 1]:
            point = centre.copy(); point[axis] += sign*step
            matched = np.flatnonzero(np.all(np.isclose(nodes, point, rtol=0, atol=1e-12), axis=1))
            if len(matched) != 1:
                raise ValueError('one matching positive/negative node per shear axis required')
            for band in range(bands):
                contrasts[2*band+axis, matched[0]*bands+band] = sign/(2*step)
    result = selection_log_contrasts(mass.reshape(len(mass), -1), contrasts)
    covariance = result['covariance']
    se = np.sqrt(np.diag(covariance))
    denominator = np.outer(se, se)
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    old = None
    if (mass > 0).all():
        chunk_scores = np.log(mass.reshape(len(mass), -1))@contrasts.T
        old = (chunk_scores.std(0, ddof=1)/np.sqrt(len(mass))).reshape(bands, 2).tolist()
    return dict(gradient=result['value'].reshape(bands, 2).tolist(),
        gradient_mcse=se.reshape(bands, 2).tolist(), covariance=covariance.tolist(),
        correlation=correlation.tolist(), old_log_chunk_mcse=old,
        zero_mass_entries=int((mass == 0).sum()), independent_chunks=len(mass),
        mean_mass=mass.mean(0).tolist(), ordering='band0_g1,band0_g2,band1_g1,band1_g2,...')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    plan_path = args.root/'plan.json'
    plan = json.loads(plan_path.read_text())
    nodes = np.asarray(plan['shear_nodes'], float)
    centre = np.asarray(plan['sampling_shear'], float)
    step = float(plan['stencil'])
    limits = np.asarray(plan['magnitude_limits'], float)
    if (nodes.ndim != 2 or nodes.shape[1] != 2 or centre.shape != (2,)
            or not np.isfinite(nodes).all() or not np.isfinite(centre).all()
            or not np.isfinite(step) or not step > 0
            or limits.ndim != 1 or not len(limits) or not np.isfinite(limits).all()
            or np.any(np.diff(limits) <= 0)):
        raise ValueError('two-component nodes, centre and positive stencil required')
    values, sources, seeds = [], {}, set()
    receipts = []
    for replica in range(plan['selection_replicas']):
        directory = args.root/f'selection{replica}'
        path, receipt_path = directory/'chunks.npz', directory/'result.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt['status'] != 'completed_selection' or receipt['replica'] != replica
                or receipt['data_sha256'] != digest(path)
                or receipt['chunks'] != plan['selection_chunks']
                or receipt['scenes_per_chunk'] != plan['selection_chunk_size']
                or receipt['flow_draws'] != [plan['selection_flow_draws']//2, plan['selection_flow_draws']]
                or not np.array_equal(receipt['grid'], nodes)):
            raise ValueError(f'inconsistent selection receipt: {directory}')
        expected_seed = plan['selection_seed_base']+10000*replica
        if receipt['seed_base'] != expected_seed:
            raise ValueError('unexpected independent selection seed')
        chunk_seeds = set(expected_seed+10*np.arange(receipt['chunks']))
        if seeds & chunk_seeds:
            raise ValueError('overlapping chunk seeds cannot be counted as independent replicas')
        seeds |= chunk_seeds
        with np.load(path) as saved:
            a = saved['values']
            if (a.shape != (receipt['chunks'], len(nodes), len(plan['magnitude_limits']), 2)
                    or not np.array_equal(saved['grid'], nodes)):
                raise ValueError('selection chunk array shape or nodes disagree with receipt')
            values.append(a[:, :, :, -1])
        receipts.append(receipt)
        for file in [path, receipt_path]:
            sources[str(file)] = digest(file)
    if (len({r['reference_protocol_sha256'] for r in receipts}) != 1
            or len({r['selection'] for r in receipts}) != 1):
        raise ValueError('selection replicas must use the same reference and selection definition')
    mass = np.concatenate(values)
    if np.any(np.diff(mass, axis=2) < 0):
        raise ValueError('cumulative magnitude selections are not nested')
    disjoint = np.diff(mass, axis=2, prepend=np.zeros_like(mass[:, :, :1]))
    output = dict(centre=centre.tolist(), step=step, magnitude_limits=plan['magnitude_limits'],
        cumulative=summarize(mass, nodes, centre, step),
        disjoint=summarize(disjoint, nodes, centre, step),
        sources=sources, plan_sha256=digest(plan_path), script_sha256=digest(__file__),
        reference_protocol_sha256=receipts[0]['reference_protocol_sha256'],
        selection=receipts[0]['selection'],
        historical_receipt_plan_hashes=sorted(set(r['plan_sha256'] for r in receipts)),
        limitations=['Delta-method Monte Carlo covariance conditional on independent, identically designed chunks.',
            'Historical plan hashes may differ after replica/resource amendments; node/count/depth/seed receipts are checked explicitly.',
            'Finite-step selection derivative; no mixed selection Hessian or new shear estimate.',
            'Full scene and chain covariance are needed before a cross-band consistency test.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print('independent chunks', len(mass), 'disjoint gradient MCSE', output['disjoint']['gradient_mcse'], flush=True)


if __name__ == '__main__':
    main()
