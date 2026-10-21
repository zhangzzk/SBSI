#!/usr/bin/env python
"""Predeclared n20 CPU comparison; whole independent populations are replicates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, ndtr

from sbsi.annealed_particles import annealed_smc
from sbsi.waste_free_particles import waste_free_smc


class Target:
    def __init__(self, name):
        self.name = name
        if name == 'gaussian':
            self.means = np.array([[1., -.5]])
            self.weights = np.ones(1)
            self.cov = np.array([[1., .92], [.92, 1.5]])
            self.qscale = np.array([3., 3.])
            self.z = 2.5
            self.step = .7*np.linalg.cholesky(self.cov)
        else:
            self.means = np.array([[-4., 0.], [4., 0.]])
            self.weights = np.array([.3, .7])
            self.cov = np.diag(np.array([.65, .9])**2)
            self.qscale = np.array([5., 2.])
            self.z = 1.7
            self.step = np.diag([.52, .72])
        self.inverse = np.linalg.inv(self.cov)
        self.logdet = np.linalg.slogdet(self.cov)[1]
        v = np.diag(self.cov)
        mean = self.weights@self.means
        second = self.weights@(self.means**2+v)
        fourth = self.weights@(self.means**4+6*self.means**2*v+3*v**2)
        region = self.weights@ndtr(self.means[:, 0]/np.sqrt(v[0]))
        self.truth = np.r_[mean, second, region]
        self.variance = np.r_[second-mean**2, fourth-second**2, region*(1-region)]

    def log_q0(self, x):
        return -np.log(2*np.pi)-np.log(self.qscale).sum()-.5*np.sum((x/self.qscale)**2, axis=1)

    def log_density(self, x):
        delta = x[:, None, :]-self.means[None, :, :]
        quadratic = np.einsum('nki,ij,nkj->nk', delta, self.inverse, delta)
        return np.log(self.z)+logsumexp(np.log(self.weights)[None, :]-.5*quadratic
            -np.log(2*np.pi)-.5*self.logdet, axis=1)


class CachedPotential:
    """Identical exact memoization in both arms; count actual target density rows."""
    def __init__(self, target):
        self.target, self.cache = target, {}
        self.rows, self.calls, self.batch_calls = 0, 0, 0

    def __call__(self, x):
        self.calls += 1
        keys = [row.tobytes() for row in x]
        missing = {}
        for i, key in enumerate(keys):
            if key not in self.cache:
                missing[key] = i
        if missing:
            batch = x[list(missing.values())]
            values = self.target.log_density(batch)-self.target.log_q0(batch)
            self.cache.update(zip(missing, values))
            self.rows += len(batch)
            self.batch_calls += 1
        return np.array([self.cache[key] for key in keys])

    def mutation(self, x, beta, rng):
        proposed = x+rng.normal(size=x.shape)@self.target.step.T
        old = self.target.log_q0(x)+beta*self(x)
        new = self.target.log_q0(proposed)+beta*self(proposed)
        accept = np.log(rng.random(len(x))) < new-old
        return np.where(accept[:, None], proposed, x)


def observable(x):
    return np.column_stack((x, x**2, x[:, 0] > 0))


def diagnostics(logz, values, target):
    z = np.exp(logz)
    weighted = z[:, None]*values
    theta = weighted.mean(0)/z.mean()
    influence = z[:, None]*(values-theta)/z.mean()
    variance = influence.var(0, ddof=1)
    zs = z.std(ddof=1)/np.sqrt(len(z))
    fs = weighted.std(0, ddof=1)/np.sqrt(len(z))
    ztest = float(abs(z.mean()-target.z)/zs)
    ftest = np.abs(weighted.mean(0)-target.z*target.truth)/fs
    return dict(mean_normalizer=float(z.mean()), normalizer_se=float(zs),
        normalizer_relative_variance=float(z.var(ddof=1)/z.mean()**2),
        normalizer_standardized_error=ztest, weighted_observable_standardized_errors=ftest.tolist(),
        unnormalized_moment_gate=bool(ztest < 5 and np.all(ftest < 5)),
        estimate=theta.tolist(), truth=target.truth.tolist(), mcse=np.sqrt(variance/len(z)).tolist(),
        ratio_influence_variance=variance.tolist(),
        primary_metric=float(np.mean(variance/target.variance)))


def metric(z, values, posterior_variance):
    theta = np.mean(z[:, None]*values, axis=0)/z.mean()
    return np.mean(np.var(z[:, None]*(values-theta)/z.mean(), axis=0, ddof=1)/posterior_variance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    files = ('scripts/check_waste_free_synthetic.py', 'sbsi/waste_free_particles.py',
             'sbsi/annealed_particles.py', 'doc/WASTE_FREE_SMC_PROTOTYPE.md')
    hashes = {n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in files}
    receipt = dict(source=str(source), sha256=hashes, settings='doc/WASTE_FREE_SMC_PROTOTYPE.md',
        populations_per_arm=128, stages=16, counted_target_density_rows=13560,
        cache='Exact per-population bytes-key memoization, identical in both arms',
        no_real_scene_integration=True, targets={})
    (args.output/'source.json').write_text(json.dumps(receipt, indent=2)+'\n')
    ratios = []
    for target_index, name in enumerate(('gaussian', 'mixture')):
        target = Target(name)
        groups = {}
        for arm_index, arm in enumerate(('standard', 'waste_free')):
            logz, values, seconds, calls, batches, ancestors = [], [], [], [], [], []
            seed_start = 2026103000+2000*target_index+1000*arm_index
            for i in range(128):
                rng = np.random.default_rng(seed_start+i)
                potential = CachedPotential(target)
                start = time.perf_counter()
                n = 120 if arm == 'standard' else 904
                initial = rng.normal(size=(n, 2))*target.qscale
                if arm == 'standard':
                    result = annealed_smc(initial, potential, (np.arange(17)/16)**2,
                        rng, mutation=potential.mutation, mutation_steps=7)
                else:
                    result = waste_free_smc(initial, potential, (np.arange(17)/16)**2,
                        rng, mutation=potential.mutation, path_length=8)
                seconds.append(time.perf_counter()-start)
                assert not result.extinct
                assert potential.rows == 13560, (name, arm, i, potential.rows)
                logz.append(result.log_normalizer)
                values.append(observable(result.particles).mean(0))
                calls.append(potential.calls)
                batches.append(potential.batch_calls)
                ancestors.append(result.stages[-1].unique_ancestors)
                if (i+1) % 32 == 0:
                    print(json.dumps(dict(target=name, arm=arm, completed=i+1)), flush=True)
            logz, values = np.array(logz), np.array(values)
            result = diagnostics(logz, values, target)
            result.update(seed_start=seed_start, seed_stop=seed_start+127, density_rows_per_population=13560,
                total_seconds=float(sum(seconds)), median_seconds=float(np.median(seconds)),
                potential_calls_per_population=int(calls[0]), density_batch_calls_per_population=int(batches[0]),
                median_initial_ancestors=float(np.median(ancestors)))
            groups[arm] = (logz, values, result)
            np.savez(args.output/f'{name}_{arm}.npz', log_normalizer=logz,
                population_observable=values, seconds=seconds, initial_ancestors=ancestors,
                seeds=np.arange(seed_start, seed_start+128))
        ratio = groups['waste_free'][2]['primary_metric']/groups['standard'][2]['primary_metric']
        rng = np.random.default_rng(2026107000+target_index)
        bootstrap = []
        for _ in range(1000):
            metrics = []
            for arm in ('standard', 'waste_free'):
                lz, f, _ = groups[arm]
                indices = rng.integers(0, 128, 128)
                metrics.append(metric(np.exp(lz[indices]), f[indices], target.variance))
            bootstrap.append(metrics[1]/metrics[0])
        comparison = dict(primary_variance_ratio_waste_free_over_standard=float(ratio),
            bootstrap_interval95=np.quantile(bootstrap, [.025, .975]).tolist(),
            observable_variance_ratios=(np.array(groups['waste_free'][2]['ratio_influence_variance'])/
                np.array(groups['standard'][2]['ratio_influence_variance'])).tolist(),
            median_walltime_ratio=groups['waste_free'][2]['median_seconds']/groups['standard'][2]['median_seconds'],
            outcome='candidate_gain' if ratio < .8 else 'regression' if ratio > 1.25 else 'inconclusive')
        receipt['targets'][name] = dict(standard=groups['standard'][2], waste_free=groups['waste_free'][2], comparison=comparison)
        ratios.append(ratio)
        print(json.dumps(dict(target=name, comparison=comparison)), flush=True)
    receipt['dev_score'] = float(1/np.sqrt(np.prod(ratios)))
    receipt['dev_score_definition'] = 'Geometric mean standard/waste-free normalized influence variance at matched counted density rows; synthetic development only'
    receipt['all_unnormalized_moment_gates_pass'] = all(receipt['targets'][name][arm]['unnormalized_moment_gate']
        for name in receipt['targets'] for arm in ('standard', 'waste_free'))
    for n, h in hashes.items():
        assert hashlib.sha256((source/n).read_bytes()).hexdigest() == h, n
    (args.output/'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(dev_score=receipt['dev_score'], moment_gates=receipt['all_unnormalized_moment_gates_pass'])), flush=True)


if __name__ == '__main__':
    main()
