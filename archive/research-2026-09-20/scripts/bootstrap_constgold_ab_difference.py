"""Paired case bootstrap on the ConstGold ``m`` difference between two models.

``evaluate_constgold_fixed_g0_response`` bootstraps each model's ``m`` on its
own, but the two arms of an A/B share cases, flow and antithetic latents, so
their errors are strongly correlated and the difference is far better
determined than either endpoint.  This re-bootstraps the same case units
*paired*, from the per-case sufficient statistics the evaluation already
writes, and prints both endpoints and their difference.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result', required=True,
                    help='result.json written by evaluate_constgold_fixed_g0_response')
parser.add_argument('--replicates', type=int, default=20000)
parser.add_argument('--seed', type=int, default=20260915)
parser.add_argument('--baseline', default='reject_baseline')
parser.add_argument('--variant', default='reject_relaxed')
args = parser.parse_args()

with open(args.result) as handle:
    d = json.load(handle)
h = d['h']
meas = d['per_case_sufficient']['measured']
pred = d['per_case_sufficient']['predicted']
cases = sorted(meas, key=int)


def response(entry):
    """R11 from one case's plus/minus sufficient statistics."""
    p, m = entry['plus'], entry['minus']
    return (p['numerator'][0] / p['denominator'] - m['numerator'][0] / m['denominator']) / (2 * h)


def weights(entry):
    return 0.5 * (entry['plus']['denominator'] + entry['minus']['denominator'])


for branch in ('actual_usable_flags', 'matched_usable', 'modeled_usable'):
    w = np.array([weights(meas[c][branch]) for c in cases])
    rm = np.array([response(meas[c][branch]) for c in cases])
    rp = {k: np.array([response(pred[k][c][branch]) for c in cases]) for k in pred}

    def m_of(idx, k):
        return 100.0 * (np.average(rm[idx], weights=w[idx])
                        / np.average(rp[k][idx], weights=w[idx]) - 1.0)

    full = np.arange(len(cases))
    point = {k: m_of(full, k) for k in rp}
    dpoint = point[args.variant] - point[args.baseline]

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(cases), size=(args.replicates, len(cases)))
    dm = np.empty(args.replicates)
    mb = np.empty(args.replicates)
    mr = np.empty(args.replicates)
    for i, idx in enumerate(draws):
        b = m_of(idx, args.baseline)
        r = m_of(idx, args.variant)
        mb[i], mr[i], dm[i] = b, r, r - b
    print('%-20s baseline m=%+.4f+/-%.4f   relaxed m=%+.4f+/-%.4f   delta=%+.4f+/-%.4f  [%.4f, %.4f]'
          % (branch, point[args.baseline], mb.std(ddof=1),
             point[args.variant], mr.std(ddof=1),
             dpoint, dm.std(ddof=1), *np.percentile(dm, [2.5, 97.5])))
