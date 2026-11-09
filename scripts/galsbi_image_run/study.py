"""GalSBI image paired run (rule.md, rule.json next to this file): units, per-leg score estimate, paired m/c.

usage: python study.py units c1            -> c1/units.txt (+ id lists), both legs at g = 0
       python study.py estimate c1 <leg>   -> c1/estimate_<leg>.json (blind; truth not read)
       python study.py units c2            -> c2/units.txt: measured mag < 21 per leg at that leg's c1 shear
       python study.py estimate c2 <leg>   -> c2/estimate_<leg>.json, pergal_<leg>.npz
       python study.py pair                -> paired_estimate.json (truth read here only)
Units are 'label leg file first last g1 g2'; seeds are base + line number.
Unit size: as large as the stage's slots allow (one unit per slot), capped by rule.json unit_size_caps.
A sampler step costs nearly the same for 40 or 640 galaxies (batch-size test, 2026-11-09), so bigger
units are faster; the cap keeps a unit within a worker's time limit.

Estimator (as galsbi_constgold100k_signsym_20261027/estimate.py, unpaired per leg):
  L_i(g) = log mean of the sampler's co-moving ratios K_i(g)/K_i(g0) at g0 +- 0.002 e_a,
  s_ia = [L_i(g0+d e_a) - L_i(g0-d e_a)]/(2d) - dlogZ_a, Z from this run's selection replicas (same model, cuts).
  E_i = (s_h1 - s_h2)(s_h1 - s_h2)^T/4 from chains 1-2 vs 3-4; I_i = s_i s_i^T - E_i.
  c1: g = I^-1 sum s_i(0). c2: mag < 21 rerun at the rounded c1 shear g1; t_i = s_i(g1) + I_i g1; g = I^-1 sum t.
Pairing (as galsbi_constgold1m_minus_signsym_20261029/pair.py): m1 = (g1+ - g1-)/0.04 - 1, c1 = (g1+ + g1-)/2,
  c2 = (g2+ + g2-)/2, R21 = (g2+ - g2-)/0.04; errors = largest of the case bootstrap, galaxy influence and
  chain floor, each plus the shared selection term.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

H = Path(__file__).resolve().parent
LEGS = ('plus', 'minus')
D = .002
NODES = [[0., 0.], [-D, 0.], [D, 0.], [0., -D], [0., D]]
BANDS = (('<21', -np.inf, 21), ('21-23', 21, 23), ('23-24', 23, 24), ('24-25', 24, 25))
MPS_CLIENTS = 48  # CUDA MPS limit of client processes per GPU (Volta and newer)


def worker_slots(resources):
    """Sampler slots per GPU worker, as worker.sh computes them: min(slots_per_worker, cpus - 2, 48)."""
    return min(resources['slots_per_worker'], resources['cpus_per_worker']-2, MPS_CLIENTS)
rule = json.loads((H/'rule.json').read_text())
lm = lambda x: logsumexp(x)-np.log(x.size)
two = lambda x: np.column_stack((x[:, 1]-x[:, 0], x[:, 3]-x[:, 2]))/(2*D)
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reference(leg):
    return H/'reference'/leg


def measured(leg):
    ref = pd.read_parquet(reference(leg)/'observations.parquet')
    mag = 30-2.5*np.log10(ref['measured_flux_from_mag_auto'].to_numpy())
    return ref, mag


def write_units(stage):
    out = H/stage
    out.mkdir(exist_ok=True)
    if (out/'units.txt').exists():
        raise SystemExit(f'{stage}/units.txt exists')
    slots = worker_slots(rule['resources'])*rule['resources'][f'{stage}_workers']
    lines = []
    for kind in (('bright', 'faint') if stage == 'c1' else ('bright21',)):
        legs = {}
        for leg in LEGS:
            _, mag = measured(leg)
            if stage == 'c1':
                ids = np.flatnonzero(mag < 23 if kind == 'bright' else mag >= 23)
                g = (0., 0.)
            else:
                ids = np.flatnonzero(mag < 21)
                g = tuple(np.round(json.loads((H/'c1'/f'estimate_{leg}.json').read_text())['g'], 5))
            legs[leg] = ids, g
        # one unit per slot at most (each leg's last unit may be short, hence slots - legs)
        total = sum(len(ids) for ids, _ in legs.values())
        size = min(rule['unit_size_caps'][kind], max(1, -(-total//(slots-len(LEGS)))))
        count = 0
        for leg, (ids, g) in legs.items():
            name = f'{leg}_{kind}_ids.txt'
            np.savetxt(out/name, ids, fmt='%d')
            for k, first in enumerate(range(0, len(ids), size)):
                last = min(first+size, len(ids))-1
                lines.append(f'{leg}_{kind}{k:04d} {leg} {stage}/{name} {first} {last} {g[0]:.5f} {g[1]:.5f}')
                count += 1
        print(f'{stage} {kind}: {total} galaxies, {size} per unit, {count} units for {slots} slots')
    (out/'units.txt').write_text('\n'.join(lines)+'\n')
    print(stage, len(lines), 'units')


def readings(stage, leg):
    """{obs: (L, h1, h2)} (4 values each: g1-, g1+, g2-, g2+) of one leg; refuses unless every unit is done."""
    R = H/stage
    units = [l.split() for l in (R/'units.txt').read_text().splitlines()]
    units = [u for u in units if u[1] == leg]
    missing = [u[0] for u in units if not (R/'claims'/u[0]/'done').exists()]
    if missing:
        raise SystemExit(f'{stage} {leg}: {len(missing)} of {len(units)} units not done (e.g. {missing[:3]})')
    identity = json.loads((reference(leg)/'protocol.json').read_text())['model_identity']
    out, stuck = {}, {}
    for label, _, file, first, last, g1, g2 in units:
        g0 = [float(g1), float(g2)]
        names = [f'g{a+1}_{round(g0[a]+s*D, 12):g}' for a in (0, 1) for s in (-1, 1)]
        run = R/label
        protocol = json.loads((run/'protocol.json').read_text())
        if protocol['target_coordinates'][1] != rule['magnitude_coordinate']:
            raise ValueError(f'{label} used magnitude coordinate {protocol["target_coordinates"][1]}')
        if protocol['model_identity'] != identity:
            raise ValueError('sampler model identity differs from the reference')
        if not np.allclose(protocol['fixed_shear'], g0, atol=1e-12, rtol=0):
            raise ValueError(f'{label} sampled at {protocol["fixed_shear"]}, expected {g0}')
        if protocol['reference'] != str(reference(leg).resolve()):
            raise ValueError(f'{label} sampled on {protocol["reference"]}')
        planned = np.loadtxt(H/file, dtype=int, ndmin=1)[int(first):int(last)+1]
        if sorted(protocol['observations']) != sorted(planned.tolist()):
            raise ValueError(f'{label} observations differ from units.txt')
        result = json.loads((run/'result.json').read_text())
        stuck.update({r['reference_observation']: r['diagnostics']['log_z']['rhat'] for r in result['observations']})
        with np.load(run/'local_profiles.npz') as z:
            for o in protocol['observations']:
                r = [z[f'observation_{o}_{n}'] for n in names]
                if o in out:
                    raise ValueError(f'galaxy {o} sampled twice')
                out[o] = ([lm(x) for x in r], [lm(x[:, :2]) for x in r], [lm(x[:, 2:]) for x in r])
    return out, stuck


def selection():
    """Chunk values (chunks, 5 nodes) of the selection replicas, after checking plan, model and receipts."""
    R = H/'normalization'
    plan = json.loads((R/'plan.json').read_text())
    expect = dict(model='like-1.0-galsbi', shear_nodes=NODES, magnitude_limits=[25.], max_radius_pixels=100.,
                  max_measured_ellipticity=.99)
    bad = {k: plan.get(k) for k, v in expect.items() if plan.get(k) != v}
    if bad:
        raise ValueError(f'selection plan differs: {bad}')
    for leg in LEGS:
        p = json.loads((reference(leg)/'protocol.json').read_text())
        c = p['cuts']
        if (c['magnitude_limit'], c['min_radius_pixels'], c['max_radius_pixels'], c['max_measured_ellipticity']) != (
                25., 3., 100., .99):
            raise ValueError('reference cuts differ from the normalization cuts')
    chunks = []
    for r in range(plan['selection_replicas']):
        s = R/f'selection{r}'
        receipt = json.loads((s/'result.json').read_text())
        if (receipt['model'] != plan['model'] or receipt['chunks'] != plan['selection_chunks']
                or receipt['data_sha256'] != digest(s/'chunks.npz') or receipt['plan_sha256'] != digest(R/'plan.json')
                or receipt['seed_base'] != plan['selection_seed_base']+r*10000
                or receipt['reference_protocol_sha256'] != digest(reference('plus')/'protocol.json')):
            raise ValueError(f'selection{r} receipt does not match its plan/reference')
        with np.load(s/'chunks.npz') as z:
            if not np.allclose(z['grid'], NODES):
                raise ValueError('selection nodes differ')
            chunks.append(z['values'][:, :, 0, 1])
    return np.concatenate(chunks)


def errors(v, e, Ii, unit, inv, sel, rng):
    E = np.einsum('ia,ib->ab', e, e)/4
    cov = dict(stat=inv, chains=inv@E@inv, selection=inv@sel@inv)
    out = {k: np.sqrt(np.diag(c)) for k, c in cov.items()}
    out['total'] = np.sqrt(np.diag(sum(cov.values())))
    n = len(v)
    gal = [np.linalg.solve(Ii[k].sum(0), v[k].sum(0)) for k in (rng.integers(0, n, n) for _ in range(2000))]
    Uu, Iu = np.zeros((unit.max()+1, 2)), np.zeros((unit.max()+1, 2, 2))
    np.add.at(Uu, unit, v)
    np.add.at(Iu, unit, Ii)
    nc = len(Uu)
    case = [np.linalg.solve(Iu[k].sum(0), Uu[k].sum(0)) for k in (rng.integers(0, nc, nc) for _ in range(2000))]
    out['bootstrap_galaxies'] = np.hypot(np.std(gal, 0), out['selection'])
    out['bootstrap_cases'] = np.hypot(np.std(case, 0), out['selection'])
    out['quoted'] = np.max([out['total'], out['bootstrap_galaxies'], out['bootstrap_cases']], 0)
    return {k: x.tolist() for k, x in out.items()}


def estimate(stage, leg):
    ref, magnitude = measured(leg)
    r0, stuck = readings('c1', leg)
    if set(r0) != set(range(len(ref))):
        raise ValueError(f'c1 does not cover the {leg} reference exactly once')
    obs = np.arange(len(ref))
    mag = magnitude[obs]
    case = pd.read_parquet(reference(leg)/'truth.parquet', columns=['source_case'])['source_case'].to_numpy()[obs]
    unit = np.unique(case, return_inverse=True)[1]
    L, h1, h2 = (np.array([r0[o][k] for o in obs]) for k in (0, 1, 2))
    z = selection()
    logz = np.log(z.mean(0))
    dlogz = np.array([logz[2]-logz[1], logz[4]-logz[3]])/(2*D)
    n = len(obs)
    sel = np.diag((n*((np.log(z[:, [2, 4]])-np.log(z[:, [1, 3]]))/(2*D)).std(0, ddof=1)/np.sqrt(len(z)))**2)
    s0, e0 = two(L)-dlogz, two(h1)-two(h2)
    Ii = np.einsum('ia,ib->iab', s0, s0)-np.einsum('ia,ib->iab', e0, e0)/4
    I = Ii.sum(0)
    inv = np.linalg.inv(I)
    out = dict(stage=stage, leg=leg, galaxies=n, cases=int(unit.max()+1), selection_chunks=len(z),
               selection_score_per_galaxy=dlogz.tolist(), information=I.tolist(),
               stuck_rhat_log_z_ge_1p1=int(sum((stuck[o] or 0) >= 1.1 for o in obs)), definitions=__doc__)
    v, e = s0, e0
    if stage == 'c2':
        g1 = np.round(json.loads((H/'c1'/f'estimate_{leg}.json').read_text())['g'], 5)
        r1, stuck1 = readings('c2', leg)
        bright = mag < 21
        if set(r1) != set(obs[bright].tolist()):
            raise ValueError('c2 does not cover the mag < 21 galaxies exactly once')
        L1, a1, b1 = (np.array([r1[o][k] for o in obs[bright]]) for k in (0, 1, 2))
        v, e = s0.copy(), e0.copy()
        v[bright] = two(L1)-dlogz+Ii[bright]@g1
        e[bright] = two(a1)-two(b1)
        out.update(c1_shear=g1.tolist(), bright_rerun=int(bright.sum()),
                   bright_score_at_0=s0[bright].sum(0).tolist(), bright_score_at_g1=(two(L1)-dlogz).sum(0).tolist(),
                   bright_line_prediction=(s0[bright].sum(0)-Ii[bright].sum(0)@g1).tolist(),
                   c2_stuck_rhat_log_z_ge_1p1=int(sum((x or 0) >= 1.1 for x in stuck1.values())))
    out['g'] = (inv@v.sum(0)).tolist()
    out['errors'] = errors(v, e, Ii, unit, inv, sel, np.random.default_rng(rule['bootstrap_seed']))
    out['bands'] = {}
    for name, lo, hi in BANDS:
        k = (mag >= lo) & (mag < hi)
        out['bands'][name] = dict(galaxies=int(k.sum()), score_sum=v[k].sum(0).tolist(),
            score_stat_sd=np.sqrt(np.diag(Ii[k].sum(0))).tolist(),
            score_chain_sd=np.sqrt(np.einsum('ia,ia->a', e[k], e[k])/4).tolist(),
            information_g1g1=float(Ii[k, 0, 0].sum()))
    (H/stage/f'estimate_{leg}.json').write_text(json.dumps(out, indent=1))
    np.savez(H/stage/f'pergal_{leg}.npz', obs=obs, case=case, mag=mag, v=v, e=e, Ii=Ii, sel=sel)
    g, err = np.array(out['g']), out['errors']
    print(f'{stage} {leg}: {n} galaxies in {out["cases"]} cases; stuck (R-hat log Z >= 1.1) '
          f'{out["stuck_rhat_log_z_ge_1p1"]}; selection score/galaxy {np.round(dlogz, 3)}')
    for a in (0, 1):
        print(f'  g{a+1} = {g[a]:+.5f} +- {err["quoted"][a]:.5f} (stat {err["stat"][a]:.5f}, chains '
              f'{err["chains"][a]:.5f}, selection {err["selection"][a]:.5f}; bootstrap galaxies '
              f'{err["bootstrap_galaxies"][a]:.5f}, cases {err["bootstrap_cases"][a]:.5f})')
    for name, b in out['bands'].items():
        print(f'  {name:6s} n={b["galaxies"]:6d} S = {np.round(b["score_sum"], 1)} +- {np.round(b["score_stat_sd"], 1)}'
              f' (stat) +- {np.round(b["score_chain_sd"], 1)} (chains)')


PARAMETER_MAP = np.array([[25., 0., -25., 0.], [.5, 0., .5, 0.], [0., .5, 0., .5], [0., 25., 0., -25.]])


def pair():
    cases = np.array(rule['cases'])
    records, galaxy, case_data, chain_cov, response = [], [], [], [], []
    for leg, injected in (('plus', .02), ('minus', -.02)):
        result = json.loads((H/'c2'/f'estimate_{leg}.json').read_text())
        truth = pd.read_parquet(reference(leg)/'truth.parquet')
        assert np.allclose(truth.injected_g1, injected, atol=1e-14, rtol=0)
        assert np.allclose(truth.injected_g2, 0, atol=1e-14, rtol=0)
        with np.load(H/'c2'/f'pergal_{leg}.npz') as z:
            obs, v, ii, e = z['obs'], z['v'], z['Ii'], z['e']
            inv = np.linalg.inv(ii.sum(0))
            g = inv@v.sum(0)
            np.testing.assert_allclose(g, result['g'], atol=1e-12, rtol=0)
            influence = (v-np.einsum('nij,j->ni', ii, g))@inv.T
            ids = truth.iloc[obs][['source_case', 'source_input_index']].reset_index(drop=True)
            for a in range(2):
                ids[leg+str(a)] = influence[:, a]
            galaxy.append(ids.set_index(['source_case', 'source_input_index']))
            u, info = np.zeros((len(cases), 2)), np.zeros((len(cases), 2, 2))
            index = np.searchsorted(cases, z['case'])
            assert np.array_equal(cases[index], z['case'])
            np.add.at(u, index, v)
            np.add.at(info, index, ii)
            case_data.append((u, info))
            chain_cov.append(inv@(e.T@e/4)@inv)
            response.append(-len(obs)*inv)
            records.append(dict(leg=leg, galaxies=len(obs), g=g.tolist(), c1_flags=result['stuck_rhat_log_z_ge_1p1'],
                                c2_flags=result['c2_stuck_rhat_log_z_ge_1p1']))
    joint = galaxy[0].join(galaxy[1], how='outer').fillna(0).to_numpy()
    matched = len(galaxy[0].index.intersection(galaxy[1].index))
    rng = np.random.default_rng(rule['bootstrap_seed'])
    samples = []
    for _ in range(2000):
        k = rng.integers(0, len(cases), len(cases))
        samples.append(np.concatenate([np.linalg.solve(info[k].sum(0), u[k].sum(0)) for u, info in case_data]))
    z = selection()
    slopes = (np.log(z[:, [2, 4]])-np.log(z[:, [1, 3]]))/(2*D)
    cov_z = np.cov(slopes, rowvar=False, ddof=1)/len(z)
    response = np.concatenate(response)
    selection_cov = response@cov_z@response.T
    chain = np.zeros((4, 4))
    chain[:2, :2], chain[2:, 2:] = chain_cov
    covs = {name: PARAMETER_MAP@c@PARAMETER_MAP.T for name, c in (
        ('paired_case_plus_selection', np.cov(np.array(samples), rowvar=False, ddof=1)+selection_cov),
        ('paired_galaxy_influence_plus_selection', joint.T@joint+selection_cov),
        ('independent_chain_plus_selection_floor', chain+selection_cov))}
    errs = {k: np.sqrt(np.maximum(np.diag(c), 0)).tolist() for k, c in covs.items()}
    quoted = np.max(np.array(list(errs.values())), axis=0)
    values = PARAMETER_MAP@np.concatenate([r['g'] for r in records])
    values[0] -= 1
    result = dict(status='completed_paired_pilot', legs=records, parameters=['m1', 'c1', 'c2', 'g1_to_g2_response'],
        values=values.tolist(), quoted_errors=quoted.tolist(), errors=errs,
        covariance={k: c.tolist() for k, c in covs.items()}, matched_selected=matched,
        plus_only_selected=len(galaxy[0])-matched, minus_only_selected=len(galaxy[1])-matched,
        conventions='m1=(g1_plus-g1_minus)/.04-1; c1=(g1_plus+g1_minus)/2; c2=(g2_plus+g2_minus)/2; '
                    'cross response=(g2_plus-g2_minus)/.04',
        limitations=['Known-shear development images (constant_g002), not independent acceptance.',
                     'Legs share the noise field per case; the paired errors include that correlation.',
                     'Chain covariance is a floor, not added twice; no finite-chain bias bound.'])
    (H/'paired_estimate.json').write_text(json.dumps(result, indent=2)+'\n')
    sd = 1 if quoted[0] == 0 else quoted[0]
    print(f'PAIRED m1 = {100*values[0]:+.2f}% +- {100*quoted[0]:.2f}% ({values[0]/sd:+.1f} sigma); '
          f'c1 = {values[1]:+.5f} +- {quoted[1]:.5f}; c2 = {values[2]:+.5f} +- {quoted[2]:.5f} '
          f'({values[2]/quoted[2]:+.1f} sigma); R21 = {values[3]:+.4f} +- {quoted[3]:.4f}; '
          f'matched {matched}, plus-only {len(galaxy[0])-matched}, minus-only {len(galaxy[1])-matched}')


if __name__ == '__main__':
    command = sys.argv[1:]
    if command[:1] == ['units'] and command[1:] in (['c1'], ['c2']):
        write_units(command[1])
    elif command[:1] == ['estimate'] and len(command) == 3 and command[1] in ('c1', 'c2') and command[2] in LEGS:
        estimate(command[1], command[2])
    elif command == ['pair']:
        pair()
    else:
        raise SystemExit(__doc__)
