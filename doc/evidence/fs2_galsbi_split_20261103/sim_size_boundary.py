"""Simulation-only look at the measured size cut, FS2 ConstGold (cases 40-59) vs GalSBI constant (1000-1019), g1 = +-0.02.

usage (Slurm CPU, py31): python sim_size_boundary.py
FS2 measured legs come from the saved V3.9 rows (same values the evaluation used); GalSBI legs are joined as the frozen
measured_leg does (CrossMatch -> Shapes, one-to-one; unmatched parents unusable and counted). Usable = finite,
|g| < 1, radius > 0, flux > 0. Per true-mag bin (<24, 24-25.5, 25.5-27.5, >=27.5) and for MAG_AUTO < 25.8:
  hist      measured FLUX_RADIUS histogram (px, plus leg), 0.1 px bins 1-8
  response  R11 with radius limits 0 (none), >3, >=3.75 px: total, kept-in-both-legs part, and the rest
            (selection part), each leg normalized by its own selected count (as the evaluation)
Writes sim_size_boundary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
C = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
FS2 = C/'v36_f8_20260924_v1/evaluation'
GSIMS = Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/constant_g002/sims')
GPARENT = C/'galsbi_dev_constant_20261103/parent'
H = .02
MAG4 = (24., 25.5, 27.5)
EDGES = np.arange(1., 8.01, .1)
LIMITS = ((None, 0.), ('>', 3.), ('>=', 3.75))
FLUX_MINIMUM = 10**(-0.4*(25.8-30))


def galsbi_leg(case, shear, ids):
    base = GSIMS/f'case{case}_{shear:.2f}/real0/catalogues'
    cross = pd.read_feather(base/'CrossMatch/tile180.0_-0.5_rot0_matched.feather', columns=['id_detec', 'id_input'])
    shape = pd.read_feather(base/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather',
                            columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS'])
    joined = cross.merge(shape, left_on='id_detec', right_on='NUMBER', how='inner',
                         validate='one_to_one').set_index('id_input')
    raw = joined.reindex(ids)[['NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO']].to_numpy(float)
    values = raw.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        values[:, 3] = 10**(-0.4*(raw[:, 3]-30))
    usable = np.isfinite(values).all(1) & (np.square(values[:, :2]).sum(1) < 1) & (values[:, 2] > 0) & (values[:, 3] > 0)
    return values, usable


def cases():
    for case in range(40, 60):
        with np.load(FS2/f'chunk{(case-40)//10}/rows/case{case:03d}.npz') as z:
            yield 'FS2', case, z['context'][:, 3], [(z[f'measured_{l}'], z[f'usable_{l}']) for l in ('plus', 'minus')]
    for case in range(1000, 1020):
        with np.load(GPARENT/f'case{case}.npz') as z:
            ids, mag = z['input_index'], z['context'][:, 3]
        yield 'GalSBI', case, mag, [galsbi_leg(case, s, ids) for s in (H, -H)]


def keep(values, usable, limit):
    k = usable & (values[:, 3] > FLUX_MINIMUM)
    with np.errstate(invalid='ignore'):
        if limit[0] == '>':
            k &= values[:, 2] > limit[1]
        elif limit[0] == '>=':
            k &= values[:, 2] >= limit[1]
    return k


def main():
    out = {}
    for family, case, mag, legs in cases():
        b = np.digitize(mag, MAG4)
        rec = out.setdefault(family, {'hist': np.zeros((4, len(EDGES)-1)), 'sums': np.zeros((len(LIMITS), 2, 4, 4)),
                                      'per_case': []})
        v, u = legs[0]
        sel = keep(v, u, LIMITS[0])
        for i in range(4):
            rec['hist'][i] += np.histogram(v[sel & (b == i), 2], EDGES)[0]
        cs = np.zeros((len(LIMITS), 2, 4, 4))
        for j, lim in enumerate(LIMITS):
            kp = [keep(v_, u_, lim) for v_, u_ in legs]
            both = kp[0] & kp[1]
            for l, (v_, _) in enumerate(legs):
                for i in range(4):
                    s, s2 = kp[l] & (b == i), both & (b == i)
                    cs[j, l, i] = s.sum(), v_[s, 0].sum(), s2.sum(), v_[s2, 0].sum()
        rec['sums'] += cs
        rec['per_case'].append(cs)
        print(family, case, flush=True)
    W = np.random.default_rng(20260925).multinomial(20, np.full(20, 1/20), size=2000)
    result = {}
    for family, rec in out.items():
        pc = np.array(rec['per_case'])   # case, limit, leg, bin, (n, e1, n_both, e1_both)

        def resp(x):   # (..., leg, bin, 4) -> per-bin total and both-leg response; leg-normalized by all-bin n
            N = x[..., 0].sum(-1, keepdims=True)
            r = lambda e: (e[..., 0, :]/N[..., 0, :]-e[..., 1, :]/N[..., 1, :])/(2*H)
            return r(x[..., 1]), r(x[..., 3])
        tot, both = resp(pc.sum(0))
        btot, bboth = resp(np.einsum('bc,c...->b...', W, pc))
        res = {}
        for j, lim in enumerate(LIMITS):
            name = 'none' if lim[0] is None else f'{lim[0]}{lim[1]}'
            res[name] = {'R_total': [float(tot[j].sum()), float(btot[:, j].sum(-1).std(ddof=1))],
                         'R_selection_part': [float((tot[j]-both[j]).sum()), float((btot[:, j]-bboth[:, j]).sum(-1).std(ddof=1))],
                         'by_mag_total': tot[j].tolist(), 'by_mag_selection_part': (tot[j]-both[j]).tolist(),
                         'by_mag_selection_part_se': (btot[:, j]-bboth[:, j]).std(0, ddof=1).tolist(),
                         'selected_fraction_by_mag': (pc.sum(0)[j, 0, :, 0]/pc.sum(0)[j, 0, :, 0].sum()).tolist()}
        h = rec['hist']
        res['radius_hist_edges'] = EDGES.tolist()
        res['radius_hist_by_mag'] = h.tolist()
        res['fraction_in_3_to_3p75_px'] = (h[:, (EDGES[:-1] >= 3) & (EDGES[:-1] < 3.75)].sum(1)/h.sum(1)).tolist()
        res['fraction_below_3_px'] = (h[:, EDGES[:-1] < 3].sum(1)/h.sum(1)).tolist()
        result[family] = res
    (S/'sim_size_boundary.json').write_text(json.dumps(result, indent=1)+'\n')
    for family, res in result.items():
        print(f'\n{family}: fraction of MAG_AUTO<25.8 usable with radius <3 px by true mag',
              np.round(res['fraction_below_3_px'], 3), ' in 3-3.75 px', np.round(res['fraction_in_3_to_3p75_px'], 3))
        for name in ('none', '>3.0', '>=3.75'):
            r = res[name]
            print(f"  radius {name:7s} R {r['R_total'][0]:.4f}+-{r['R_total'][1]:.4f}  selection part {r['R_selection_part'][0]:+.4f}+-{r['R_selection_part'][1]:.4f}"
                  f"  by mag sel part {np.round(r['by_mag_selection_part'], 4)} +- {np.round(r['by_mag_selection_part_se'], 4)}")


if __name__ == '__main__':
    main()
