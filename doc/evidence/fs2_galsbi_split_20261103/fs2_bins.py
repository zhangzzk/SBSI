"""Per-case binned sums of the saved V3.9-like FS2 ConstGold evaluation (cases 40-119), for splitting m by truth.

usage (Slurm CPU, py31): python fs2_bins.py
Reads the saved per-galaxy rows (v36_f8_20260924_v1/evaluation/chunk*/rows) and the primary C8 probabilities
(v39_constgold_ecut07_20261005/c8). Nothing is redrawn. Writes fs2/case<NNN>.npz with, per axis, sums
(leg, cut, bin, quantity); quantities Q below. Cuts are the three saved rows (uncut, main, >=0.75").
"""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
C = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
F8 = C/'v36_f8_20260924_v1/evaluation'
C8 = C/'v39_constgold_ecut07_20261005/c8'
LEGS = ('plus', 'minus')
CUT_RADIUS = ((None, None), ('>', 3.0), ('>=', 3.75))
FLUX_MINIMUM = 10**(-0.4*(25.8-30))
# simulation count, e1; C8 mass, e1 (flow+disk), e1 flow only; true-flag mass, e1 (flow+disk)
Q = ('parent', 'sim_n', 'sim_e1', 'c8_n', 'c8_e1', 'c8_e1_flow', 'tf_n', 'tf_e1')
# truth axes: name -> (context column or derived, edges)
AXES = {
    'mag': (3, np.arange(18., 29.01, .5)),
    'size': (4, np.array([.05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .8, 1., 1.5])),
    'sersic': (2, np.array([.75, 1., 1.5, 2., 3., 4.])),
    'abs_e': ('abs_e', np.array([.1, .2, .3, .4, .5, .6, .7])),
    'crowd100': (9, np.array([.01, .1, .3, .6, 1., 2.])),
}
MAG4 = (24., 25.5, 27.5)
SIZE2D = np.array([.1, .15, .2, .3, .4, .6, 1.])


def sim_keep(values, usable, k):
    keep = usable.copy()
    if k:
        keep &= values[:, 3] > FLUX_MINIMUM
        keep &= values[:, 2] > 3.0 if k == 1 else values[:, 2] >= 3.75
    return keep


def main():
    (S/'fs2').mkdir(exist_ok=True)
    for case in range(40, 120):
        with np.load(F8/f'chunk{(case-40)//10}/rows/case{case:03d}.npz') as z, np.load(C8/f'case{case:03d}.npz') as q:
            if not np.array_equal(z['input_index'], q['input_index']):
                raise ValueError(f'case {case}: rows differ')
            ctx = z['context']
            index = {}
            for name, (col, edges) in AXES.items():
                x = np.hypot(ctx[:, 0], ctx[:, 1]) if col == 'abs_e' else ctx[:, col]
                index[name] = (np.digitize(x, edges), len(edges)+1)
            index['mag4_size'] = (np.digitize(ctx[:, 3], MAG4)*(len(SIZE2D)+1)+np.digitize(ctx[:, 4], SIZE2D),
                                  4*(len(SIZE2D)+1))
            nm, ns = index['mag'][1], index['size'][1]
            index['mag_size'] = (index['mag'][0]*ns+index['size'][0], nm*ns)
            out = {name: np.zeros((2, 3, nb, len(Q))) for name, (_, nb) in index.items()}
            for l, leg in enumerate(LEGS):
                values, usable = z[f'measured_{leg}'], z[f'usable_{leg}']
                p, tf = q[f'probability_{leg}'], usable.astype(float)
                for k in range(3):
                    keep = sim_keep(values, usable, k)
                    mass = z[f'mass_{leg}'][:, k]
                    e_tot = z[f'flow_{leg}'][:, k, 0]+z[f'blend_{leg}'][:, k, 0]
                    e_flow = z[f'flow_{leg}'][:, k, 0]
                    cols = (np.ones(len(keep)), keep.astype(float), np.where(keep, values[:, 0], 0.), p*mass, p*e_tot, p*e_flow,
                            tf*mass, tf*e_tot)
                    for name, (b, nb) in index.items():
                        for j, x in enumerate(cols):
                            out[name][l, k, :, j] = np.bincount(b, x, nb)
        np.savez(S/f'fs2/case{case:03d}.npz', **out)
        print('case', case, 'sim main plus', int(out['mag'][0, 1, :, 0].sum()), flush=True)
    (S/'fs2/axes.json').write_text(json.dumps({k: [str(c), e.tolist()] for k, (c, e) in AXES.items()} |
                                              {'mag4_size': [list(MAG4), SIZE2D.tolist()], 'Q': Q}))


if __name__ == '__main__':
    main()
