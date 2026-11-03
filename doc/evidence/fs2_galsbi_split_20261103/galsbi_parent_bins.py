"""Parent (all rendered galaxies) counts of the GalSBI constant-shear family (cases 1000-1139) on the fs2_bins grids.

usage (Slurm CPU, py31): python galsbi_parent_bins.py
Reads galsbi_dev_constant_20261103/parent/case*.npz (8 F8 inputs). Writes galsbi_parent_bins.npz: per axis the
summed parent count per bin (cases summed) and per case.
"""
from pathlib import Path

import numpy as np

from fs2_bins import AXES, MAG4, SIZE2D

S = Path(__file__).resolve().parent
P = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_dev_constant_20261103/parent')
COLUMN = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 6}   # FS2 11-column context -> GalSBI 8-column context


def main():
    out = {}
    for c, case in enumerate(range(1000, 1140)):
        with np.load(P/f'case{case}.npz') as z:
            ctx = z['context']
        index = {}
        for name, (col, edges) in AXES.items():
            x = np.hypot(ctx[:, 0], ctx[:, 1]) if col == 'abs_e' else ctx[:, COLUMN[col]]
            index[name] = (np.digitize(x, edges), len(edges)+1)
        index['mag4_size'] = (np.digitize(ctx[:, 3], MAG4)*(len(SIZE2D)+1)+np.digitize(ctx[:, 4], SIZE2D),
                              4*(len(SIZE2D)+1))
        nm, ns = index['mag'][1], index['size'][1]
        index['mag_size'] = (index['mag'][0]*ns+index['size'][0], nm*ns)
        for name, (b, nb) in index.items():
            out.setdefault(name, np.zeros((140, nb)))[c] = np.bincount(b, minlength=nb)
    np.savez(S/'galsbi_parent_bins.npz', **out)
    print({k: int(v.sum()) for k, v in out.items()})


if __name__ == '__main__':
    main()
