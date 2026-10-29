"""Read-only, model-free: per-case sums of measured-minus-intrinsic e1/e2 in measured-magnitude bands for every
ConstGold catalogue (cases 0-139, shear legs +0.02 and -0.02), with the 1M run's measured cuts
(radius>3 px, radius<100 px, mag_auto<25.8, |e|<0.99, usable shapes, no truth cuts)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

S = Path('/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant')
EDGES = np.array([-np.inf, 21, 23, 24, 25, 25.8])
OUT = Path(sys.argv[1])


def one(job):
    case, leg = job
    base = S/f'case{case}_{leg}'/'real0'/'catalogues'
    t = pd.read_feather(base/'input/gals_info_tile180.0_-0.5.feather',
                        columns=['index_input', 'e1_input_rot0', 'e2_input_rot0', 'gamma1_input', 'gamma2_input'])
    x = pd.read_feather(base/'CrossMatch/tile180.0_-0.5_rot0_matched.feather', columns=['id_detec', 'id_input'])
    s = pd.read_feather(base/'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather',
                        columns=['NUMBER', 'NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS'])
    j = x.merge(s, left_on='id_detec', right_on='NUMBER', validate='one_to_one').merge(
        t, left_on='id_input', right_on='index_input', validate='one_to_one')
    v = j[['NGMIX_G1', 'NGMIX_G2', 'MAG_AUTO', 'FLUX_RADIUS']].to_numpy(float)
    ok = np.isfinite(v).all(1) & (v[:, 3] > 3) & (v[:, 3] < 100) & (v[:, 2] < 25.8) & (v[:, 0] != -1.) \
        & ~((v[:, 0] == 0) & (v[:, 1] == 0)) & (np.hypot(v[:, 0], v[:, 1]) < .99)
    j, v = j[ok], v[ok]
    ei = j[['e1_input_rot0', 'e2_input_rot0']].to_numpy()
    d = v[:, :2]-ei
    b = np.digitize(v[:, 2], EDGES)-1
    res = np.zeros((5, 9))
    for k in range(5):
        m = b == k
        res[k] = [m.sum(), *d[m].sum(0), *(d[m]**2).sum(0), *ei[m].sum(0), *v[m, :2].sum(0)]
    return case, leg, float(j.gamma1_input.iloc[0]), float(j.gamma2_input.iloc[0]), res


if __name__ == '__main__':
    jobs = [(c, l) for c in range(140) for l in ('0.02', '-0.02')]
    with Pool(int(sys.argv[2])) as p:
        out = p.map(one, jobs, chunksize=1)
    np.savez(OUT, case=[o[0] for o in out], leg=[float(o[1]) for o in out], g1=[o[2] for o in out],
             g2=[o[3] for o in out], sums=np.array([o[4] for o in out]))
    print('saved', OUT, len(out))
