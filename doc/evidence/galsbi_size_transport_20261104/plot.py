#!/usr/bin/env python
"""Figure for the size-wave diagnosis: alignment and response per measured size by true size; error by threshold."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_size_transport_20261104')
HERE = Path(__file__).resolve().parent
al = json.loads((ROOT/'alignment.json').read_text())['alignment']
jo = json.loads((ROOT/'joint.json').read_text())
groups = ['Re<0.15', '0.15-0.25', '0.25-0.4', '>=0.4']
label = {'Re<0.15': '<0.15', '0.15-0.25': '0.15-0.25', '0.25-0.4': '0.25-0.4', '>=0.4': '>=0.4'}
colors = dict(zip(groups, ['#1b6ca8', '#d1495b', '#2a9d55', '#8a6bbe']))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for g in groups:
    rows = [r for r in al[g] if r['n_sim'] >= 2000]
    x = [r['radius_lo'] + 0.05 for r in rows]
    ax[0].errorbar(x, [r['sim'][0] for r in rows], [r['sim'][1] for r in rows], color=colors[g], marker='o', ms=3,
                   label=f'sim, true Re {label[g]}"')
    ax[0].plot(x, [r['steps4'][0] for r in rows], color=colors[g], ls='--', label='GalSBI-dev flow')
    prof = [r for r in jo['profile'] if r[g]['share'] > 0.02]
    xp = [r['radius_lo'] + 0.05 for r in prof]
    ax[1].plot(xp, [r[g]['sim'] for r in prof], color=colors[g], marker='o', ms=3)
    ax[1].plot(xp, [r[g]['steps4'] for r in prof], color=colors[g], ls='--')
    bt = {float(k.split('_')[1]): v for k, v in jo['by_threshold'].items()}
    ts = sorted(bt)
    ax[2].errorbar(ts, [bt[t][g][0] for t in ts], [bt[t][g][1] for t in ts], color=colors[g], marker='o', ms=3,
                   label=f'true Re {label[g]}"')
ts = sorted(bt)
ax[2].errorbar(ts, [bt[t]['total'][0] for t in ts], [bt[t]['total'][1] for t in ts], color='k', marker='s', ms=4,
               label='total')
ax[0].set(xlabel='measured FLUX_RADIUS (px), zero shear', ylabel='alignment: sum(x s)/sum(x x)',
          title='Measured vs intrinsic shape at g = 0 (MAG_AUTO < 25)')
ax[1].set(xlabel='measured FLUX_RADIUS (px), each leg its own', ylabel='shear response per galaxy in slice',
          title='Response per measured-size slice (solid sim, dashed flow)')
ax[2].set(xlabel='cut: FLUX_RADIUS >= T (px), MAG_AUTO < 25', ylabel='100 (flow - sim) / sim   (m ~ minus this)',
          title='Flow error at the cut, by true size')
for a in ax:
    a.axvline(3.0, color='0.6', lw=0.8)
    a.axhline(0, color='0.6', lw=0.8)
ax[0].set_xlim(2.2, 4.1)
ax[1].set_xlim(2.2, 4.1)
ax[0].legend(fontsize=7, ncol=2)
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE/'size_wave_cause.png', dpi=130)
shutil.copy(HERE/'size_wave_cause.png', '/home/z/Zekang.Zhang/SBSI/plots/galsbi_size_wave_cause_20261104.png')
