"""Figure: FS2 V3.9 contributions to m by true size and magnitude; measured-radius distributions at the size cut."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = Path(__file__).resolve().parent
d = json.loads((S/'fs2_split.json').read_text())
b = json.loads((S/'sim_size_boundary.json').read_text())
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, axis, xlabel in ((ax[0], 'size', 'true circularized size [arcsec]'), (ax[1], 'mag', 'true magnitude')):
    edges = np.array(d['axes'][axis][1])
    centres = np.concatenate([[edges[0]-.5*(edges[1]-edges[0])], .5*(edges[1:]+edges[:-1]), [edges[-1]+.5*(edges[-1]-edges[-2])]])
    for cut, colour in (('uncut', 'grey'), ('main', 'k'), ('ge0.75', 'tab:orange')):
        r = d['results'][f'c8/{cut}/{axis}']
        a.errorbar(centres, r['contribution'], r['contribution_se'], fmt='o-', ms=3, color=colour,
                   label=f"{cut}: m {r['m']:+.2f} ± {r['m_se']:.2f}%")
    a.axhline(0, color='k', lw=.5)
    a.set_xlabel(xlabel)
    a.set_ylabel('contribution to m [pp]')
    a.set_title(f'V3.9 on FS2 ConstGold, by {axis}')
    a.legend(fontsize=8)
ax[0].set_xscale('log')
e = np.array(b['FS2']['radius_hist_edges'])
c = .5*(e[1:]+e[:-1])
for fam, ls in (('FS2', '-'), ('GalSBI', '--')):
    h = np.array(b[fam]['radius_hist_by_mag'])
    for i, (lab, colour) in enumerate(zip(('<24', '24-25.5', '25.5-27.5'), ('tab:blue', 'tab:green', 'tab:red'))):
        ax[2].plot(c, h[i]/h[i].sum()/.1, ls, color=colour, label=f'{fam} true mag {lab}')
for x in (3, 3.75):
    ax[2].axvline(x, color='k', lw=.8, ls=':')
ax[2].set_xlim(1.5, 8)
ax[2].set_xlabel('measured FLUX_RADIUS [px] (MAG_AUTO < 25.8, plus leg)')
ax[2].set_ylabel('density')
ax[2].set_title('size cut 3 px sits on the GalSBI peak')
ax[2].legend(fontsize=7)
fig.tight_layout()
for out in (S/'fs2_galsbi_split_20261103.png', Path('/home/z/Zekang.Zhang/SBSI/plots/fs2_galsbi_split_20261103.png')):
    fig.savefig(out, dpi=130)
