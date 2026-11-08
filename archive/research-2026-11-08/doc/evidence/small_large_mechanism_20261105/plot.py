"""Plot small saved summaries; no catalogue scan or model evaluation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

O = Path(__file__).resolve().parent
r = json.loads((O/'report.json').read_text())
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)
crowds = ['<0.01', '0.01-0.1', '0.1-0.3', '>=0.3']
x = np.arange(4)
for dx, name, label, color, marker in [(-.12, 'total', 'Total error', '#126782', 'o'),
                                      (.12, 'opposite', 'Opposite-group error', '#d78126', 's')]:
    rows = [r['response']['actual_U']['small_crowd/'+c][name] for c in crowds]
    axes[0].errorbar(x+dx, [v['value'] for v in rows], yerr=[v['se'] for v in rows],
                    fmt=marker, color=color, label=label, capsize=4, markersize=6)
axes[0].axhline(0, color='.5', linewidth=.8)
axes[0].set(xticks=x, xticklabels=crowds, xlabel='Close-crowding proxy (log1p weighted flux ratio)',
            ylabel='Contribution to global response error', title='Residual peaks at modest crowding\nIntrinsic radius 0.15–0.4 arcsec; measured radius > 3 px')
axes[0].legend(frameon=False, fontsize=10)
mid = [2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.1, 3.35]
for c, label, color, marker in [('<0.1', 'Crowding < 0.1', '#126782', 'o'),
                               ('>=0.1', 'Crowding ≥ 0.1', '#d78126', 's')]:
    rows = r['alignment']['0.15-0.25/'+c][1:9]
    axes[1].errorbar(mid, [v['slope']['value'] for v in rows],
                    yerr=[v['slope']['se'] for v in rows], fmt=marker+'-', color=color,
                    label=label, capsize=3, markersize=4)
axes[1].axhline(0, color='.5', linewidth=.8)
axes[1].axvline(3, color='.3', linewidth=1, linestyle='--', label='Size cut')
axes[1].set(xlabel='Measured FLUX_RADIUS (pixels)', ylabel='Measured–intrinsic shape alignment slope',
            title='The size cut selects different shape errors\nIntrinsic radius 0.15–0.25 arcsec')
axes[1].legend(frameon=False, fontsize=10, loc='upper left')
fig.suptitle('GalSBI constant-shear diagnostic · MAG_AUTO < 25 · four paired cases', fontsize=14)
fig.supxlabel('Error = simulation − model; 1σ case-bootstrap bars. Alignment pools ±0.02 legs; it is not shear response.', fontsize=10)
for ext in ['png', 'pdf']:
    fig.savefig(O/f'mechanism.{ext}', dpi=180, facecolor='white')
plt.close(fig)
