"""Render the requested FS2 panel from the preserved profile summary."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / 'profile_comparison.json').read_text())
profile = data['families']['fs2']['25.8']
centres = 2.05 + 0.1 * np.arange(30)
fig, ax = plt.subplots(figsize=(7.2, 4.8), layout='constrained')
for branch, label, colour in [(0, 'Simulation', '#252525'), (1, 'Flow', '#d35400')]:
    values = np.asarray(profile['alignment'][branch])[1:-1]
    errors = np.asarray(profile['case_se'][branch])[1:-1]
    ax.plot(centres, values, color=colour, label=label, linewidth=2)
    ax.fill_between(centres, values-errors, values+errors, color=colour, alpha=0.2)
ax.axvline(3, color='#4783a7', linestyle='--', linewidth=1.2, label='Radius cut: 3 px')
ax.axhline(0, color='0.75', linewidth=0.8)
ax.set(xlim=(2.4, 3.8), ylim=(-0.9, 0.9),
       xlabel='Measured FLUX_RADIUS [pixels]', ylabel='Measured–intrinsic alignment',
       title='FS2: MAG_AUTO < 25.8\nIntrinsic circularized $R_e$ = 0.15–0.25 arcsec')
ax.legend(frameon=False, loc='upper left')
fig.supxlabel('Shading: ±1 case-bootstrap SE; zero-shear measurements', fontsize=10)
for suffix in ['png', 'pdf']:
    fig.savefig(root / f'fs2_mag258_alignment.{suffix}', dpi=200)
plt.close(fig)
