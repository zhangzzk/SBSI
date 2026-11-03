"""Print the FS2 true-mag (4 bins) x true-size contributions and within-bin m from fs2_split.json."""
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
d = json.loads((S/'fs2_split.json').read_text())

mags = ('<24', '24-25.5', '25.5-27.5', '>=27.5')
ns = len(d['axes']['mag4_size'][1])+1
for key in ('c8/main/mag4_size', 'flow/main/mag4_size', 'c8/ge0.75/mag4_size', 'c8/uncut/mag4_size'):
    r = d['results'][key]
    print(f"\n{key}: m {r['m']:+.3f} +- {r['m_se']:.3f}")
    print('size edges', d['axes']['mag4_size'][1])
    for i, lab in enumerate(mags):
        sl = slice(i*ns, (i+1)*ns)
        c, ce = np.array(r['contribution'][sl]), np.array(r['contribution_se'][sl])
        w, we = np.array(r['within_bin_m'][sl]), np.array(r['within_bin_se'][sl])
        f = np.array(r['selected_fraction'][sl])
        print(f'{lab:10s} contrib ' + ' '.join(f'{x:+5.2f}' for x in c) + f'  | sum {c.sum():+.2f}')
        print(f'{"":10s} within  ' + ' '.join(f'{x:+5.1f}' if y > 1e-3 else '   . ' for x, y in zip(w, f)))
        print(f'{"":10s} +-      ' + ' '.join(f'{x:5.1f}' if y > 1e-3 else '   . ' for x, y in zip(we, f)))
        print(f'{"":10s} selfrac ' + ' '.join(f'{x:5.3f}' for x in f))
