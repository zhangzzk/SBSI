"""GalSBI-dev emulator vs half-shear simulation labels by size and magnitude (rule.md, 2026-11-04).

usage (py31):
  python emulator_check.py sums      # Slurm CPU job: per-case binned pair sums -> $STUDY/sums.npz
  python emulator_check.py report    # login node: case bootstrap -> emulator_check.json (+ printed table)
"""
import json
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
D = Path('/project/ls-gruen/users/zekang.zhang')
STUDY = D/'sbsi_caches/galsbi_emulator_check_20261104'
PAIRS = D/'blendemu_runs/disk_response_moment_galsbi_cut_20261103'
MODELS = {'galsbi_dev': D/'blendemu_runs/disk_response_moment_galsbi_20261103/model',
          'cut_variant': PAIRS/'model'}
TRUTH = D/'sbsi/galsbi_resim/training/sims/case{}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather'
CASES = range(40)
RADIUS_EDGES = (3.2, 3.5, 3.75, 4.5, 6.)          # px, rows all have FLUX_RADIUS > 3
MAG_EDGES = (22., 23., 24., 25.)                  # MAG_AUTO, rows all < 25.8
TRUE_EDGES = (24., 25.5, 27.5)
RADIUS_LABELS = ('3-3.2', '3.2-3.5', '3.5-3.75', '3.75-4.5', '4.5-6', '>6')
MAG_LABELS = ('<22', '22-23', '23-24', '24-25', '25-25.8')
TRUE_LABELS = ('<24', '24-25.5', '25.5-27.5', '>=27.5')
# per cell: rows, label, null, then (predicted, predicted null) per model
Q = ('rows', 'label', 'null') + tuple(f'{m}_{q}' for m in MODELS for q in ('pred', 'pred_null'))


def sums():
    import pandas as pd
    import xgboost as xgb
    boosters = {}
    for name, root in MODELS.items():
        meta = json.loads((root/'metadata.json').read_text())
        booster = xgb.Booster()
        booster.load_model(root/'model.json')
        booster.set_param({'nthread': 8})
        boosters[name] = (booster, meta['y_mean'], meta['y_std'])
        print(name, meta['model_sha256'], meta['y_mean'], meta['y_std'], flush=True)
    manifest = json.loads((PAIRS/'pairs/manifest.json').read_text())
    shape = (len(CASES), len(RADIUS_LABELS), len(MAG_LABELS), len(TRUE_LABELS), len(Q))
    out = np.zeros(shape)
    for case in CASES:
        truth = pd.read_feather(TRUTH.format(case), columns=['index_input', 'r_input'])
        if not np.array_equal(truth['index_input'].to_numpy(), np.arange(len(truth))):
            raise ValueError(f'case {case}: truth rows not indexed by input index')
        r_true = truth['r_input'].to_numpy(float)
        parts = [p for p in manifest['parts'] if p['case'] == case]
        for part in parts:
            with np.load(PAIRS/'pairs'/part['path']) as z, np.load(PAIRS/'prepared/gains'/Path(part['path']).name) as g:
                x, label, null, primary = z['x'], z['y'].astype(float), z['null'].astype(float), z['primary']
                gain, cross = g['gain'].astype(float), g['cross_gain'].astype(float)
            if len(x) != part['rows']:
                raise ValueError(f'{part["path"]}: row count mismatch')
            cell = np.ravel_multi_index((np.digitize(np.exp(x[:, 7]), RADIUS_EDGES), np.digitize(x[:, 8], MAG_EDGES),
                                         np.digitize(r_true[primary], TRUE_EDGES)), shape[1:4])
            values = [np.ones(len(x)), label, null]
            for booster, mean, std in boosters.values():
                b = booster.inplace_predict(np.ascontiguousarray(x)).astype(float)*std+mean
                if not np.isfinite(b).all():
                    raise ValueError(f'{part["path"]}: nonfinite prediction')
                values += [gain*b, cross*b]
            for k, v in enumerate(values):
                out[case, ..., k] += np.bincount(cell, v, minlength=np.prod(shape[1:4])).reshape(shape[1:4])
        print(f'case {case}: parts {len(parts)} rows {int(out[case, ..., 0].sum())}', flush=True)
    np.savez(STUDY/'sums.npz', sums=out, quantities=np.array(Q), cases=np.array(list(CASES)))


def report():
    z = np.load(STUDY/'sums.npz')
    x = z['sums']                                              # case, radius, mag, true, q
    n = len(x)
    W = np.random.default_rng(20260925).multinomial(n, np.full(n, 1/n), size=10000).astype(float)
    q = {name: k for k, name in enumerate(Q)}
    ge375 = RADIUS_LABELS.index('3.75-4.5')
    selections = {'radius>3 & mag<25 (target)': (slice(None), slice(0, 4)),
                  'radius>3 & mag<25.8 (main)': (slice(None), slice(None)),
                  'radius>=3.75 & mag<25': (slice(ge375, None), slice(0, 4))}
    for i, label in enumerate(RADIUS_LABELS):
        selections[f'radius {label} px (mag<25.8)'] = (slice(i, i+1), slice(None))
    for j, label in enumerate(MAG_LABELS):
        selections[f'MAG_AUTO {label} (radius>3)'] = (slice(None), slice(j, j+1))

    def ratios(agg):
        rows = {}
        for model in MODELS:
            p, pn = q[f'{model}_pred'], q[f'{model}_pred_null']
            rows[model] = np.stack([100*(agg[..., q['label']]/agg[..., p]-1), agg[..., q['null']]/agg[..., p],
                                    agg[..., pn]/agg[..., p]])
        return rows

    out = {}
    print('d = 100 (sum label / sum emulator prediction - 1) [%], positive = emulator too low; '
          'null = sum of the orthogonal component / sum prediction (label | emulator)')
    for sel, (rs, ms) in selections.items():
        sub = x[:, rs, ms].sum((1, 2))                          # case, true, q
        full = np.concatenate([sub, sub.sum(1, keepdims=True)], 1)     # last = all true magnitudes
        point = ratios(full.sum(0))
        boot = ratios(np.einsum('bc,c...->b...', W, full))
        entry = {}
        print(f'\n{sel}')
        for t, tl in enumerate(TRUE_LABELS+('all',)):
            row = {'pairs': int(full[:, t, q['rows']].sum()),
                   'label_per_pair': float(full[:, t, q['label']].sum()/full[:, t, q['rows']].sum())}
            line = f'  true r {tl:9s} pairs {row["pairs"]:>10d}'
            for model in MODELS:
                pt, se = point[model][:, t], np.std(boot[model][:, :, t], axis=1, ddof=1)
                row[model] = {'d_percent': [float(pt[0]), float(se[0])], 'null_label': [float(pt[1]), float(se[1])],
                              'null_emulator': [float(pt[2]), float(se[2])]}
                line += f'  {model} d {pt[0]:+6.1f}±{se[0]:.1f}% (null {pt[1]:+.3f}±{se[1]:.3f} | {pt[2]:+.3f})'
            entry[tl] = row
            print(line)
        out[sel] = entry
    (S/'emulator_check.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    {'sums': sums, 'report': report}[sys.argv[1]]()
