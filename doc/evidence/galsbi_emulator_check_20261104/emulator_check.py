"""GalSBI-dev emulator vs half-shear simulation labels by size and magnitude (rule.md, 2026-11-04).

usage (py31):
  python emulator_check.py sums      # Slurm CPU job: per-case binned pair sums -> $STUDY/sums.npz
  python emulator_check.py amplitude # Slurm CPU job: same pairs at neighbour shear 0.2 and 0.05 -> $STUDY/amplitude.npz
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
TRUTH = str(D/'sbsi/galsbi_resim/training/sims/case{}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather')
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
        index = truth['index_input'].to_numpy()
        if len(np.unique(index)) != len(index):
            raise ValueError(f'case {case}: duplicate truth input index')
        r_true = np.full(index.max()+1, np.nan)                 # lookup by input index (some cases skip one)
        r_true[index] = truth['r_input'].to_numpy(float)
        parts = [p for p in manifest['parts'] if p['case'] == case]
        for part in parts:
            with np.load(PAIRS/'pairs'/part['path']) as z, np.load(PAIRS/'prepared/gains'/Path(part['path']).name) as g:
                x, label, null, primary = z['x'], z['y'].astype(float), z['null'].astype(float), z['primary']
                gain, cross = g['gain'].astype(float), g['cross_gain'].astype(float)
            if len(x) != part['rows']:
                raise ValueError(f'{part["path"]}: row count mismatch')
            if not np.isfinite(r_true[primary]).all():
                raise ValueError(f'{part["path"]}: primary without truth row')
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


PROFILE_EDGES = {'flux_radius': np.arange(3., 8.01, .25), 'mag_auto': np.arange(20., 25.61, .2),
                 'true_r': np.arange(20., 30.01, .5), 'neighbour_true_r': np.arange(16., 30.01, .5)}


def profiles():
    """Fine one-property profiles for the plot: per case, selection (main, target) and bin: rows, label, gain*b."""
    import pandas as pd
    import xgboost as xgb
    root = MODELS['galsbi_dev']
    meta = json.loads((root/'metadata.json').read_text())
    booster = xgb.Booster()
    booster.load_model(root/'model.json')
    booster.set_param({'nthread': 8})
    manifest = json.loads((PAIRS/'pairs/manifest.json').read_text())
    out = {k: np.zeros((len(CASES), 2, len(e)+1, 3)) for k, e in PROFILE_EDGES.items()}
    for case in CASES:
        truth = pd.read_feather(TRUTH.format(case), columns=['index_input', 'r_input'])
        r_true = np.full(truth['index_input'].max()+1, np.nan)
        r_true[truth['index_input'].to_numpy()] = truth['r_input'].to_numpy(float)
        for part in [p for p in manifest['parts'] if p['case'] == case]:
            with np.load(PAIRS/'pairs'/part['path']) as z, np.load(PAIRS/'prepared/gains'/Path(part['path']).name) as g:
                x, label, primary, gain = z['x'], z['y'].astype(float), z['primary'], g['gain'].astype(float)
            b = booster.inplace_predict(np.ascontiguousarray(x)).astype(float)*meta['y_std']+meta['y_mean']
            rp = r_true[primary]
            # scaled magnitudes are true magnitudes plus one survey constant (checked 2.6413 for every pair)
            prop = {'flux_radius': np.exp(x[:, 7]), 'mag_auto': x[:, 8], 'true_r': rp,
                    'neighbour_true_r': x[:, 3]-(x[:, 2]-rp)}
            for s, keep in enumerate((np.ones(len(x), bool), x[:, 8] < 25.)):
                for k, e in PROFILE_EDGES.items():
                    idx = np.digitize(prop[k][keep], e)
                    for q, v in enumerate((np.ones(keep.sum()), label[keep], (gain*b)[keep])):
                        out[k][case, s, :, q] += np.bincount(idx, v, minlength=len(e)+1)
        print(f'case {case} done', flush=True)
    np.savez(STUDY/'profiles.npz', **out, **{f'{k}_edges': e for k, e in PROFILE_EDGES.items()})


AMPLITUDE_Q = ('rows', 'label20', 'label05', 'null20', 'null05')


def amplitude_case(case):
    """Same pairs rebuilt with the production label code at neighbour shear 0.2 and 0.05 (legs share noise)."""
    sys.path.insert(0, '/home/z/Zekang.Zhang/blendemu_galsbi_resim')
    import pandas as pd
    from blendemu import response
    manifest = json.loads((PAIRS/'pairs/manifest.json').read_text())
    primary, log_radius, mag, rows, label_sum = [], [], [], 0, 0.
    for part in [p for p in manifest['parts'] if p['case'] == case]:
        with np.load(PAIRS/'pairs'/part['path']) as z:
            primary.append(z['primary'])
            log_radius.append(z['x'][:, 7])
            mag.append(z['x'][:, 8])
            rows += len(z['y'])
            label_sum += float(z['y'].astype(float).sum())
    population = pd.DataFrame({'input_index': np.concatenate(primary), 'radius': np.exp(np.concatenate(log_radius)),
                               'mag': np.concatenate(mag)}).drop_duplicates('input_index')
    frames = {}
    for leg, h in (('0.2', .2), ('0.05', .05)):
        df = response.retrieve_response(case=case, r_max=10, r_min=0, k=20, real='real0', tile_name='tile180.0_-0.5',
                                        data_path=str(D/'sbsi/galsbi_resim/training/sims'), shear_cases=['0.0', leg])
        df = df[np.isfinite(df['delta_et1'].to_numpy(float)) & df['input_index'].isin(population['input_index'])]
        frames[leg] = pd.DataFrame({'input_index': df['input_index'].to_numpy(), 'distance': df['distance'].to_numpy(),
                                    'angle': df['shear_angle'].to_numpy(float), 'r_true': df['r_input_p'].to_numpy(float),
                                    'label': df['delta_et1'].to_numpy(float)/h, 'null': df['delta_et2'].to_numpy(float)/h})
    check = {'parts_rows': rows, 'parts_label_sum': label_sum, 'rebuilt20_rows': len(frames['0.2']),
             'rebuilt20_label_sum': float(frames['0.2']['label'].sum()), 'rebuilt05_rows': len(frames['0.05'])}
    both = frames['0.2'].merge(frames['0.05'], on=['input_index', 'distance'], suffixes=('20', '05'))
    angle = np.abs((both['angle20']-both['angle05']+90) % 180-90)
    if angle.max() > 1e-3 or not np.allclose(both['r_true20'], both['r_true05']):
        raise ValueError(f'case {case}: 0.2 and 0.05 pairs disagree (angle {angle.max()})')
    both = both.merge(population, on='input_index')
    check['matched_rows'] = len(both)
    shape = (len(RADIUS_LABELS), len(MAG_LABELS), len(TRUE_LABELS))
    cell = np.ravel_multi_index((np.digitize(both['radius'], RADIUS_EDGES), np.digitize(both['mag'], MAG_EDGES),
                                 np.digitize(both['r_true20'], TRUE_EDGES)), shape)
    out = np.stack([np.bincount(cell, v, minlength=np.prod(shape)).reshape(shape) for v in
                    (np.ones(len(both)), both['label20'], both['label05'], both['null20'], both['null05'])], -1)
    print(f'case {case}: {check}', flush=True)
    return out, check


def amplitude():
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', '4')))(
        delayed(amplitude_case)(case) for case in CASES)
    np.savez(STUDY/'amplitude.npz', sums=np.array([r[0] for r in results]), quantities=np.array(AMPLITUDE_Q),
             cases=np.array(list(CASES)))
    (STUDY/'amplitude_checks.json').write_text(json.dumps({c: r[1] for c, r in zip(CASES, results)}, indent=1)+'\n')


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
    if (STUDY/'amplitude.npz').exists():
        a = np.load(STUDY/'amplitude.npz')['sums']
        qa = {name: k for k, name in enumerate(AMPLITUDE_Q)}
        print('\namplitude: d05 = 100 (sum label at neighbour shear 0.05 / sum label at 0.2 - 1) [%] on the same pairs; '
              'positive = response at small shear larger than the emulator was trained on')
        out['amplitude'] = {}
        for sel, (rs, ms) in selections.items():
            sub = a[:, rs, ms].sum((1, 2))
            full = np.concatenate([sub, sub.sum(1, keepdims=True)], 1)
            ratio = lambda agg: 100*(agg[..., qa['label05']]/agg[..., qa['label20']]-1)
            point, se = ratio(full.sum(0)), np.std(ratio(np.einsum('bc,c...->b...', W, full)), axis=0, ddof=1)
            out['amplitude'][sel] = {tl: {'pairs': int(full[:, t, qa['rows']].sum()), 'd05_percent': [float(point[t]), float(se[t])]}
                                     for t, tl in enumerate(TRUE_LABELS+('all',))}
            print(f'  {sel:32s} ' + '  '.join(f'{tl} {point[t]:+.1f}±{se[t]:.1f}' for t, tl in enumerate(TRUE_LABELS+('all',))))
    (S/'emulator_check.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    {'sums': sums, 'amplitude': amplitude, 'profiles': profiles, 'report': report}[sys.argv[1]]()
