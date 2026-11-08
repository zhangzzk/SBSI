"""Forward evaluation of a model preset on the GalSBI constant-shear family (g1 = +-0.02, cases 1000-1139).

Same estimator as the V3.9-like ConstGold evaluation (doc/V39_MODEL.md#development-evidence): per parent galaxy,
64 antithetic flow draws at the +-0.02-sheared inputs (seed 7301 + 10000019 case + 1000003 block, blocks of 1024),
Mobius disk transport by the summed neighbour-pair emulator response, the same output cuts on draws and on the
measured legs, each leg normalized by its own selected count, galaxy weights = C8 usable probability (or, as a
diagnostic, the simulation's own usable flag), m = 100 (R_sim/R_model - 1). The draw stage runs the frozen V3.9
evaluation code (checkout sbsi_flow_f8_20260924) for flow draws and measured legs; the cuts (CUTS: the three frozen
rows plus separate size and magnitude limits) are applied here. C8 runs in its frozen checkout.

Shape/selection split (true usable flags): the shape part is the response of galaxies (simulation) or draws (model)
kept in both legs, the selection part is the rest; their contributions to m add up to m.

Parent = every rendered galaxy of a case (gals_info rows; the renderer's edge drops are absent and reported), no
truth cut. Inputs as in scripts/prepare_galsbi_f8c8_domain.py: intrinsic e1, e2 (sheared by +-h in the draw
stage), rendered Sersic n, LSST-i magnitude, Re*sqrt(q), close crowding over all rendered objects. Emulator pairs:
up to k = 20 nearest rendered objects within 10 arcsec (self excluded), no cuts, scaled with the preset's
observing conditions (sbsi.forward_catalogue; the BlendEMU rule, checked by the pairs-check stage on FS2).

Stages (each needs SLURM_JOB_ID):
  prep <chunk>        py31 + this repo: parent/caseNNNN.npz
  pairs-check         py31 + this repo: rebuild FS2 ConstGold case 40 pairs, compare with the saved V3.9 table
  c8 <chunk>          C8 venv: c8/caseNNNN.npz (probability per leg)
  draws <chunk>       F8 venv, GPU: sums/caseNNNN.npz + records
  summary             py31: summary.json
A chunk is ten consecutive cases from 1000.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe')
DATA = Path('/project/ls-gruen/users/zekang.zhang')
SIMS = DATA/'sbsi/galsbi_resim/constant_g002/sims'
# EVAL_OUT / EVAL_PRESET select another study root and preset file (configs/); defaults are the first study.
OUT = Path(os.environ.get('EVAL_OUT', DATA/'sbsi_caches/galsbi_dev_constant_20261103'))
PRESET = REPO/'configs'/os.environ.get('EVAL_PRESET', 'models_galsbi_dev.json')
F8_CHECKOUT = DATA/'sbsi_flow_f8_20260924'
C8_CHECKOUT = DATA/'sbsi_classifier_f8_inputs_20260925'
TILE = 'tile180.0_-0.5'
CASES = tuple(range(1000, 1140))
H, AXIS, DRAWS, SEED, BATCH = .02, 1, 64, 7301, 1024
K, R_MAX = 20, 10.
FEATURES = ['e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p', 'circularized_Re_input_p',
            'nbr_close_050', 'nbr_close_100', 'nbr_close_200']
LEGS = (('plus', H), ('minus', -H))
MAG_EDGES = (24., 25.5, 27.5)
# Output cuts on both sides: name, FLUX_RADIUS bound in pixels (0.2", '>' or '>='), MAG_AUTO upper limit.
# Rows 0-2 are the frozen V3.9 rows (evaluate_smooth_classifier_joint.output_cuts); the rest separate the limits.
CUTS = (('uncut', None, None),
        ('mag25.8_radius_gt0.60', ('>', 3.0), 25.8),
        ('mag25.8_radius_ge0.75', ('>=', 3.75), 25.8),
        ('mag25_radius_gt0.60', ('>', 3.0), 25.0),
        ('radius_gt0.60', ('>', 3.0), None),
        ('radius_ge0.75', ('>=', 3.75), None),
        ('mag25.8', None, 25.8),
        ('mag25', None, 25.0),
        # target grid around MAG_AUTO < 25 and radius > 3 px (band-guard diagnostic, 2026-11-04)
        ('mag25_radius_gt0.56', ('>', 2.8), 25.0),
        ('mag25_radius_gt0.58', ('>', 2.9), 25.0),
        ('mag25_radius_gt0.62', ('>', 3.1), 25.0),
        ('mag25_radius_gt0.64', ('>', 3.2), 25.0),
        ('mag25_radius_ge0.75', ('>=', 3.75), 25.0),
        ('mag24.6_radius_gt0.60', ('>', 3.0), 24.6),
        ('mag25.4_radius_gt0.60', ('>', 3.0), 25.4))
CUT_NAMES = tuple(c[0] for c in CUTS)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()


def output_cuts(values):
    """Keep mask (..., cut) for measured (numpy) or generated (torch) joint outputs: radius in pixels, flux."""
    radius, flux = values[..., 2], values[..., 3]
    rows = []
    for _, size, mag in CUTS:
        keep = radius == radius   # all true for finite rows; nonfinite measured rows are not usable anyway
        if size is not None:
            keep = keep & ((radius > size[1]) if size[0] == '>' else (radius >= size[1]))
        if mag is not None:
            keep = keep & (flux > 10**(-0.4*(mag-30)))
        rows.append(keep)
    return (np if isinstance(values, np.ndarray) else __import__('torch')).stack(rows, -1)


def draw_moments(keep, values, shift):
    """Selected mass, flow shape and disk shift over draws (as the frozen selected_draw_moments)."""
    keep = keep.to(values.dtype)
    return (keep.mean(1), (keep[..., None]*values[..., None, :2]).mean(1),
            (keep[..., None]*shift[:, :, None, :]).mean(1))


def chunk_cases(chunk):
    cases = CASES[10*chunk:10*chunk+10]
    if not cases:
        raise ValueError(f'empty chunk {chunk}')
    return cases


def preset():
    spec = json.loads(PRESET.read_text())
    paths = {'flow': DATA/'sbsi_caches'/spec['flow']['path'],
             'classifier': DATA/'sbsi_caches'/spec['classifier']['path'],
             'emulator': DATA/'blendemu_runs'/spec['emulator']['path']}
    for key, digest in (('flow', spec['flow']['sha256']), ('classifier', spec['classifier']['sha256']),
                        ('emulator', spec['emulator']['sha256'])):
        if sha256(paths[key]) != digest:
            raise ValueError(f'{key} artifact differs from the GalSBI-dev pin')
    conditions = {k: spec['observing_conditions'][k]
                  for k in ('pixel_size', 'zero_point', 'psf_fwhm', 'moffat_beta', 'pixel_rms')}
    return spec, paths, conditions


def full_context(context):
    """Pad to the 11 frozen FULL_FLOW_FEATURES columns; the unused FS2 nbr_flux_* columns are zero.

    Both frozen checkouts build sheared frames on these columns; F8 and C8 read their eight inputs by name."""
    full = np.zeros((len(context), 11))
    full[:, :5], full[:, 8:] = context[:, :5], context[:, 5:]
    return full


def truth(case, shear):
    import pandas as pd
    path = SIMS/f'case{case}_{shear:.2f}/real0/catalogues/input/gals_info_{TILE}.feather'
    frame = pd.read_feather(path).sort_values('index_input', kind='stable').reset_index(drop=True)
    return frame, path


def pair_table(scene, parent_rows, conditions):
    """Up to K nearest rendered neighbours within R_MAX of each parent row, scaled emulator features."""
    from sbsi.forward_catalogue import make_pair_catalogue, rescale_emulator_pairs
    from sbsi.output_conditioned_response import TRUTH_FEATURES
    pairs = make_pair_catalogue(scene.iloc[parent_rows], scene, r_max_arcsec=R_MAX, k=K)
    pairs = rescale_emulator_pairs(pairs, conditions)
    pairs = pairs[['index_input_p', 'index_input_s', *TRUTH_FEATURES]]
    pairs = pairs.sort_values(['index_input_p', 'index_input_s'], kind='stable').reset_index(drop=True)
    if pairs.duplicated(['index_input_p', 'index_input_s']).any() or not np.isfinite(
            pairs[list(TRUTH_FEATURES)].to_numpy()).all():
        raise ValueError('invalid emulator pairs')
    return pairs


def scene_frame(t):
    import pandas as pd
    return pd.DataFrame({name.replace('_input', ''): t[name] for name in t.columns})


def prep(chunk):
    sys.path.insert(0, str(REPO))
    import pandas as pd
    from sbsi.v39_inference import physical_close_crowding
    (OUT/'parent').mkdir(parents=True, exist_ok=True)
    for case in chunk_cases(chunk):
        started = time.monotonic()
        t = {name: truth(case, shear) for name, shear in LEGS}
        plus, minus = t['plus'][0], t['minus'][0]
        for c in plus.columns:
            if c in ('gamma1_input', 'gamma2_input'):
                continue
            if not np.array_equal(plus[c].to_numpy(), minus[c].to_numpy()):
                raise ValueError(f'case {case}: truth column {c} differs between legs')
        for (name, shear), frame in zip(LEGS, (plus, minus)):
            if not (np.all(frame['gamma1_input'] == shear) and np.all(frame['gamma2_input'] == 0)):
                raise ValueError(f'case {case}: leg {name} is not g = ({shear}, 0)')
        scene = pd.read_feather(SIMS.parent/f'scenes/scene{case}.feather', columns=['RA', 'DEC'])
        ids = plus['index_input'].to_numpy(np.int64)
        if len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(scene):
            raise ValueError(f'case {case}: rendered ids are not scene rows')
        if not (np.allclose(scene['RA'].to_numpy()[ids], plus['RA_input'], rtol=0, atol=1e-9)
                and np.allclose(scene['DEC'].to_numpy()[ids], plus['DEC_input'], rtol=0, atol=1e-9)):
            raise ValueError(f'case {case}: rendered positions differ from the scene rows')
        unrendered = np.setdiff1d(np.arange(len(scene)), ids)
        crowd = physical_close_crowding(pd.DataFrame({'RA': plus['RA_input'], 'DEC': plus['DEC_input'],
                                                      'r': plus['r_input']}), np.arange(len(plus)))
        q = plus['axis_ratio_input'].to_numpy(float)
        context = np.column_stack([plus['e1_input_rot0'], plus['e2_input_rot0'], plus['sersic_n_input'],
                                   plus['r_input'], plus['Re_input'].to_numpy(float)*np.sqrt(q), crowd])
        if context.shape[1] != len(FEATURES) or not np.isfinite(context).all():
            raise ValueError(f'case {case}: invalid parent context')
        path = OUT/f'parent/case{case}.npz'
        np.savez(path, input_index=ids, context=context)
        record = {'case': case, 'scene_rows': len(scene), 'rendered_rows': len(ids),
                  'unrendered_dropped': unrendered.tolist(), 'max_true_mag': float(plus['r_input'].max()),
                  'sha256': sha256(path), 'truth_sha256': {n: sha256(p) for n, (_, p) in t.items()}}
        path.with_suffix('.json').write_text(json.dumps(record, indent=1)+'\n')
        print(f'PREP case={case} rendered={len(ids)} dropped={len(unrendered)} '
              f'seconds={time.monotonic()-started:.0f}', flush=True)


def pairs_check():
    """Rebuild the saved V3.9 ConstGold case-40 pair table (BlendEMU rule, FS2 conditions) with pair_table."""
    sys.path.insert(0, str(REPO))
    import pandas as pd
    root = DATA/'sbsi_caches/constgold_truth_parent_unbounded_20260922_v1'
    saved = pd.read_feather(root/'pairs/case040.feather')
    meta = json.loads((root/'pairs/case040.json').read_text())
    t = pd.read_feather(DATA/'lsst_sims_fs2_25876_constant/case40_0.02/real0/catalogues/input/'
                        f'gals_info_{TILE}.feather').sort_values('index_input', kind='stable').reset_index(drop=True)
    with np.load(root/'parent/case040.npz') as z:
        parent = np.asarray(z['input_index'], np.int64)
    rows = np.flatnonzero(np.isin(t['index_input'].to_numpy(np.int64), parent))
    if len(rows) != len(parent):
        raise ValueError('FS2 parent not found in gals_info')
    rebuilt = pair_table(scene_frame(t), rows, meta['conditions'])
    saved = saved.sort_values(['index_input_p', 'index_input_s'], kind='stable').reset_index(drop=True)
    same_pairs = (len(saved) == len(rebuilt) and np.array_equal(saved['index_input_p'], rebuilt['index_input_p'])
                  and np.array_equal(saved['index_input_s'], rebuilt['index_input_s']))
    result = {'saved_pairs': len(saved), 'rebuilt_pairs': len(rebuilt), 'identical_pair_identities': bool(same_pairs)}
    if same_pairs:
        result['max_abs_feature_difference'] = {
            name: float(np.max(np.abs(saved[name].to_numpy(float)-rebuilt[name].to_numpy(float))))
            for name in meta['features']}
    (OUT/'pairs_check.json').write_text(json.dumps(result, indent=1)+'\n')
    print('PAIRS_CHECK', json.dumps(result), flush=True)
    if not same_pairs or max(result['max_abs_feature_difference'].values()) > 1e-6:
        raise ValueError('pair rebuild does not reproduce the saved V3.9 table')


def c8(chunk):
    sys.path.insert(0, str(C8_CHECKOUT))
    import pandas as pd
    import torch
    from scripts.joint_response_axes import shear_frames
    from sbsi.selection_model import load_selection_model
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    _, paths, _ = preset()
    model = load_selection_model(paths['classifier'], device='cpu')
    (OUT/'c8').mkdir(exist_ok=True)
    for case in chunk_cases(chunk):
        with np.load(OUT/f'parent/case{case}.npz') as z:
            ids, context = z['input_index'], z['context']
        out = {}
        for name, frame in zip(('plus', 'minus'), shear_frames(full_context(context), H, AXIS)):
            frame = frame.drop(columns=['nbr_flux_near', 'nbr_flux_far', 'nbr_flux_max'])
            if list(frame.columns) != FEATURES:
                raise ValueError('unexpected C8 input order')
            q = np.asarray(model.predict_proba(frame), float)
            if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
                raise ValueError('invalid probabilities')
            out[f'probability_{name}'] = q
        np.savez(OUT/f'c8/case{case}.npz', input_index=ids, **out)
        print(f'C8 case={case} mean_p_plus={out["probability_plus"].mean():.4f}', flush=True)


def draws(chunk):
    import pandas as pd
    import torch
    import scripts.evaluate_smooth_classifier_joint as ev
    from scripts import joint_response_axes as axes
    from sbsi.disk_response_transport import disk_response_transport
    from sbsi.flow_guard_response import physical_context_draws
    from sbsi.measurement_model import load_measurement_model
    from sbsi.output_conditioned_response import DiskResponseMoment, predict_scene_draw_responses
    if Path(ev.__file__).resolve().parents[1] != F8_CHECKOUT:
        raise ValueError(f'expected the frozen F8 checkout, got {ev.__file__}')
    if ev.CUTS != CUT_NAMES[:3] or ev.FLUX_MINIMUM != 10**(-0.4*(25.8-30)):
        raise ValueError('unexpected frozen cut definition')
    _, paths, conditions = preset()
    device = torch.device('cuda')
    candidate = DiskResponseMoment(paths['emulator'].parent, nthread=4, device='cuda')
    if not candidate.conditioned or candidate.response_coordinate != 'disk_velocity':
        raise ValueError('expected the output-conditioned Mobius disk response')
    bundle = load_measurement_model(paths['flow'], device=device)
    bundle.model.eval()
    (OUT/'sums').mkdir(exist_ok=True)
    print('GPU', torch.cuda.get_device_name(0), flush=True)
    for case in chunk_cases(chunk):
        started = time.monotonic()
        with np.load(OUT/f'parent/case{case}.npz') as z, np.load(OUT/f'c8/case{case}.npz') as c:
            ids, context = z['input_index'], z['context'].astype(np.float64)
            if not np.array_equal(ids, c['input_index']):
                raise ValueError('parent and C8 rows differ')
            probability = {d: c[f'probability_{d}'] for d, _ in LEGS}
        if not np.all(np.diff(ids) > 0):
            raise ValueError('parent ids must be sorted')
        t, _ = truth(case, H)
        if not np.array_equal(t['index_input'].to_numpy(np.int64), ids):
            raise ValueError('parent rows differ from gals_info')
        pairs = pair_table(scene_frame(t), np.arange(len(t)), conditions)
        pair_ids = pairs['index_input_p'].to_numpy()
        frames = dict(zip(('plus', 'minus'), axes.shear_frames(full_context(context), H, AXIS)))
        contexts = [bundle.context_tensor(frames[d]) for d in ('plus', 'minus')]
        # per cut: mass, flow e1, e2, blend e1, e2 over all draws, then the same over draws kept in both legs
        rows = {d: np.zeros((len(ids), len(CUTS), 10)) for d, _ in LEGS}
        with torch.no_grad():
            for block, start in enumerate(range(0, len(ids), BATCH)):
                stop = min(len(ids), start+BATCH)
                generated = physical_context_draws(bundle.model, [x[start:stop] for x in contexts], DRAWS,
                                                   SEED+10000019*case+1000003*block)
                lo = np.searchsorted(pair_ids, ids[start], side='left')
                hi = np.searchsorted(pair_ids, ids[stop-1], side='right')
                keep = [output_cuts(values) for values in generated]
                both = keep[0] & keep[1]
                for direction, values, kept in zip(('plus', 'minus'), generated, keep):
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError('nonfinite generated measurements')
                    response, _ = predict_scene_draw_responses(candidate, pairs.iloc[lo:hi], ids[start:stop],
                                                               values.cpu().numpy())
                    shear = torch.tensor(axes.shear_vector(H if direction == 'plus' else -H, AXIS),
                                         dtype=torch.float64, device=values.device)
                    velocity = torch.as_tensor(response, dtype=torch.float64, device=values.device)[..., None]*shear
                    transformed, logdet = disk_response_transport(values[..., :2], velocity)
                    if not bool(torch.isfinite(transformed).all() and torch.isfinite(logdet).all()):
                        raise ValueError('nonfinite disk-response transport')
                    shift = transformed-values[..., :2]
                    rows[direction][start:stop] = np.concatenate(
                        [np.concatenate([mass[..., None], flow, blend], axis=-1)
                         for mass, flow, blend in ((x.cpu().numpy() for x in draw_moments(k, values, shift))
                                                   for k in (kept, both))], axis=-1)
        bins = np.digitize(context[:, 3], MAG_EDGES)
        measured, reports = {}, {}
        for direction, shear in LEGS:
            values, usable, reports[direction] = ev.measured_leg(SIMS, case, shear, ids)
            measured[direction] = values, usable, output_cuts(values) & usable[:, None]
        usable_both = (measured['plus'][1] & measured['minus'][1]).astype(float)
        sim_both = measured['plus'][2] & measured['minus'][2]
        # weight (C8, true flags), leg, cut, true-mag bin, quantity; the both-leg part (5:) uses the true
        # usable flags of both legs for either weight, as C8 gives no joint probability
        model_sums = np.zeros((2, 2, len(CUTS), 4, 10))
        sim_sums = np.zeros((2, len(CUTS), 4, 6))   # leg, cut, bin, (count, e1, e2) all then kept in both legs
        for l, (direction, _) in enumerate(LEGS):
            values, usable, keep = measured[direction]
            for w, weight in enumerate((probability[direction], usable.astype(float))):
                for k in range(len(CUTS)):
                    for b in range(4):
                        sel = bins == b
                        model_sums[w, l, k, b, :5] = weight[sel] @ rows[direction][sel, k, :5]
                        model_sums[w, l, k, b, 5:] = usable_both[sel] @ rows[direction][sel, k, 5:]
            for k in range(len(CUTS)):
                for b in range(4):
                    sel, common = keep[:, k] & (bins == b), sim_both[:, k] & (bins == b)
                    sim_sums[l, k, b] = (sel.sum(), *values[sel, :2].sum(0),
                                         common.sum(), *values[common, :2].sum(0))
        np.savez(OUT/f'sums/case{case}.npz', model=model_sums, sim=sim_sums, cuts=np.array(CUT_NAMES))
        record = {'case': case, 'parents': len(ids), 'pairs': len(pairs), 'measured': reports,
                  'gpu': torch.cuda.get_device_name(0), 'seconds': time.monotonic()-started}
        (OUT/f'sums/case{case}.json').write_text(json.dumps(record, indent=1)+'\n')
        print(f'DRAWS case={case} parents={len(ids)} pairs={len(pairs)} '
              f'seconds={record["seconds"]:.0f}', flush=True)


def summary():
    model, sim, missing = [], [], []
    for case in CASES:
        path = OUT/f'sums/case{case}.npz'
        if not path.is_file():
            missing.append(case)
            continue
        with np.load(path) as z:
            model.append(z['model'])
            sim.append(z['sim'])
    if missing:
        raise ValueError(f'missing cases {missing}')
    model, sim = np.array(model), np.array(sim)   # (case, weight, leg, cut, bin, q), (case, leg, cut, bin, q)
    n = len(CASES)
    weights = np.random.default_rng(20260925).multinomial(n, np.full(n, 1/n), size=10000)

    def response(sums):   # sums (..., leg, 3): count/mass, e1, e2 -> (R11, R21)
        mean = sums[..., 1:]/sums[..., :1]
        return (mean[..., 0, :]-mean[..., 1, :])/(2*H)

    def estimate(sim_sel, model_sel):
        r_sim, r_model = response(sim_sel.sum(0)), response(model_sel.sum(0))
        boot_sim = response(np.einsum('bc,c...->b...', weights, sim_sel))
        boot_model = response(np.einsum('bc,c...->b...', weights, model_sel))
        m = 100*(r_sim[0]/r_model[0]-1)
        boot_m = 100*(boot_sim[:, 0]/boot_model[:, 0]-1)
        se = float(boot_m.std(ddof=1))
        diff21 = boot_sim[:, 1]-boot_model[:, 1]
        return {'m_percent': float(m), 'case_se_pp': se, 'sigma_from_zero': float(m/se),
                'ci95_pp': np.quantile(boot_m, [.025, .975]).tolist(),
                'R11_sim': [float(r_sim[0]), float(boot_sim[:, 0].std(ddof=1))],
                'R11_model': [float(r_model[0]), float(boot_model[:, 0].std(ddof=1))],
                'R21_sim': [float(r_sim[1]), float(boot_sim[:, 1].std(ddof=1))],
                'R21_model': [float(r_model[1]), float(boot_model[:, 1].std(ddof=1))],
                'R21_sim_minus_model': [float(r_sim[1]-r_model[1]), float(diff21.std(ddof=1))],
                'selected_count_ratio_model_over_sim': (model_sel.sum(0)[:, 0]/sim_sel.sum(0)[:, 0]).tolist()}

    def model_view(w, k, b, part, both=False):
        x = model[:, w, :, k] if b is None else model[:, w, :, k, b][..., None, :]
        x = x[..., 5:] if both else x[..., :5]
        shape = {'total': x[..., 1:3]+x[..., 3:5], 'flow only': x[..., 1:3]}[part]
        return np.concatenate([x[..., :1], shape], axis=-1).sum(-2)

    def sim_view(k, b, both=False):
        x = sim[:, :, k] if b is None else sim[:, :, k, b][..., None, :]
        return (x[..., 3:] if both else x[..., :3]).sum(-2)

    def split(k):
        """True-flag m at cut k as shape part (galaxies/draws kept in both legs) + selection part (the rest)."""
        parts = []
        for sums in (lambda s, m: (s.sum(0), m.sum(0)), lambda s, m: (np.einsum('bc,c...->b...', weights, s),
                                                         np.einsum('bc,c...->b...', weights, m))):
            rs, rm = (response(x)[..., 0] for x in sums(sim_view(k, None), model_view(1, k, None, 'total')))
            bs, bm = (response(x)[..., 0] for x in sums(sim_view(k, None, True),
                                                        model_view(1, k, None, 'total', True)))
            parts.append((100*(bs-bm)/rm, 100*((rs-bs)-(rm-bm))/rm, bs, bm))
        (shape, selection, bs, bm), (boot_shape, boot_selection, _, _) = parts
        both = sim_view(k, None, True).sum(0)[0, 0]/sim_view(k, None).sum(0)[0, 0]
        return {'shape_part_pp': [float(shape), float(boot_shape.std(ddof=1))],
                'selection_part_pp': [float(selection), float(boot_selection.std(ddof=1))],
                'R11_both_legs_sim': float(bs), 'R11_both_legs_model': float(bm),
                'sim_fraction_kept_in_both_legs': float(both)}

    out = {'definitions': __doc__, 'cases': [CASES[0], CASES[-1]], 'cuts': CUTS, 'results': {}, 'split': {}}
    labels = ('<24', '24-25.5', '25.5-27.5', '>=27.5')
    for w, weight in enumerate(('C8', 'true flags')):
        for k, cut in enumerate(CUT_NAMES):
            out['results'][f'{weight}/{cut}'] = estimate(sim_view(k, None), model_view(w, k, None, 'total'))
            for b, label in enumerate(labels):
                out['results'][f'{weight}/{cut}/true mag {label}'] = estimate(sim_view(k, b),
                                                                              model_view(w, k, b, 'total'))
        out['results'][f'{weight}/{CUT_NAMES[1]}/flow only (no emulator)'] = estimate(
            sim_view(1, None), model_view(w, 1, None, 'flow only'))
    for k, cut in enumerate(CUT_NAMES):
        out['split'][cut] = split(k)
    (OUT/'summary.json').write_text(json.dumps(out, indent=1)+'\n')
    for name, r in out['results'].items():
        if '/true mag' in name:
            continue
        print(f"{name:55s} m = {r['m_percent']:+.3f}% +- {r['case_se_pp']:.3f} ({r['sigma_from_zero']:+.1f} sigma)  "
              f"R11 sim {r['R11_sim'][0]:.4f} model {r['R11_model'][0]:.4f}  "
              f"R21 sim-model {r['R21_sim_minus_model'][0]:+.5f} +- {r['R21_sim_minus_model'][1]:.5f}", flush=True)
    for cut, r in out['split'].items():
        print(f"split {cut:25s} shape {r['shape_part_pp'][0]:+.3f} +- {r['shape_part_pp'][1]:.3f} pp  "
              f"selection {r['selection_part_pp'][0]:+.3f} +- {r['selection_part_pp'][1]:.3f} pp  "
              f"kept in both legs {r['sim_fraction_kept_in_both_legs']:.3f}", flush=True)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('run through Slurm')
    stage = sys.argv[1]
    if stage in ('prep', 'c8', 'draws'):
        {'prep': prep, 'c8': c8, 'draws': draws}[stage](int(sys.argv[2]))
    else:
        {'pairs-check': pairs_check, 'summary': summary}[stage]()
