"""Paired +/-0.02 model mock made from the TRUE ConstGold (Flagship-2) galaxies, for inference with the GalSBI prior.

Owner question 2026-10-31: is the measured paired m = -5.5% caused by the prior population (GalSBI F24) differing from
the ConstGold population (Flagship-2)? This mock keeps the population of the images but replaces the images by the
symmetric F8 + C8 + disk model the inference uses. If its paired inference gives ~-6%, the prior/population mismatch is
the cause; if ~0, the cause is between the model and the images.

Per case 40-119 (the cases of the forward-response test), with the full truth catalogue as the scene:
  targets: a uniform random subset of all input galaxies (no truth cuts);
  model inputs: e1/e2_input_rot0, sersic n, r, circularized Re = Re*sqrt(q), physical_close_crowding over all
    galaxies within 10 arcsec (training projection) -- checked against the cached ConstGold truth parent where it overlaps;
  disk response pairs: nearest 20 neighbours within 10 arcsec, the inference contract (response_pairs in galsbi_summary),
    semimajor Re_input as given by the catalogue;
  each leg: the inference's own mock recipe (check_galsbi_summary_sampler.mock_panel): C8 usable Bernoulli at the sheared
    shape, one F8 draw, measured radius > 3 px and mag < 25.8, disk transport b*g; then the image runs' final cuts
    (radius < 100 px, |e| < 0.99). Common random numbers across legs: the same usable uniforms and the same flow latents
    (same seed, same rows, same batches), like the shared noise image of the two ConstGold legs.
Run under Slurm on one GPU from the frozen source snapshot (cwd), with '.' and 'scripts' on sys.path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from signsym_candidate import CANDIDATE, IDENTITY
from sbsi.catalogue_disk_response import CatalogueDiskResponse
from sbsi.catalogue_likelihood import predict_detection_probability
from sbsi.disk_precision import LOG_GAP_COLUMN, transport_with_log_gap
from sbsi.forward_catalogue import rescale_emulator_pairs
from sbsi.measurement_model import load_measurement_model
from sbsi.models import load_detection_classifier, load_disk_response
from sbsi.output_conditioned_response import TRUTH_FEATURES
from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.v39_inference import FEATURES, physical_close_crowding

SIMS = Path('/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant')
PARENT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_closecrowd_20260924_v1/parent')
TEMPLATE = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028/recovery_v1/reference')
COLUMNS = ['index_input', 'RA_input', 'DEC_input', 'Re_input', 'axis_ratio_input', 'sersic_n_input', 'r_input',
           'e1_input_rot0', 'e2_input_rot0', 'gamma1_input', 'gamma2_input']
PARENT_COLUMNS = [0, 1, 2, 3, 4, 8, 9, 10]   # parent 'features' list -> the eight V3.9 inputs


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def scene(case):
    plus = pd.read_feather(SIMS/f'case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather', columns=COLUMNS)
    minus = pd.read_feather(SIMS/f'case{case}_-0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather', columns=COLUMNS)
    same = COLUMNS[:9]
    pd.testing.assert_frame_equal(plus[same], minus[same], check_exact=True)
    if not (np.allclose(plus[['gamma1_input', 'gamma2_input']], [.02, 0], atol=1e-14, rtol=0)
            and np.allclose(minus[['gamma1_input', 'gamma2_input']], [-.02, 0], atol=1e-14, rtol=0)):
        raise ValueError(f'case {case}: legs are not the declared +/-0.02')
    t = plus[same].sort_values('index_input').reset_index(drop=True)
    if t.index_input.duplicated().any() or not np.isfinite(t[same[1:]].to_numpy()).all():
        raise ValueError(f'case {case}: duplicate or nonfinite truth rows')
    return t


def inputs(t, rows):
    """Eight V3.9 inputs for target rows of the full scene t (all galaxies are neighbours)."""
    crowd = physical_close_crowding(t.rename(columns={'RA_input': 'RA', 'DEC_input': 'DEC', 'r_input': 'r'})
                                    [['RA', 'DEC', 'r']], rows, workers=8)
    x = t.iloc[rows]
    zero = pd.DataFrame(np.column_stack((x.e1_input_rot0, x.e2_input_rot0, x.sersic_n_input, x.r_input,
                                         x.Re_input*np.sqrt(x.axis_ratio_input), crowd)), columns=FEATURES)
    if not np.isfinite(zero.to_numpy()).all():
        raise ValueError('nonfinite model inputs')
    return zero


def parent_check(case, t, rows, zero):
    """Compare with the cached forward-test inputs where the target lies in the cached half."""
    with np.load(PARENT/f'case{case:03d}.npz', allow_pickle=False) as z:
        ids, context = z['input_index'], z['context'].astype(np.float64)[:, PARENT_COLUMNS]
    target_ids = t.index_input.to_numpy()[rows]
    k = np.searchsorted(ids, target_ids)
    hit = (k < len(ids)) & (ids[np.minimum(k, len(ids)-1)] == target_ids)
    if hit.sum() < 100:
        raise ValueError(f'case {case}: too few targets in the cached parent ({hit.sum()})')
    diff = np.abs(zero.to_numpy()[hit]-context[k[hit]])
    scale = np.maximum(1., np.abs(context[k[hit]]))
    worst = (diff/scale).max(0)
    if worst.max() > 1e-5:   # the cache stores float32
        raise ValueError(f'case {case}: inputs differ from the cached parent: {dict(zip(FEATURES, worst))}')
    return int(hit.sum()), worst.tolist()


def pairs(t, rows, conditions):
    """Nearest 20 neighbours within 10 arcsec (training projection), as galsbi_summary.response_pairs."""
    xy = t[['RA_input', 'DEC_input']].to_numpy(float).copy()
    xy[:, 0] *= np.cos(np.deg2rad(xy[:, 1].mean()))
    xy *= 3600.
    tree = cKDTree(xy)
    d, j = tree.query(xy[rows], k=21, distance_upper_bound=10., workers=8)
    owner = np.repeat(np.arange(len(rows)), 21)
    d, j = d.ravel(), j.ravel()
    use = np.isfinite(d) & (j != np.repeat(rows, 21)) & (d < 10.)
    owner, d, j = owner[use], d[use], j[use]
    order = np.lexsort((j, d, owner))
    owner, d, j = owner[order], d[order], j[order]
    _, starts, counts = np.unique(owner, return_index=True, return_counts=True)
    rank = np.arange(len(owner))-np.repeat(starts, counts)
    owner, d, j = owner[rank < 20], d[rank < 20], j[rank < 20]
    p, s = t.iloc[rows[owner]], t.iloc[j]
    frame = pd.DataFrame(dict(Re_input_p=p.Re_input.to_numpy(), Re_input_s=s.Re_input.to_numpy(),
        r_input_p=p.r_input.to_numpy(), r_input_s=s.r_input.to_numpy(), sersic_n_input_p=p.sersic_n_input.to_numpy(),
        sersic_n_input_s=s.sersic_n_input.to_numpy(), distance=d))
    features = rescale_emulator_pairs(frame, conditions).loc[:, TRUTH_FEATURES].to_numpy(float)
    if not np.isfinite(features).all():
        raise ValueError('nonfinite pair features')
    return np.r_[0, np.cumsum(np.bincount(owner, minlength=len(rows)))], features


def leg(zero, response, flow, detector, uniform, seed, shear):
    """check_galsbi_summary_sampler.mock_panel with shared uniforms and latents for every target."""
    context = zero.copy()
    context['e1_input_p'], context['e2_input_p'] = apply_shear_to_ellipticity(
        zero.e1_input_p.to_numpy(), zero.e2_input_p.to_numpy(), *shear)
    usable = uniform < predict_detection_probability(detector, context)
    torch.manual_seed(seed)
    draws, gap = flow.sample_with_shape_log_gap(context, batch_size=4096)
    draws, gap = draws[:, 0], gap[:, 0]
    keep = np.flatnonzero(usable & (draws[:, 2] > 3.) & (draws[:, 3] > 10**(.4*(30-25.8))))
    draws, gap = draws[keep], gap[keep]
    b, _ = response.predict(keep, draws[:, 2], draws[:, 3])
    device = flow.device
    moved, gap, _ = transport_with_log_gap(torch.as_tensor(draws[:, :2], device=device),
        torch.as_tensor(gap, device=device), torch.as_tensor(b[:, None]*np.array(shear), device=device))
    draws[:, :2] = moved.cpu().numpy()
    frame = pd.DataFrame(draws.astype(np.float64), columns=flow.target_transform.target_names)
    frame[LOG_GAP_COLUMN] = gap.cpu().numpy().astype(np.float64)
    final = (frame.iloc[:, 2].to_numpy() < 100) & (np.hypot(frame.iloc[:, 0], frame.iloc[:, 1]).to_numpy() < .99)
    return frame.loc[final].reset_index(drop=True), keep[final], dict(
        usable=int(usable.sum()), measured_cuts=len(keep), final=int(final.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--targets-per-case', type=int, default=11000)
    parser.add_argument('--cases', type=int, nargs=2, default=(40, 120), help='first, last+1')
    parser.add_argument('--seed', type=int, default=2026103101)
    args = parser.parse_args()
    template = json.loads((TEMPLATE/'protocol.json').read_text())
    conditions = template['conditions']
    if template.get('model_identity') != IDENTITY:
        raise ValueError('template reference uses a different model')
    flow = load_measurement_model(CANDIDATE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(CANDIDATE, device='cuda')
    model = load_disk_response(CANDIDATE, device='cpu', nthread=8)
    out = {'plus': [], 'minus': []}
    reports = []
    for case in range(*args.cases):
        t = scene(case)
        rng = np.random.default_rng([args.seed, case])
        rows = np.sort(rng.choice(len(t), args.targets_per_case, replace=False))
        zero = inputs(t, rows)
        overlap, worst = parent_check(case, t, rows, zero)
        ptr, features = pairs(t, rows, conditions)
        response = CatalogueDiskResponse(ptr, features, model)
        uniform = rng.random(len(rows))
        latent_seed = int(rng.integers(0, 2**62))
        report = dict(case=case, scene_rows=len(t), targets=len(rows), parent_overlap=overlap,
                      parent_max_relative_difference=worst, response_pairs=len(features))
        for name, g in (('plus', (.02, 0.)), ('minus', (-.02, 0.))):
            frame, kept, counts = leg(zero, response, flow, detector, uniform, latent_seed, g)
            frame.insert(0, 'source_input_index', t.index_input.to_numpy()[rows[kept]])
            frame.insert(0, 'source_case', case)
            out[name].append(frame)
            report[name] = counts
        reports.append(report)
        print('MOCK_CASE', json.dumps(report), flush=True)
    for name, injected in (('plus', .02), ('minus', -.02)):
        all_rows = pd.concat(out[name], ignore_index=True)
        all_rows = all_rows.iloc[np.random.default_rng([args.seed, 1 if name == 'plus' else 2]).permutation(
            len(all_rows))].reset_index(drop=True)
        if all_rows.duplicated(['source_case', 'source_input_index']).any():
            raise ValueError('duplicate galaxies')
        obs_columns = list(pd.read_parquet(TEMPLATE/'observations.parquet').columns)
        if list(all_rows.columns[2:]) != obs_columns:
            raise ValueError(f'observation columns {list(all_rows.columns[2:])} differ from the template {obs_columns}')
        reference = args.root/name/'reference'
        reference.mkdir(parents=True)
        all_rows[obs_columns].to_parquet(reference/'observations.parquet', index=False)
        truth = all_rows[['source_case', 'source_input_index']].copy()
        truth['injected_g1'], truth['injected_g2'] = injected, 0.
        truth.to_parquet(reference/'truth.parquet', index=False)
        protocol = dict(template, mock_kind='model_from_constgold_truth', source=str(args.root),
            selection=('uniform random targets from the full truth catalogue of each case; per leg C8 Bernoulli, F8 draw, '
                       'radius>3 px, mag<25.8, disk transport, radius<100 px, |e|<0.99; shared uniforms and latents'),
            comparison_reference=str(TEMPLATE), mock_builder_sha256=digest(__file__))
        for key in ('reused_c1_prefix', 'source_manifest_sha256'):
            protocol.pop(key, None)
        save(reference/'protocol.json', protocol)
        save(reference/'result.json', dict(status='prepared_model_mock_from_constgold_truth', preparation_checks_pass=True,
            observation_count=len(all_rows), cases=int(truth.source_case.nunique()),
            observations_sha256=digest(reference/'observations.parquet'), truth_sha256=digest(reference/'truth.parquet'),
            protocol_sha256=digest(reference/'protocol.json'),
            limitation='Model-made measurements of the true ConstGold galaxies; known shear'))
        print('MOCK_LEG', name, len(all_rows), flush=True)
    save(args.root/'mock_report.json', dict(cases=reports, targets_per_case=args.targets_per_case, seed=args.seed,
                                            model_identity=IDENTITY, builder_sha256=digest(__file__)))
    print('MOCK_READY', flush=True)


if __name__ == '__main__':
    main()
