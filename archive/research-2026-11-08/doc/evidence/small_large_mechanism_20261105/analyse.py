"""CPU-only mechanism analysis of existing, replay-verified constant scenes."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

O = Path(__file__).resolve().parent
D = Path('/project/ls-gruen/users/zekang.zhang')
BASE = D / 'sbsi/galsbi_resim'
OLD = BASE / 'constant_g002/sims'
STUDY = BASE / 'constant_split_20261105'
CASES = [1000, 1040, 1080, 1120]
SIZE = ['<0.15', '0.15-0.25', '0.25-0.4', '0.4-0.7', '>=0.7']
CROWD = ['<0.01', '0.01-0.1', '0.1-0.3', '>=0.3']
MAG = ['<23', '23-24', '24-25.5', '>=25.5']
NCELL = 80
OWN = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) / .08
OPP = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]]) / .08


def measured(root, case, h, ids):
    p = root / f'case{case}_{h:.2f}/real0/catalogues'
    cross = pd.read_feather(p / 'CrossMatch/tile180.0_-0.5_rot0_matched.feather')
    shape = pd.read_feather(p / 'Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather')
    assert cross.id_input.is_unique and cross.id_detec.is_unique and shape.NUMBER.is_unique
    joined = cross.merge(shape, left_on='id_detec', right_on='NUMBER', validate='one_to_one')
    frame = joined.set_index('id_input').reindex(ids)
    v = frame[['NGMIX_G1', 'NGMIX_G2', 'FLUX_RADIUS', 'MAG_AUTO']].to_numpy(float)
    v[:, 3] = 10 ** (-.4 * (v[:, 3] - 30))
    u = np.isfinite(v).all(1) & (np.square(v[:, :2]).sum(1) < 1) & (v[:, 2] > 0) & (v[:, 3] > 0)
    receipt = dict(parent=len(ids), cross=len(cross), shapes=len(shape),
                   cross_without_shape=len(cross)-len(joined), shape_without_cross=len(shape)-len(joined),
                   parent_absent_crossmatch=int((~np.isin(ids, cross.id_input)).sum()), usable=int(u.sum()))
    return v, u, frame, receipt


def entry(x):
    valid = np.isfinite(x[1:])
    # Do not silently drop empty bootstrap cells or report a conditional SE.
    return dict(value=float(x[0]) if np.isfinite(x[0]) else None,
                se=float(x[1:].std(ddof=1)) if valid.all() else None,
                **({'undefined_bootstrap_resamples': int((~valid).sum())} if not valid.all() else {}))


def psf_summary():
    noise = pd.read_csv(OLD/'noise.csv').iloc[0]
    beta, fwhm = float(noise.InputBeta_r), float(noise.InputSeeing_r)
    assert noise.seeing_e1_r == noise.seeing_e2_r == 0
    alpha = fwhm/(2*np.sqrt(2**(1/beta)-1))
    trunc = 4.5*fwhm  # Original ImSimPSF.MoffatPSF implementation.
    outer = (1+(trunc/alpha)**2)**(1-beta)
    half = alpha*np.sqrt((.5*(1+outer))**(1/(1-beta))-1)
    return dict(beta=beta, fwhm_arcsec=fwhm, truncation_arcsec=trunc,
        half_light_radius_arcsec=float(half), half_light_radius_pixels=float(half/.2),
        note='Circular truncated Moffat before pixel convolution; not a SExtractor FLUX_RADIUS prediction.')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    sums, models, descriptions, alignments, edges, receipts = [], [], [], [], [], []
    for case in CASES:
        with np.load(STUDY / f'model/case{case}.npz') as z:
            ids, group, ctx, mom = z['input_index'], z['group'], z['context'], z['moments']
        sizebin = np.digitize(ctx[:, 4], [.15, .25, .4, .7])
        cell = (sizebin * 4 + np.digitize(ctx[:, 6], [.01, .1, .3])) * 4 + np.digitize(ctx[:, 3], [23, 24, 25.5])
        vals, us, frames, rec = [], [], [], []
        for root, h in [(OLD, .02), (STUDY/'sims', .02), (STUDY/'sims', -.02), (OLD, -.02)]:
            v, u, f, r = measured(root, case, h, ids)
            vals.append(v); us.append(u); frames.append(f); rec.append(r)
        vals, us = np.array(vals), np.array(us)
        common = us.all(0)
        # actual/common U, state, group, truth cell, count / e1 numerator.
        ss = np.zeros((2, 4, 2, NCELL, 2)); mm = np.zeros_like(ss)
        cut = (vals[:, :, 2] > 3) & (vals[:, :, 3] > 100)
        for mode in range(2):
            for state in range(4):
                u = us[state] if mode == 0 else common
                for g in [0, 1]:
                    rows = u & (group == g)
                    sel = rows & cut[state]
                    ss[mode, state, g, :, 0] = np.bincount(cell[sel], minlength=NCELL)
                    ss[mode, state, g, :, 1] = np.bincount(cell[sel], weights=vals[state, sel, 0], minlength=NCELL)
                    for j, q in enumerate([mom[state, :, 3, 0], mom[state, :, 3, 1]+mom[state, :, 3, 3]]):
                        mm[mode, state, g, :, j] = np.bincount(cell[rows], weights=q[rows], minlength=NCELL)
        with np.load(D/f'sbsi_caches/galsbi_dev_constant_grid_20261104/sums/case{case}.npz') as old:
            k = list(old['cuts']).index('mag25_radius_gt0.60')
            for leg, state in enumerate([0, 3]):
                np.testing.assert_allclose(ss[0, state].sum((0, 1)), old['sim'][leg, k].sum(0)[[0, 1]], atol=2e-7, rtol=2e-10)
                oldm = old['model'][1, leg, k].sum(0)
                np.testing.assert_allclose(mm[0, state].sum((0, 1)), [oldm[0], oldm[1]+oldm[3]], atol=2e-7, rtol=2e-10)
        sums.append(ss); models.append(mm); receipts.append(dict(case=case, states=rec, common=int(common.sum())))
        # Per-leg descriptive sums; repeated leg observations are not independent.
        dd = np.zeros((NCELL, 15)); aa = np.zeros((5, 2, 10, 4))
        for state in [0, 3]:
            v, f = vals[state], frames[state]
            sel = us[state] & cut[state]
            snr = (f.FLUX_AUTO/f.FLUXERR_AUTO).to_numpy()
            ratio = v[:, 3] / 10**(-.4*(ctx[:, 3]-30))
            flags = f.FLAGS.fillna(0).to_numpy(int)
            fields = [np.ones(len(ids)), v[:, 2], snr, ratio,
                      v[:, 2] < 3.2, v[:, 2] < 3.5, snr < 10, snr < 20,
                      ratio > 1.5, ratio > 2, flags != 0, (flags & 2) != 0,
                      f.distance_pixel_CM.to_numpy(), ctx[:, 3], ctx[:, 6]]
            for j, value in enumerate(fields):
                assert np.isfinite(value[sel]).all()
                dd[:, j] += np.bincount(cell[sel], weights=value[sel], minlength=NCELL)
            # Alignment slope dot(e_measured,e_intrinsic)/|e_intrinsic|^2.
            # Pool +/- shear symmetrically; descriptive, not a zero-shear test.
            rbin = np.digitize(v[:, 2], [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.2, 3.5])
            ab = (sizebin*2+(ctx[:, 6]>=.1))*10+rbin
            keep = us[state] & (v[:, 3] > 100)
            dot = (v[:, :2]*ctx[:, :2]).sum(1)
            norm = np.square(ctx[:, :2]).sum(1)
            for j, value in enumerate([np.ones(len(ids)), dot, norm, v[:, 2]]):
                aa[..., j] += np.bincount(ab[keep], weights=value[keep], minlength=100).reshape(5, 2, 10)
        descriptions.append(dd); alignments.append(aa)
        # Edges with fixed target own shear, on common U. Cohorts: all, small,
        # small+low crowd, small+higher crowd. Selected ratios normalized within cohort/group.
        small = (ctx[:, 4] >= .15) & (ctx[:, 4] < .4)
        cohorts = [np.ones(len(ids), bool), small, small & (ctx[:, 6] < .1), small & (ctx[:, 6] >= .1)]
        ee = np.zeros((4, 2, 2, 12))
        for p, population in enumerate(cohorts):
            for g in [0, 1]:
                rows = common & population & (group == g)
                for j, (a, b) in enumerate([(0, 1), (2, 3)] if g == 0 else [(0, 2), (1, 3)]):
                    sa, sb = cut[a, rows], cut[b, rows]
                    ea, eb = vals[a, rows, 0], vals[b, rows, 0]
                    ra, rb = vals[a, rows, 2], vals[b, rows, 2]
                    mag = (vals[a, rows, 3] > 100) & (vals[b, rows, 3] > 100)
                    ee[p, g, j] = [sa.sum(), sb.sum(), ea[sa].sum(), eb[sb].sum(),
                        np.sum(.5*(sa.astype(float)+sb)*(ea-eb)),
                        np.sum(.5*(ea+eb)*(sa.astype(float)-sb)),
                        np.sum((sa|sb)&mag), np.sum(((ra>3)!=(rb>3))&mag),
                        np.sum(np.abs(ra[mag]-rb[mag])), mag.sum(),
                        np.sum((np.abs(ra-rb)>.1)&mag), np.sum((np.abs(ra-rb)>.5)&mag)]
        edges.append(ee)
        print('REPLAYED_AND_REDUCED', case, flush=True)
    w = np.vstack([np.ones(4), np.random.default_rng(20260925).multinomial(4, [.25]*4, size=10000)])
    out = dict(cases=CASES, bootstrap='10000 paired case resamples; seed20260925; fixed draws',
               response={}, descriptors={}, alignment={}, opposite_edges={}, receipts=receipts)
    for mode, name in enumerate(['actual_U', 'common_U']):
        ss = np.einsum('bc,csgpq->bsgpq', w, np.array(sums)[:, mode])
        mm = np.einsum('bc,csgpq->bsgpq', w, np.array(models)[:, mode])
        fs = ss[..., 1]/ss[..., 0].sum((2, 3))[:, :, None, None]
        fm = mm[..., 1]/mm[..., 0].sum((2, 3))[:, :, None, None]
        err = fs-fm
        own = np.einsum('bsgp,sg->bp', err, OWN)
        opp = np.einsum('bsgp,sg->bp', err, OPP)
        total = (err[:, 0].sum(1)-err[:, 3].sum(1))/.04
        np.testing.assert_allclose(own+opp, total, atol=2e-12, rtol=0)
        ci = np.arange(NCELL)
        masks = {'all': np.ones(NCELL, bool), 'small': np.isin(ci//16, [1, 2])}
        for i, label in enumerate(SIZE): masks['size/'+label] = ci//16 == i
        for i, label in enumerate(CROWD):
            masks['small_crowd/'+label] = masks['small'] & (ci//4%4 == i)
            for j, mag in enumerate(MAG):
                masks['small_crowd_mag/'+label+'/'+mag] = masks['small'] & (ci//4%4 == i) & (ci%4 == j)
        for j, mag in enumerate(MAG): masks['small_mag/'+mag] = masks['small'] & (ci%4 == j)
        out['response'][name] = {key: {k: entry(a[:, mask].sum(1)) for k, a in [('own', own), ('opposite', opp), ('total', total)]} for key, mask in masks.items()}
    dd = np.einsum('bc,cpq->bpq', w, np.array(descriptions))
    names = ['mean_radius_px', 'mean_snr', 'mean_flux_ratio', 'radius_lt3.2_fraction', 'radius_lt3.5_fraction',
             'snr_lt10_fraction', 'snr_lt20_fraction', 'flux_ratio_gt1.5_fraction', 'flux_ratio_gt2_fraction',
             'nonzero_flags_fraction', 'flag_bit2_fraction', 'mean_match_distance_px', 'mean_true_mag', 'mean_crowding']
    for key, mask in masks.items():
        d = dd[:, mask].sum(1)
        out['descriptors'][key] = dict(selected_leg_rows=int(d[0, 0]), selected_fraction=entry(d[:, 0]/dd[:, :, 0].sum(1)),
            **{n: entry(d[:, j+1]/d[:, 0]) for j, n in enumerate(names)})
    aa = np.einsum('bc,csrjq->bsrjq', w, np.array(alignments))
    for i, size in enumerate(SIZE):
        for c, crowd in enumerate(['<0.1', '>=0.1']):
            out['alignment'][size+'/'+crowd] = [dict(bin=label, n=int(aa[0, i, c, j, 0]),
                slope=entry(aa[:, i, c, j, 1]/aa[:, i, c, j, 2])) for j, label in enumerate(
                    ['<2.4', '2.4-2.5', '2.5-2.6', '2.6-2.7', '2.7-2.8', '2.8-2.9', '2.9-3', '3-3.2', '3.2-3.5', '>=3.5'])]
    ee = np.einsum('bc,cpgjq->bpgjq', w, np.array(edges))
    for p, name in enumerate(['all', 'small', 'small_low_crowd', 'small_higher_crowd']):
        e = ee[:, p]; da, db = e[..., 0], e[..., 1]
        fa, fb = e[..., 2]/da, e[..., 3]/db
        shape = e[..., 4]/(.5*(da+db))
        selection = (e[..., 5]-.5*(fa+fb)*(da-db))/(.5*(da+db))
        np.testing.assert_allclose(shape+selection, fa-fb, atol=2e-12, rtol=0)
        agg = e.sum((1, 2))
        out['opposite_edges'][name] = dict(
            radius_crossings=int(agg[0, 7]), radius_crossing_fraction_union_selected=entry(agg[:, 7]/agg[:, 6]),
            mean_abs_radius_change_mag_both=entry(agg[:, 8]/agg[:, 9]),
            radius_change_gt_0p1_fraction=entry(agg[:, 10]/agg[:, 9]),
            radius_change_gt_0p5_fraction=entry(agg[:, 11]/agg[:, 9]),
            shape_response_AB=[entry(shape[:, g].sum(1)/.08) for g in [0, 1]],
            selection_response_AB=[entry(selection[:, g].sum(1)/.08) for g in [0, 1]])
    out['analytic_psf'] = psf_summary()
    old = json.loads((O.parent/'constant_split_20261105/report.json').read_text())['comparisons']['actual_U']['radius3_mag25']['all']
    for new, prior in [('own','own_group'),('opposite','opposite_group'),('total','coherent_total')]:
        for field in ['value','se']:
            np.testing.assert_allclose(out['response']['actual_U']['all'][new][field], old[prior]['error'][field], atol=2e-12, rtol=0)
    out['validation'] = 'All original coherent catalogue/model sums, total own/opposite errors and bootstrap SEs replay; disjoint cell and exact shape/selection identities pass.'
    (O/'report.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    np.savez(O/'case_sums.npz', sim=sums, model=models, descriptors=descriptions, alignment=alignments, edges=edges)
    print(json.dumps(out['response']['actual_U'], indent=2), flush=True)


if __name__ == '__main__':
    main()
