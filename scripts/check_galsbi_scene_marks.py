#!/usr/bin/env python
"""Replay a model's scene-input contract on written GalSBI scene files (run under Slurm).

The scene writer stored the native prior marks (redshift, M_B, c0..c4, n,
R_kpc, a, psi) next to the rendered inputs (r = LSST-i, sersic_n, Re,
r50_circ). With SceneMarks hooked in, the prior photometry must reproduce
the rendered magnitude, independently of batching, and freshly drawn scenes
must carry it. Writes a JSON receipt with differences and photometry cost.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from check_galsbi_continuous_prior import build_prior
from sbsi.galsbi.likelihood import native_shapes
from sbsi.galsbi.model_inputs import SceneMarks
from sbsi.galsbi.scenes import GalSBIScenePrior
from sbsi.models import get_model


def timed(function, *args):
    started = time.perf_counter()
    out = function(*args)
    return out, time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, nargs='+', required=True)
    parser.add_argument('--model', default='like-1.0-galsbi')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--magnitude-tolerance', type=float, default=1e-9)
    args = parser.parse_args()
    model = get_model(args.model)
    galaxy, cosmo, _ = build_prior()
    distance = lambda z: cosmo.background.dist_ang_a(a=1/(1+np.asarray(z)))
    receipts = []
    for path in args.scenes:
        scene = pd.read_feather(path)
        data = dict(z=scene.redshift.to_numpy(float), M_B=scene.M_B.to_numpy(float),
                    c=scene[[f'c{k}' for k in range(5)]].to_numpy(float), n=scene.n.to_numpy(float),
                    R_kpc=scene.R_kpc.to_numpy(float), a=scene.a.to_numpy(float), psi=scene.psi.to_numpy(float))
        hook, galaxy.model_band = galaxy.model_band, None
        plain, plain_seconds = timed(galaxy.photometry, data['z'], data['M_B'], data['c'])
        galaxy.model_band = hook
        if hook is None:
            marks = SceneMarks(model.scene_inputs, galaxy)
        photo, hooked_seconds = timed(galaxy.photometry, data['z'], data['M_B'], data['c'])
        order = np.random.default_rng(0).permutation(len(scene))
        chunked = np.empty(len(scene))
        for rows in np.array_split(order, 97):
            chunked[rows] = galaxy.photometry(data['z'][rows], data['M_B'][rows], data['c'][rows])['m_model']
        magnitude = marks.magnitude(photo)
        shapes = native_shapes(data, distance)
        receipt = dict(scenes=str(path), rows=len(scene),
            magnitude_max_abs_difference=float(np.abs(magnitude-scene.r.to_numpy(float)).max()),
            magnitude_batch_independent=bool(np.array_equal(chunked, magnitude)),
            other_photometry_unchanged=bool(all(np.array_equal(plain[k], photo[k]) for k in plain)),
            sersic_n_max_abs_difference=float(np.abs(marks.sersic_n(data)-scene.sersic_n.to_numpy(float)).max()),
            semimajor_max_relative_difference=float(np.abs(shapes['semimajor']/scene.Re.to_numpy(float)-1).max()),
            circularized_max_relative_difference=float(np.abs(shapes['circularized']/scene.r50_circ.to_numpy(float)-1).max()),
            native_hsc_r_max_abs_difference=float(np.abs(photo['m_r']-scene.m_r_hsc.to_numpy(float)).max()),
            lsst_i_minus_native_hsc_r=dict(mean=float(np.mean(magnitude-photo['m_r'])),
                                           std=float(np.std(magnitude-photo['m_r']))),
            photometry_seconds_per_million=dict(without_model_band=plain_seconds/len(scene)*1e6,
                                                with_model_band=hooked_seconds/len(scene)*1e6))
        receipt['pass'] = bool(receipt['magnitude_batch_independent'] and receipt['other_photometry_unchanged']
                               and receipt['magnitude_max_abs_difference'] <= args.magnitude_tolerance
                               and receipt['sersic_n_max_abs_difference'] <= 1e-12)
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    # Freshly drawn scenes (targets and neighbours) carry the hooked magnitude, equal to a direct recomputation.
    drawn, _ = GalSBIScenePrior(galaxy).sample(256, 20261108)
    carried = {}
    for part, values in (('targets', drawn.targets), ('neighbours', drawn.neighbours)):
        direct = galaxy.photometry(values['z'], values['M_B'], values['c'])['m_model']
        carried[part] = dict(rows=len(direct), equal=bool(np.array_equal(marks.magnitude(values), direct)))
    result = dict(model=args.model, scene_inputs=model.scene_inputs, receipts=receipts, drawn_scenes=carried,
                  all_pass=all(r['pass'] for r in receipts) and all(c['equal'] for c in carried.values()))
    args.output.write_text(json.dumps(result, indent=1)+'\n')
    print('DRAWN', json.dumps(carried))
    print('SCENE_MARKS_REPLAY', result['all_pass'], flush=True)


if __name__ == '__main__':
    main()
