"""Scheduler-only production-prior support check; no posterior/artifact outcomes."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.source), str(args.source/'scripts')]

import numpy as np
import torch  # Match the production driver: import before PyCosmo/build_prior.
from check_galsbi_continuous_prior import build_prior
from sbsi.galsbi_prior import upper_gamma
from sbsi.scene_exchange import numerical_over_analytic
from sbsi.numerical_native_intensity import numerical_native_log_intensity

galaxy, _, config = build_prior()
logtiny = np.log(np.finfo(float).tiny)
result = dict(source=str(args.source), intrinsic_i_cut=[galaxy.p['gals_mag_min'],
    galaxy.p['gals_mag_max']], log_tiny=float(logtiny), type_scan=[], examples=[],
    limitation='Existence check, not a bound on occurrence or posterior impact.')
for kind, name in enumerate(galaxy.types):
    z = np.linspace(.01, galaxy.z_max[name]*.999, 96)
    ms, _, alpha = galaxy.luminosity_parameters(z, name)
    cap = np.minimum(galaxy.magnitude_cap(z), galaxy.p['lum_fct_m_max'])
    logmax = np.log(upper_gamma(alpha+1, 10**(.4*(ms-cap))))
    tab = galaxy.tables[name]
    bottom = tab['log_g'][0]-logmax
    use = np.flatnonzero(bottom < logtiny)
    scan = dict(type=name, z_max=float(galaxy.z_max[name]), grid_points=len(z),
        table_tail_points=len(use), passing_intrinsic_cut=0, finite_old_ratio=0)
    if len(use):
        logu = (bottom[use]+logtiny)/2
        M = ms[use]-np.interp(logmax[use]+logu, tab['log_g'], tab['inverse_y'])
        c = galaxy.sed_alpha(z[use], name)
        c /= c.sum(axis=1, keepdims=True)
        photo = galaxy.photometry(z[use], M, c)
        mean, _ = galaxy.size_parameters(z[use], photo['M_i'], name)
        data = dict(type=np.full(len(use), kind), z=z[use], M_B=M, c=c,
            R_kpc=np.exp(mean), n=np.full(len(use), 2.), a=np.full(len(use), .3),
            psi=np.full(len(use), 1.))
        lp = galaxy.log_intensity(data)
        old = numerical_over_analytic(galaxy, data)
        continuous = numerical_native_log_intensity(galaxy, data)
        finite = np.isfinite(lp) & np.isfinite(old)
        scan['passing_intrinsic_cut'] = int(np.isfinite(lp).sum())
        scan['finite_old_ratio'] = int(finite.sum())
        assert np.isneginf(continuous[finite]).all()
        if finite.any():
            j = np.flatnonzero(finite)[0]
            result['examples'].append(dict(type=name, native={key: value[j].tolist()
                for key, value in data.items()}, m_r=float(photo['m_r'][j]),
                m_i=float(photo['m_i'][j]), log_analytic_intensity=float(lp[j]),
                old_log_q_over_p=float(old[j]), log_u=float(logu[j]),
                log_u_table_min=float(bottom[use[j]]), continuous_density_zero=True))
    result['type_scan'].append(scan)
result['production_support_leak_exists'] = bool(result['examples'])
result['source_sha256'] = {name: hashlib.sha256((args.source/name).read_bytes()).hexdigest()
    for name in ('sbsi/galsbi_prior.py', 'sbsi/scene_exchange.py',
                 'sbsi/numerical_native_intensity.py', 'scripts/check_galsbi_continuous_prior.py')}
result['external_config'] = dict(path=config.__file__,
    sha256=hashlib.sha256(Path(config.__file__).read_bytes()).hexdigest())
args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result, indent=2, allow_nan=False))
