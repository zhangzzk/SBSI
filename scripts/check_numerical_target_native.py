"""CPU-only installed-prior check of the continuous numerical target routing.

Fixed fresh native draws, no likelihood evaluations, fitting or posterior claim.
The frozen earlier declaration is used solely as an external-input manifest;
no posterior outputs or acceptance outcomes are read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch  # Load before PyCosmo, matching the supported cluster environment.
from scipy.special import logsumexp

from check_galsbi_continuous_prior import build_prior
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.numerical_native_intensity import numerical_native_log_intensity
from sbsi.numerical_prior_inputs import build_bound_numerical_prior

DECLARATION_SHA256 = 'ad718136d4ac00a37ffcaddf09dcfad92b15e68e4ebbf82ead597ed1b7c1f758'
SEED = 2026104200
PARENT_DRAWS = 4096
ROUNDTRIP_ATOL = 2e-8


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def run(declaration, output):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise ValueError('CPU-only scheduler allocation required')
    if digest(declaration) != DECLARATION_SHA256:
        raise ValueError('frozen external-input declaration differs')
    external = json.loads(Path(declaration).read_text())['external_files']
    def verify_inputs():
        for path, expected in external.items():
            if digest(path) != expected:
                raise ValueError('changed external input: '+path)
    verify_inputs()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(status='failed', schema='sbsi-installed-numerical-target-check-v1',
        seed=SEED, parent_draws=PARENT_DRAWS, declaration=str(Path(declaration).resolve()),
        declaration_sha256=DECLARATION_SHA256, external_files=external,
        job=os.environ['SLURM_JOB_ID'], python=sys.version.split()[0], numpy=np.__version__,
        torch=str(torch.__version__), script_sha256=digest(__file__),
        roundtrip_log_density_atol=ROUNDTRIP_ATOL,
        scope='Installed-law routing and coordinate roundtrip only; no likelihood, posterior, convergence or efficiency validation.')
    try:
        galaxy, cosmo, _, inputs, input_paths = build_bound_numerical_prior(build_prior)
        receipt.update(numerical_prior_inputs=inputs, numerical_prior_input_paths=input_paths)
        sample, metadata = galaxy.sample(PARENT_DRAWS, SEED)
        if not len(sample['z']):
            raise ValueError('native generator retained no rows; no replacement draws permitted')
        coordinates = GalSBITargetCoordinates(galaxy,
            lambda z: cosmo.background.dist_ang_a(a=1/(1+z)), target_prior='numerical')
        x = coordinates.encode(sample)
        lp, decoded = coordinates.log_prior(x, sample['type'])
        expected_native = numerical_native_log_intensity(galaxy, sample)
        decoded_native = numerical_native_log_intensity(galaxy, decoded)
        _, logj = coordinates.decode(x, sample['type'])
        if not np.isfinite(expected_native).all() or not np.isfinite(lp).all():
            raise ValueError('generated native draws must be in the declared continuous support')
        np.testing.assert_array_equal(lp, decoded_native+logj)
        np.testing.assert_allclose(decoded_native, expected_native, atol=ROUNDTRIP_ATOL, rtol=0)
        mixture, placeholder, probabilities = coordinates.log_mixture_prior(x)
        terms = []
        for kind in range(len(galaxy.types)):
            data = dict(decoded, type=np.full(len(x), kind, dtype=int))
            terms.append(numerical_native_log_intensity(galaxy, data))
        terms = np.column_stack(terms)
        total = logsumexp(terms, axis=1)
        np.testing.assert_array_equal(mixture, total+logj)
        np.testing.assert_array_equal(probabilities, np.exp(terms-total[:, None]))
        np.testing.assert_allclose(probabilities.sum(1), 1., atol=2e-14, rtol=0)
        if not np.isfinite(mixture).all() or not np.isfinite(probabilities).all():
            raise ValueError('nonfinite generated-row numerical mixture')
        np.testing.assert_array_equal(placeholder['type'], np.zeros(len(x), dtype=int))
        payload = dict(sample, coordinates=x, numerical_log_prior=lp,
            numerical_log_mixture=mixture, conditional_type_probabilities=probabilities)
        np.savez_compressed(output/'native_draws.npz', **payload)
        verify_inputs()
        receipt.update(status='passed', retained_draws=len(x), sample_metadata=metadata,
            native_roundtrip_max_absolute_log_error=float(np.max(np.abs(decoded_native-expected_native))),
            exact_decoded_density_and_type_sum=True, external_inputs_unchanged=True,
            draws_sha256=digest(output/'native_draws.npz'))
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic()-started
        with (output/'result.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k:receipt[k] for k in ('status','retained_draws','native_roundtrip_max_absolute_log_error','elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.declaration, args.output)
