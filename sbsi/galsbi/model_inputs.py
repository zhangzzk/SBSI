"""Magnitude and Sersic index a model was trained on, from native GalSBI prior marks.

FS2-trained presets (V3.x) read the prior's HSC-r ``m_r`` and native ``n``; that
is the historical behaviour and stays the default. Models trained on the GalSBI
image re-simulation declare ``scene_inputs`` in their manifest: those images were
rendered with the LSST-i magnitude from the same SED coefficients and with the
Sersic index clipped to GalSim's range and rounded to a log grid
(SBSI 1c16564 scripts/write_galsbi_scenes.py). The sampler must give flow,
classifier, crowding and emulator exactly those rendered values.
"""
import hashlib
from pathlib import Path

import numpy as np

NATIVE = 'native_hsc_r'


def render_sersic_n(n, clip=(0.3, 6.2), size=122):
    """Nearest node in log n of the renderer's grid (identical arithmetic to the scene writer)."""
    grid = np.geomspace(*clip, size)
    k = np.rint(np.log(np.clip(n, *clip)/clip[0])/np.log(grid[1]/grid[0])).astype(int)
    return grid[k]


class SceneMarks:
    """Callable data -> (magnitude, sersic_n) for one model's scene-input contract.

    For a GalSBI-trained model, construction hooks the model band into the
    prior's photometry (build_prior), so it must precede any scene draw.
    """

    def __init__(self, spec=None, galaxy_prior=None, *, verify_table=True):
        self.spec = spec
        if spec is None:
            self.kind = NATIVE
            return
        if spec.get('magnitude') != 'LSST_i' or spec.get('sersic_n') != 'rendered_log_grid':
            raise ValueError(f'unsupported scene-input contract: {spec}')
        if galaxy_prior is None or not hasattr(galaxy_prior, 'model_band'):
            raise ValueError('LSST-i magnitudes need a prior whose photometry has the model_band hook (build_prior)')
        if galaxy_prior.model_band is not None:
            raise ValueError('prior photometry already carries a model magnitude')
        table = Path(spec['magnitude_table'])
        if verify_table:
            digest = hashlib.sha256(table.read_bytes()).hexdigest()
            if digest != spec['magnitude_table_sha256']:
                raise ValueError(f'magnitude table hash mismatch: {table}')
        from galsbi.ucat.magnitude_calculator import MagCalculatorTable
        # Same filter list as the scene writer, so the interpolation tables are identical.
        self.table = MagCalculatorTable(['LSST_i', 'HSC_i2'], str(table))
        self.clip, self.size = tuple(spec['sersic_n_clip']), int(spec['sersic_n_grid_size'])
        self.kind = 'galsbi_resim_lsst_i'
        # From now on every galaxy the prior draws or decodes carries 'm_model', computed once from the
        # photometry's own apparent coefficients (scene neighbours are re-mapped at every sampler step).
        galaxy_prior.model_band = self.band

    def band(self, z, coeffs):
        """LSST-i apparent magnitude from the apparent SED coefficients (as the scene writer)."""
        if not len(z):
            return np.zeros(0)
        return self.table(redshifts=z, excess_b_v=np.zeros(len(z)), coeffs=coeffs, filter_names=['LSST_i'])['LSST_i']

    def magnitude(self, data):
        if self.kind == NATIVE:
            return np.asarray(data['m_r'])
        if 'm_model' not in data:
            raise KeyError('galaxies lack m_model: draw or decode them after creating SceneMarks')
        return np.asarray(data['m_model'])

    def sersic_n(self, data):
        if self.kind == NATIVE:
            return np.asarray(data['n'])
        return render_sersic_n(np.asarray(data['n'], float), self.clip, self.size)

    def describe(self):
        return dict(kind=self.kind, spec=self.spec)
