"""The one place SBSI talks to blendemu.

SBSI is deliberately decoupled from blendemu: you run blendemu yourself to produce the
simulations and measurement catalogues, and hand SBSI the paths (see
:mod:`sbs_shear.paths`).  For that flow blendemu need not be installed at all.

The single exception is the *blending-response emulator*.  R_blend -- the neighbour term
of the response decomposition m = R_sim / (R_flow + R_blend) - 1 -- is evaluated by
blendemu's trained ``BlendingPredictor``, so scripts that build R_blend lookups do need
blendemu importable.  Rather than scattering ``sys.path`` edits and hardcoded model
directories across those scripts, they all come through here.

Usage
-----
    from sbs_shear.emulator import load_blending_predictor
    pred = load_blending_predictor(tag="lsst_r_extnbr_ho")
    responses = pred.predict_response(primary_catalogue, secondary_catalogue)

Make blendemu importable either by installing it, or by exporting ``BLENDEMU_ROOT``
(this module will put it on ``sys.path`` for you).
"""
from __future__ import annotations

import sys

from . import paths

__all__ = [
    "SURVEY_CONDITIONS", "RESCALE_KW", "DEFAULT_EMULATOR_TAG",
    "import_blendemu", "load_blending_predictor",
]

# Observing conditions of the LSST r-band sim set the emulator was trained against.
# Previously copy-pasted as a `COND` literal into eight separate scripts.
SURVEY_CONDITIONS = dict(
    pixel_size=0.2,      # arcsec / pixel
    zero_point=30.0,     # magnitude zero point
    psf_fwhm=0.73,       # arcsec
    moffat_beta=2.224,
    pixel_rms=0.312,     # per-pixel noise
)

# The same constants in the keyword spelling `sbs_shear.preprocessing.rescale_features`
# expects (`zero_mag` rather than `zero_point`, no separate ordering).
RESCALE_KW = dict(
    pixel_rms=SURVEY_CONDITIONS["pixel_rms"],
    pixel_size=SURVEY_CONDITIONS["pixel_size"],
    zero_mag=SURVEY_CONDITIONS["zero_point"],
    psf_fwhm=SURVEY_CONDITIONS["psf_fwhm"],
    moffat_beta=SURVEY_CONDITIONS["moffat_beta"],
)

# Extended-neighbour, held-out emulator. The older `lsst_r` tag discarded bright
# (r<18) and extreme-size neighbours, which left an uncorrected bright-neighbour
# tail in m; see Gold-V1.md.
DEFAULT_EMULATOR_TAG = "lsst_r_extnbr_ho"

_MISSING = """\
blendemu is required for the blending-response emulator but could not be imported.

SBSI only needs it for R_blend; the rest of the pipeline runs on finished catalogues
without it. To enable it, either install blendemu into this environment, or point SBSI
at a checkout:

    export BLENDEMU_ROOT=/path/to/blendemu

(currently BLENDEMU_ROOT={root}, which {state})
"""


def import_blendemu():
    """Import and return the ``blendemu`` package, adding ``BLENDEMU_ROOT`` to the path.

    Raises ``ImportError`` with actionable instructions if it cannot be found.
    """
    root = str(paths.BLENDEMU_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import blendemu
    except ImportError as exc:
        state = "exists" if paths.BLENDEMU_ROOT.exists() else "does not exist"
        raise ImportError(_MISSING.format(root=root, state=state)) from exc
    return blendemu


def load_blending_predictor(tag: str | None = None, models_dir=None,
                            conditions: dict | None = None, device: str = "cpu"):
    """Load blendemu's trained ``BlendingPredictor``.

    Parameters
    ----------
    tag :
        Emulator tag; defaults to :data:`DEFAULT_EMULATOR_TAG`.
    models_dir :
        Directory of trained emulator weights; defaults to ``paths.BLENDEMU_MODELS``
        (override with the ``BLENDEMU_MODELS`` environment variable).
    conditions :
        Observing conditions; defaults to :data:`SURVEY_CONDITIONS`.
    """
    import_blendemu()
    from blendemu.inference import BlendingPredictor

    models_dir = paths.require(
        paths.BLENDEMU_MODELS if models_dir is None else models_dir, "emulator model directory")
    return BlendingPredictor.load(
        str(models_dir),
        tag=DEFAULT_EMULATOR_TAG if tag is None else tag,
        conditions=dict(SURVEY_CONDITIONS if conditions is None else conditions),
        device=device,
    )
