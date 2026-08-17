"""Standalone SBI / shear-calibration library for the SBSI pipeline.

The certified result is the parameter-free response decomposition

    m = R_sim / (R_flow + R_blend) - 1

where only ``R_flow`` (the isolated-galaxy shape response) is a trained model.

SBSI consumes *finished* catalogues.  You run blendemu yourself to produce the
simulations and measurements, then point SBSI at the output; see
:mod:`sbs_shear.paths` for the environment variables that do the pointing.  The one
piece of blendemu that SBSI calls directly is the blending-response emulator used for
``R_blend``, and that is isolated behind :mod:`sbs_shear.emulator`.

Public API
----------
Configuration:
    paths                       filesystem roots, all environment-overridable
    emulator                    the single blendemu boundary (R_blend only)

Model loaders:
    load_measurement_model      the R_flow measurement flow (ConditionalMeanFlow)
    load_selection_model        the differentiable selection classifier
    load_scene_measurement_model

Response evaluation (``sbs_shear.response``):
    model_mean_proj             induced flow first moment projected on the shear direction
    flow_response               the antithetic +/-g secant R_flow = (m_+g - m_-g)/(2g)
    load_sheared_sample         stream + select + reservoir-sample a sheared catalogue
"""

from . import emulator, paths
from .measurement_model import load_measurement_model
from .response import flow_response, load_sheared_sample, model_mean_proj
from .scene_model import load_scene_measurement_model
from .selection_model import load_selection_model

__all__ = [
    "paths",
    "emulator",
    "load_measurement_model",
    "load_scene_measurement_model",
    "load_selection_model",
    "model_mean_proj",
    "flow_response",
    "load_sheared_sample",
]
