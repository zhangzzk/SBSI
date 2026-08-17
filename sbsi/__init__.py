"""SBSI: flow training, response prediction, and Bayesian inference."""

from .catalogue import Catalogue, load_catalogue
from .domain import Domain
from .forward_catalogue import (
    DEFAULT_CROWDING_RADII_ARCSEC,
    EmulatorPairingConfig,
    INPUT_CATALOGUE_COLUMNS,
    PreparedForwardCatalogue,
    find_neighbours,
    make_pair_catalogue,
    prepare_emulator_pairs,
    prepare_flow_inputs,
    prepare_forward_catalogue,
    validate_input_catalogue,
)
from .flow import FlowTrainingConfig, FlowTrial, load_flow, train_flow, tune_flow
from .inference import BayesianInference, PosteriorResult, RadialShapePrior, ShearInferenceResult
from .models import (
    MODEL_PRESETS,
    SHAPE_SEEDS,
    V3,
    V3B,
    ModelPaths,
    get_model,
    load_emulator,
    validate_models,
)
from .response import (
    MeasurementSample,
    ResponsePrediction,
    ResponsePredictor,
    predict_blend_response,
    sample_measurement,
)

__version__ = "3.0.0"

__all__ = [
    "BayesianInference",
    "Catalogue",
    "FlowTrainingConfig",
    "FlowTrial",
    "DEFAULT_CROWDING_RADII_ARCSEC",
    "EmulatorPairingConfig",
    "INPUT_CATALOGUE_COLUMNS",
    "MODEL_PRESETS",
    "MeasurementSample",
    "ModelPaths",
    "PosteriorResult",
    "PreparedForwardCatalogue",
    "RadialShapePrior",
    "ResponsePrediction",
    "ResponsePredictor",
    "ShearInferenceResult",
    "SHAPE_SEEDS",
    "Domain",
    "V3",
    "V3B",
    "get_model",
    "find_neighbours",
    "load_catalogue",
    "load_flow",
    "load_emulator",
    "make_pair_catalogue",
    "prepare_emulator_pairs",
    "prepare_flow_inputs",
    "prepare_forward_catalogue",
    "train_flow",
    "tune_flow",
    "predict_blend_response",
    "sample_measurement",
    "validate_models",
    "validate_input_catalogue",
]
