"""Backward-compatible aliases for the first SBS selection model.

Historically this module was named for detection.  The scientific model is now
written as P(s=1 | x, n, gamma), where the current pilot uses SExtractor
detection as the first concrete selection event.
"""

from .selection_model import (  # noqa: F401
    DEFAULT_SELECTION_FEATURES as DEFAULT_FEATURES,
    FocalLossWithLogits,
    SelectionMLP as DetectionMLP,
    SelectionModelBundle as DetectionClassifierBundle,
    TabularPreprocessor,
    collect_logits_and_targets,
    fit_temperature,
    load_selection_model as load_detection_classifier,
    save_selection_model as save_detection_classifier,
)
