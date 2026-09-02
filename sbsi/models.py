"""Optional named references to external SBSI model artifacts.

The SBSI workflow is model-name agnostic.  V3.1, V3.2, and V3.3-like are
convenience path presets only; no training, catalogue selection, response logic,
or inference behavior branches on these names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple

from .paths import RELEASE_MODELS_ROOT


V31_SEEDS: Tuple[int, ...] = (501, 502, 503, 504)
V32_SEEDS: Tuple[int, ...] = V31_SEEDS
V33_LIKE_SEEDS: Tuple[int, ...] = (501,)


@dataclass(frozen=True)
class ModelPaths:
    """Paths for one trained flow ensemble and its companion models."""

    flow_checkpoints: Tuple[Path, ...]
    emulator_model: Optional[Path] = None
    emulator_metadata: Optional[Path] = None
    detection_classifier: Optional[Path] = None
    name: Optional[str] = None
    emulator_sha256: Optional[str] = None
    detection_classifier_sha256: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(
            self, "flow_checkpoints", tuple(Path(path) for path in self.flow_checkpoints)
        )
        if self.emulator_model is not None:
            object.__setattr__(self, "emulator_model", Path(self.emulator_model))
        if self.emulator_metadata is not None:
            object.__setattr__(self, "emulator_metadata", Path(self.emulator_metadata))
        if self.detection_classifier is not None:
            object.__setattr__(
                self, "detection_classifier", Path(self.detection_classifier)
            )
        if not self.flow_checkpoints:
            raise ValueError("at least one flow checkpoint is required")

    def validate(self, verify_emulator_hash: bool = True) -> None:
        missing = [path for path in self.flow_checkpoints if not path.is_file()]
        if self.emulator_model is not None and not self.emulator_model.is_file():
            missing.append(self.emulator_model)
        if self.emulator_metadata is not None and not self.emulator_metadata.is_file():
            missing.append(self.emulator_metadata)
        if (
            self.detection_classifier is not None
            and not self.detection_classifier.is_file()
        ):
            missing.append(self.detection_classifier)
        if missing:
            rendered = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError(f"missing model artifacts:\n  {rendered}")
        if (
            verify_emulator_hash
            and self.emulator_model is not None
            and self.emulator_sha256 is not None
        ):
            actual = _file_sha256(self.emulator_model)
            if actual != self.emulator_sha256:
                raise RuntimeError(
                    f"emulator hash mismatch: expected {self.emulator_sha256}, found {actual}"
                )
        if (
            verify_emulator_hash
            and self.detection_classifier is not None
            and self.detection_classifier_sha256 is not None
        ):
            actual = _file_sha256(self.detection_classifier)
            if actual != self.detection_classifier_sha256:
                raise RuntimeError(
                    "detection-classifier hash mismatch: expected "
                    f"{self.detection_classifier_sha256}, found {actual}"
                )


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(variable: str, default: Path) -> Path:
    """Location of a preset's artifacts, overridable for external model stores.

    Defaults are anchored to the imported SBSI checkout, not the current working
    directory.  The emulator SHA-256 pins model identity.  Users who keep artifacts
    elsewhere can override the roots without editing source files.
    """

    return Path(os.environ.get(variable) or default).expanduser().resolve()


_CACHE_ROOT = _root("SBSI_CACHE_DIR", RELEASE_MODELS_ROOT)
_EMU_ROOT = _CACHE_ROOT / "derisk"
_MIXED_SHEAR_ROOT = _CACHE_ROOT / "mixed_shear_cde"
_PHOTOMETRIC_FLOW_ROOT = _CACHE_ROOT / "mixed_shear_e_photometric"
_DETECTION_ROOT = _CACHE_ROOT / "detection_classifier_transition_lambda1_v1" / "models"
_BLENDEMU_MODELS = _root("BLENDEMU_MODELS", RELEASE_MODELS_ROOT / "blendemu")

V31 = ModelPaths(
    name="V3.1",
    flow_checkpoints=tuple(
        _MIXED_SHEAR_ROOT
        / f"measurement_flow_mixed_g0_g005_E_s{seed}_swaavg.pt"
        for seed in V31_SEEDS
    ),
    emulator_model=(
        _EMU_ROOT / "v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json"
    ),
    emulator_metadata=_BLENDEMU_MODELS / "emulator_metadata_lsst_r_extnbr_v22.json",
    emulator_sha256="01decd1335ce1c23aac1ef6ba055ae01c3950e47c4046345dcb6a1813033c21f",
)

V32 = ModelPaths(
    name="V3.2",
    flow_checkpoints=tuple(
        _MIXED_SHEAR_ROOT
        / f"measurement_flow_mixed_g0_g005_E_s{seed}_swaavg.pt"
        for seed in V32_SEEDS
    ),
    emulator_model=(
        _EMU_ROOT / "v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json"
    ),
    emulator_metadata=_BLENDEMU_MODELS / "emulator_metadata_lsst_r_extnbr_v22.json",
    detection_classifier=_DETECTION_ROOT / "transition_aware.pt",
    emulator_sha256="01decd1335ce1c23aac1ef6ba055ae01c3950e47c4046345dcb6a1813033c21f",
    detection_classifier_sha256=(
        "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455"
    ),
)

V33_LIKE = ModelPaths(
    name="V3.3-like",
    flow_checkpoints=(
        _PHOTOMETRIC_FLOW_ROOT
        / "measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt",
    ),
    emulator_model=(
        _EMU_ROOT / "v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json"
    ),
    emulator_metadata=_BLENDEMU_MODELS / "emulator_metadata_lsst_r_extnbr_v22.json",
    detection_classifier=_DETECTION_ROOT / "transition_aware.pt",
    emulator_sha256="01decd1335ce1c23aac1ef6ba055ae01c3950e47c4046345dcb6a1813033c21f",
    detection_classifier_sha256=(
        "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455"
    ),
)

MODEL_PRESETS: dict[str, ModelPaths] = {
    "V3.1": V31,
    "V3.2": V32,
    "V3.3-like": V33_LIKE,
}


def get_model(name: str) -> ModelPaths:
    """Return a case-insensitive convenience model-path preset."""

    normalized = name.strip().lower()
    for key, model in MODEL_PRESETS.items():
        if key.lower() == normalized:
            return model
    choices = ", ".join(MODEL_PRESETS)
    raise KeyError(f"unknown model preset {name!r}; choose one of: {choices}")


def load_emulator(
    models: ModelPaths,
    *,
    conditions: dict,
    device: str = "cpu",
):
    """Load the companion BlendEMU response model from explicit paths."""

    if models.emulator_model is None or models.emulator_metadata is None:
        raise ValueError("emulator_model and emulator_metadata paths are required")
    missing = [
        path
        for path in (models.emulator_metadata, models.emulator_model)
        if not path.is_file()
    ]
    if missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"missing emulator artifacts:\n  {rendered}\n"
            "Presets use the checkout's models/ directory by default. Override external "
            "stores with SBSI_CACHE_DIR and BLENDEMU_MODELS."
        )
    try:
        from blendemu import BlendingPredictor
    except ImportError as error:
        if isinstance(error, ModuleNotFoundError) and error.name not in {None, "blendemu"}:
            raise
        raise ImportError(
            "BlendEMU is not installed correctly in this Python environment. Install its "
            "checkout editable (`python -m pip install --config-settings "
            "editable_mode=compat -e /path/to/blendemu`) and restart "
            "the Python process or notebook kernel."
        ) from error

    return BlendingPredictor.load(
        model_dir=str(models.emulator_metadata.parent),
        metadata_file=models.emulator_metadata.name,
        # BlendEMU joins reg_file onto model_dir; resolve so a relative preset
        # root (SBSI_CACHE_DIR=models) is not concatenated onto model_dir.
        reg_file=str(models.emulator_model.resolve()),
        conditions=dict(conditions),
        device=device,
        load_self=False,
    )


def load_detection_classifier(models: ModelPaths, *, device: str = "cpu"):
    """Load a preset's explicit SBSI detection classifier."""

    if models.detection_classifier is None:
        raise ValueError("detection_classifier path is required")
    if not models.detection_classifier.is_file():
        raise FileNotFoundError(models.detection_classifier)
    if models.detection_classifier_sha256 is not None:
        actual = _file_sha256(models.detection_classifier)
        if actual != models.detection_classifier_sha256:
            raise RuntimeError(
                "detection-classifier hash mismatch: expected "
                f"{models.detection_classifier_sha256}, found {actual}"
            )
    from .selection_model import load_selection_model

    return load_selection_model(models.detection_classifier, device=device)


__all__ = [
    "MODEL_PRESETS",
    "ModelPaths",
    "V31_SEEDS",
    "V32_SEEDS",
    "V33_LIKE_SEEDS",
    "V31",
    "V32",
    "V33_LIKE",
    "get_model",
    "load_detection_classifier",
    "load_emulator",
]
