"""Optional named references to external SBSI model artifacts.

The SBSI workflow is model-name agnostic.  V3 and V3b are convenience path
presets only; no training, catalogue selection, response logic, or inference
behavior branches on these names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .paths import RELEASE_MODELS_ROOT


SHAPE_SEEDS: Tuple[int, ...] = (501, 502, 503, *range(505, 518))


@dataclass(frozen=True)
class ModelPaths:
    """Paths for one trained flow ensemble and its companion emulator."""

    flow_checkpoints: Tuple[Path, ...]
    emulator_model: Optional[Path] = None
    emulator_metadata: Optional[Path] = None
    name: Optional[str] = None
    emulator_sha256: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(
            self, "flow_checkpoints", tuple(Path(path) for path in self.flow_checkpoints)
        )
        if self.emulator_model is not None:
            object.__setattr__(self, "emulator_model", Path(self.emulator_model))
        if self.emulator_metadata is not None:
            object.__setattr__(self, "emulator_metadata", Path(self.emulator_metadata))
        if not self.flow_checkpoints:
            raise ValueError("at least one flow checkpoint is required")

    def validate(self, verify_emulator_hash: bool = True) -> None:
        missing = [path for path in self.flow_checkpoints if not path.is_file()]
        if self.emulator_model is not None and not self.emulator_model.is_file():
            missing.append(self.emulator_model)
        if self.emulator_metadata is not None and not self.emulator_metadata.is_file():
            missing.append(self.emulator_metadata)
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
_FLOW_ROOT = _CACHE_ROOT / "ablation"
_EMU_ROOT = _CACHE_ROOT / "derisk"
_BLENDEMU_MODELS = _root("BLENDEMU_MODELS", RELEASE_MODELS_ROOT / "blendemu")

V3 = ModelPaths(
    name="V3",
    flow_checkpoints=tuple(
        _FLOW_ROOT / f"measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s{seed}_swaavg.pt"
        for seed in SHAPE_SEEDS
    ),
    emulator_model=(
        _EMU_ROOT / "v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json"
    ),
    emulator_metadata=_BLENDEMU_MODELS / "emulator_metadata_lsst_r_extnbr_v22.json",
    emulator_sha256="01decd1335ce1c23aac1ef6ba055ae01c3950e47c4046345dcb6a1813033c21f",
)

V3B = ModelPaths(
    name="V3b",
    flow_checkpoints=tuple(
        _FLOW_ROOT / f"measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s{seed}_swaavg.pt"
        for seed in SHAPE_SEEDS
    ),
    emulator_model=_EMU_ROOT / "v2_reweighted_vector_fixed_v1/weighted_model.json",
    emulator_metadata=(
        _BLENDEMU_MODELS / "emulator_metadata_lsst_r_extnbr_indom_tuned.json"
    ),
    emulator_sha256="3cf70b6e74ad382f3ec59c6e8a2d0a5b9b0615d4c4677c7a71dd2344cbf35553",
)

MODEL_PRESETS: Dict[str, ModelPaths] = {"V3": V3, "V3b": V3B}


def get_model(name: str) -> ModelPaths:
    """Return a case-insensitive convenience model-path preset."""

    normalized = name.strip().lower()
    for key, model in MODEL_PRESETS.items():
        if key.lower() == normalized:
            return model
    choices = ", ".join(MODEL_PRESETS)
    raise KeyError(f"unknown model preset {name!r}; choose one of: {choices}")


def validate_models(names: Iterable[str] = ("V3", "V3b")) -> None:
    for name in names:
        get_model(name).validate()


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


__all__ = [
    "MODEL_PRESETS",
    "ModelPaths",
    "SHAPE_SEEDS",
    "V3",
    "V3B",
    "get_model",
    "load_emulator",
    "validate_models",
]
