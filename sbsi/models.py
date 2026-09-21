"""Model identities used by the SBSI likelihood and inference pipeline.

Model names are path conveniences only. Scientific behavior is defined by an
explicit likelihood configuration and verified artifact hashes.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple

from .paths import RELEASE_MODELS_ROOT


V35_LIKE_CLASSIFIER_SEEDS: Tuple[int, ...] = (20260913, 20260914, 20260915)


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable) or default).expanduser().resolve()


@dataclass(frozen=True)
class ModelPaths:
    """Paths and optional hashes for one flow, emulator, and classifier set."""

    flow_checkpoints: Tuple[Path, ...]
    emulator_model: Optional[Path] = None
    emulator_metadata: Optional[Path] = None
    detection_classifiers: Tuple[Path, ...] = ()
    name: Optional[str] = None
    flow_sha256s: Tuple[str, ...] = ()
    emulator_sha256: Optional[str] = None
    emulator_metadata_sha256: Optional[str] = None
    detection_classifier_sha256s: Tuple[str, ...] = ()
    response_backend: str = "blendemu"

    def __post_init__(self):
        object.__setattr__(
            self, "flow_checkpoints", tuple(Path(path) for path in self.flow_checkpoints)
        )
        object.__setattr__(
            self,
            "detection_classifiers",
            tuple(Path(path) for path in self.detection_classifiers),
        )
        if self.emulator_model is not None:
            object.__setattr__(self, "emulator_model", Path(self.emulator_model))
        if self.emulator_metadata is not None:
            object.__setattr__(self, "emulator_metadata", Path(self.emulator_metadata))
        if not self.flow_checkpoints:
            raise ValueError("at least one flow checkpoint is required")
        for paths, hashes, label in (
            (self.flow_checkpoints, self.flow_sha256s, "flow"),
            (
                self.detection_classifiers,
                self.detection_classifier_sha256s,
                "detection classifier",
            ),
        ):
            if hashes and len(hashes) != len(paths):
                raise ValueError(
                    f"{label} hashes must be empty or match the number of paths"
                )

    def validate(self, verify_hashes: bool = True) -> None:
        paths = [*self.flow_checkpoints, *self.detection_classifiers]
        if self.emulator_model is not None:
            paths.append(self.emulator_model)
        if self.emulator_metadata is not None:
            paths.append(self.emulator_metadata)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            rendered = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError(f"missing model artifacts:\n  {rendered}")
        if not verify_hashes:
            return
        expected = [
            *zip(self.flow_checkpoints, self.flow_sha256s),
            *zip(self.detection_classifiers, self.detection_classifier_sha256s),
        ]
        if self.emulator_model is not None and self.emulator_sha256 is not None:
            expected.append((self.emulator_model, self.emulator_sha256))
        if (
            self.emulator_metadata is not None
            and self.emulator_metadata_sha256 is not None
        ):
            expected.append((self.emulator_metadata, self.emulator_metadata_sha256))
        for path, expected_hash in expected:
            actual = _file_sha256(path)
            if actual != expected_hash:
                raise RuntimeError(
                    f"model hash mismatch for {path}: expected {expected_hash}, found {actual}"
                )


_CACHE_ROOT = _root("SBSI_CACHE_DIR", RELEASE_MODELS_ROOT)
_V35_FLOW_ROOT = (
    _CACHE_ROOT
    / "plain_complete_flow_response_re037_v1"
    / "physical_circularized_grid_free_shape_radius_lambda10_v2"
)
_V35_CLASSIFIER_ROOT = (
    _CACHE_ROOT
    / "minimal_coherent_u_flow_rblend_ablation_v2"
    / "models"
    / "flow8_rblend"
)
_V35_EMULATOR_ROOT = _V35_FLOW_ROOT / "constgold50_trial9_v1"

V35_LIKE = ModelPaths(
    name="V3.5-like",
    flow_checkpoints=(_V35_FLOW_ROOT / "paired" / "epoch154.pt",),
    flow_sha256s=(
        "9c5bb028437a4714b454d2bcd202243a2eda6263c36b4c70c46de3d12f725a6f",
    ),
    emulator_model=_V35_EMULATOR_ROOT / "trial9_base.json",
    emulator_metadata=_V35_EMULATOR_ROOT / "metadata.json",
    emulator_sha256="9723589880234fa6825ae242d9b978120a6385faf5ccc9942d1993696952e46d",
    emulator_metadata_sha256=(
        "48908a9cebc7475be83eef767c95175694531d78b1dd789c512fc49c0699f2b4"
    ),
    detection_classifiers=tuple(
        _V35_CLASSIFIER_ROOT / f"seed{seed}" / "selected.pt"
        for seed in V35_LIKE_CLASSIFIER_SEEDS
    ),
    detection_classifier_sha256s=(
        "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
        "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
        "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
    ),
)

_V36_SPEC = json.loads(
    (RELEASE_MODELS_ROOT.parent / "configs" / "models_v3_6_like.json").read_text()
)
_V36_RUNS_ROOT = _root("BLENDEMU_RUNS_DIR", RELEASE_MODELS_ROOT)
V36_LIKE = ModelPaths(
    name="V3.6-like",
    flow_checkpoints=(_CACHE_ROOT / _V36_SPEC["flow"]["path"],),
    flow_sha256s=(_V36_SPEC["flow"]["sha256"],),
    emulator_model=_V36_RUNS_ROOT / _V36_SPEC["emulator"]["path"],
    emulator_metadata=_V36_RUNS_ROOT / _V36_SPEC["emulator"]["metadata_path"],
    emulator_sha256=_V36_SPEC["emulator"]["sha256"],
    emulator_metadata_sha256=_V36_SPEC["emulator"]["metadata_sha256"],
    detection_classifiers=(_CACHE_ROOT / _V36_SPEC["classifier"]["path"],),
    detection_classifier_sha256s=(_V36_SPEC["classifier"]["sha256"],),
    response_backend="disk_response_moment",
)

MODEL_PRESETS: dict[str, ModelPaths] = {"V3.5-like": V35_LIKE, "V3.6-like": V36_LIKE}


def get_model(name: str) -> ModelPaths:
    """Return a case-insensitive model-path preset."""

    normalized = name.strip().lower()
    for key, model in MODEL_PRESETS.items():
        if key.lower() == normalized:
            return model
    choices = ", ".join(MODEL_PRESETS)
    raise KeyError(f"unknown model preset {name!r}; choose one of: {choices}")


def load_emulator(models: ModelPaths, *, conditions: dict, device: str = "cpu"):
    """Load the companion BlendEMU response model from explicit paths."""

    if models.response_backend != "blendemu":
        raise ValueError(
            "disk-response models require load_disk_response and disk transport; "
            "they cannot be used as a fixed additive R_blend"
        )
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
            "Set SBSI_CACHE_DIR to the external model store when using a preset."
        )
    try:
        from blendemu import BlendingPredictor
    except ImportError as error:
        if isinstance(error, ModuleNotFoundError) and error.name not in {None, "blendemu"}:
            raise
        raise ImportError(
            "BlendEMU is unavailable. Install its checkout in the active environment."
        ) from error

    return BlendingPredictor.load(
        model_dir=str(models.emulator_metadata.parent),
        metadata_file=models.emulator_metadata.name,
        reg_file=str(models.emulator_model.resolve()),
        conditions=dict(conditions),
        device=device,
        load_self=False,
    )


def load_disk_response(models: ModelPaths, *, device: str = "cpu", nthread: int = 1):
    """Load the V3.6-like output-conditioned disk velocity predictor.

    Use its per-draw predictions with ``sbsi.disk_response_transport``; this
    object does not implement the additive BlendEMU predictor interface.
    """
    if models.response_backend != "disk_response_moment":
        raise ValueError("a disk_response_moment model preset is required")
    for path, expected in (
        (models.emulator_model, models.emulator_sha256),
        (models.emulator_metadata, models.emulator_metadata_sha256),
    ):
        if path is None or expected is None:
            raise ValueError("pinned disk-response model and metadata required")
        if _file_sha256(path) != expected:
            raise RuntimeError(f"model hash mismatch for {path}")
    if (models.emulator_model.name != "model.json"
            or models.emulator_metadata.name != "metadata.json"
            or models.emulator_model.parent != models.emulator_metadata.parent):
        raise ValueError("disk-response model.json and metadata.json must share a directory")
    from .output_conditioned_response import DiskResponseMoment

    return DiskResponseMoment(models.emulator_model.parent, device=device, nthread=nthread)


def load_detection_classifier(models: ModelPaths, *, device: str = "cpu"):
    """Load the preset's one classifier or equal-probability ensemble."""

    if not models.detection_classifiers:
        raise ValueError("detection classifier paths are required")
    missing = [path for path in models.detection_classifiers if not path.is_file()]
    if missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing detection-classifier artifacts:\n  {rendered}")
    if models.detection_classifier_sha256s:
        for path, expected in zip(
            models.detection_classifiers,
            models.detection_classifier_sha256s,
        ):
            actual = _file_sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"detection-classifier hash mismatch for {path}: "
                    f"expected {expected}, found {actual}"
                )
    from .selection_model import load_selection_model, load_selection_model_ensemble

    if len(models.detection_classifiers) == 1:
        return load_selection_model(models.detection_classifiers[0], device=device)
    return load_selection_model_ensemble(models.detection_classifiers, device=device)


__all__ = [
    "MODEL_PRESETS",
    "ModelPaths",
    "V35_LIKE",
    "V35_LIKE_CLASSIFIER_SEEDS",
    "V36_LIKE",
    "get_model",
    "load_detection_classifier",
    "load_emulator",
    "load_disk_response",
]
