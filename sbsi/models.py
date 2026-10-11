"""Named model identities: artifact paths plus verified hashes.

Model names are path conveniences only. The one preset, V3.9-like, is bundled
under ``models/v3_9_like/``; its hashes and provenance are pinned in
``configs/models_v3_9_like.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional

from .paths import RELEASE_MODELS_ROOT, repository_path


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelPaths:
    """Paths and optional hashes for one flow, emulator, and classifier."""

    flow_checkpoint: Path
    emulator_model: Optional[Path] = None
    emulator_metadata: Optional[Path] = None
    detection_classifier: Optional[Path] = None
    name: Optional[str] = None
    flow_sha256: Optional[str] = None
    emulator_sha256: Optional[str] = None
    emulator_metadata_sha256: Optional[str] = None
    detection_classifier_sha256: Optional[str] = None

    def __post_init__(self):
        for field in ("flow_checkpoint", "emulator_model", "emulator_metadata", "detection_classifier"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, Path(value))
        if self.flow_checkpoint is None:
            raise ValueError("a flow checkpoint is required")

    def _pinned(self):
        return (
            (self.flow_checkpoint, self.flow_sha256),
            (self.detection_classifier, self.detection_classifier_sha256),
            (self.emulator_model, self.emulator_sha256),
            (self.emulator_metadata, self.emulator_metadata_sha256),
        )

    def validate(self, verify_hashes: bool = True) -> None:
        missing = [path for path, _ in self._pinned() if path is not None and not path.is_file()]
        if missing:
            rendered = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError(f"missing model artifacts:\n  {rendered}")
        if not verify_hashes:
            return
        for path, expected_hash in self._pinned():
            if path is None or expected_hash is None:
                continue
            actual = _file_sha256(path)
            if actual != expected_hash:
                raise RuntimeError(
                    f"model hash mismatch for {path}: expected {expected_hash}, found {actual}"
                )


def _model_preset(filename: str) -> ModelPaths:
    spec = json.loads(repository_path("configs", filename).read_text())
    return ModelPaths(
        name=spec["name"],
        flow_checkpoint=RELEASE_MODELS_ROOT / spec["flow"]["path"],
        flow_sha256=spec["flow"]["sha256"],
        emulator_model=RELEASE_MODELS_ROOT / spec["emulator"]["path"],
        emulator_metadata=RELEASE_MODELS_ROOT / spec["emulator"]["metadata_path"],
        emulator_sha256=spec["emulator"]["sha256"],
        emulator_metadata_sha256=spec["emulator"]["metadata_sha256"],
        detection_classifier=RELEASE_MODELS_ROOT / spec["classifier"]["path"],
        detection_classifier_sha256=spec["classifier"]["sha256"],
    )


V39_LIKE = _model_preset("models_v3_9_like.json")
MODEL_PRESETS: dict[str, ModelPaths] = {V39_LIKE.name: V39_LIKE}


def get_model(name: str) -> ModelPaths:
    """Return a case-insensitive model-path preset."""

    normalized = name.strip().lower()
    for key, model in MODEL_PRESETS.items():
        if key.lower() == normalized:
            return model
    choices = ", ".join(MODEL_PRESETS)
    raise KeyError(f"unknown model preset {name!r}; choose one of: {choices}")


def load_emulator(models: ModelPaths, *, conditions: dict, device: str = "cpu", nthread: Optional[int] = None):
    """Load the blending model: flow, usable classifier and per-draw disk emulator.

    The blending response acts on each simulated measurement, so the returned
    :class:`~sbsi.disk_predictor.DiskResponsePredictor` carries the flow too.
    """

    from .disk_predictor import DiskResponsePredictor

    return DiskResponsePredictor(models, device=device, conditions=conditions, nthread=nthread)


def load_disk_response(models: ModelPaths, *, device: str = "cpu", nthread: int = 1):
    """Load the pinned output-conditioned disk velocity predictor.

    Use its per-draw predictions with ``sbsi.disk_response_transport``.
    """
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
    """Load the preset's detection classifier."""

    path = models.detection_classifier
    if path is None:
        raise ValueError("a detection classifier path is required")
    if not path.is_file():
        raise FileNotFoundError(f"missing detection-classifier artifact: {path}")
    expected = models.detection_classifier_sha256
    if expected is not None:
        actual = _file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"detection-classifier hash mismatch for {path}: "
                f"expected {expected}, found {actual}"
            )
    from .selection_model import load_selection_model

    return load_selection_model(path, device=device)


__all__ = [
    "MODEL_PRESETS",
    "ModelPaths",
    "V39_LIKE",
    "get_model",
    "load_detection_classifier",
    "load_emulator",
    "load_disk_response",
]
