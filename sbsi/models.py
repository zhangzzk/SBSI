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
from typing import Optional, Tuple

from .paths import RELEASE_MODELS_ROOT


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _model_preset(filename: str) -> ModelPaths:
    spec = json.loads((RELEASE_MODELS_ROOT.parent / "configs" / filename).read_text())
    return ModelPaths(
        name=spec["name"],
        flow_checkpoints=(RELEASE_MODELS_ROOT / spec["flow"]["path"],),
        flow_sha256s=(spec["flow"]["sha256"],),
        emulator_model=RELEASE_MODELS_ROOT / spec["emulator"]["path"],
        emulator_metadata=RELEASE_MODELS_ROOT / spec["emulator"]["metadata_path"],
        emulator_sha256=spec["emulator"]["sha256"],
        emulator_metadata_sha256=spec["emulator"]["metadata_sha256"],
        detection_classifiers=(RELEASE_MODELS_ROOT / spec["classifier"]["path"],),
        detection_classifier_sha256s=(spec["classifier"]["sha256"],),
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

    if len(models.detection_classifiers) != 1:
        raise ValueError("exactly one detection classifier path is required")
    path = models.detection_classifiers[0]
    if not path.is_file():
        raise FileNotFoundError(f"missing detection-classifier artifact: {path}")
    if models.detection_classifier_sha256s:
        actual = _file_sha256(path)
        expected = models.detection_classifier_sha256s[0]
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
