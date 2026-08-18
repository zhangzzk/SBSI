"""Optional named references to external SBSI model artifacts.

The SBSI workflow is model-name agnostic.  V3 and V3b are convenience path
presets only; no training, catalogue selection, response logic, or inference
behavior branches on these names.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


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


def _root(variable: str, default: str) -> Path:
    """Location of a preset's artifacts, overridable so presets work off this checkout.

    The defaults are the release tree shipped in this repository (models/), so an unset
    environment resolves get_model("V3") inside any clone; the emulator SHA-256 in each
    preset is what actually pins identity. A user who keeps the caches or the BlendEMU
    artifacts elsewhere sets the variable instead of editing this file.
    """

    return Path(os.environ.get(variable) or default).expanduser()


_RELEASE_ROOT = Path(__file__).resolve().parents[1] / "models"
_CACHE_ROOT = _root("SBSI_CACHE_DIR", str(_RELEASE_ROOT))
_FLOW_ROOT = _CACHE_ROOT / "ablation"
_EMU_ROOT = _CACHE_ROOT / "derisk"
# BLENDEMU_ROOT is deliberately not consulted for artifact paths: it names the BlendEMU
# code checkout, and deriving artifact paths from it made a code checkout silently shadow
# the release tree. load_emulator does read it as an import fallback (_blendemu_import_root).
_BLENDEMU_MODELS = _root("BLENDEMU_MODELS", str(_RELEASE_ROOT / "blendemu"))

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


def _blendemu_configured_root() -> str:
    """Raw BlendEMU location from $BLENDEMU_ROOT or ~/.config/sbsi/blendemu_root."""

    configured = os.environ.get("BLENDEMU_ROOT")
    if not configured:
        try:
            configured = (Path.home() / ".config/sbsi/blendemu_root").read_text().strip()
        except OSError:
            configured = ""
    return configured or ""


def _blendemu_import_root() -> Optional[Path]:
    """Directory whose insertion into sys.path makes `import blendemu` succeed.

    The location comes from $BLENDEMU_ROOT or, failing that, a one-line
    ~/.config/sbsi/blendemu_root file — the file survives JupyterHub-style launches
    that skip the shell exports.
    """

    configured = _blendemu_configured_root()
    if not configured:
        return None
    root = Path(configured).expanduser()
    if (root / "blendemu").is_dir():
        return root
    if root.name == "blendemu" and root.is_dir():
        return root.parent
    return None


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
            "Presets resolve through SBSI_CACHE_DIR and BLENDEMU_MODELS; unset, they "
            "default to the repository's models/ release tree (see models/README.md)."
        )
    try:
        from blendemu import BlendingPredictor
    except ModuleNotFoundError as error:
        if error.name is not None and not error.name.startswith("blendemu"):
            raise  # a dependency inside BlendEMU is missing; a search path cannot fix that
        root = _blendemu_import_root()
        if root is None:
            hint = (
                "export BLENDEMU_ROOT=/path/to/blendemu, or write that path once to"
                f" {Path.home() / '.config/sbsi/blendemu_root'}"
            )
            configured = _blendemu_configured_root()
            if configured:
                hint += (
                    f" (configured {configured!r} does not contain"
                    " a blendemu package)"
                )
            raise ModuleNotFoundError(
                "BlendEMU is required to load the emulator: install it, put its checkout"
                f" on PYTHONPATH, or {hint} — SBSI imports it from there"
            ) from error
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from blendemu import BlendingPredictor

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
