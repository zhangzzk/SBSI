"""Training implementation behind the SBSI measurement-flow CLI.

The detailed response-aware objective lives in :mod:`sbs_shear.flow_training`.
This module exposes the typed configuration and reusable operations consumed by
``sbsi flow``. All catalogue and calibration paths are supplied by the user;
model release names never alter the training behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from .catalogue import Catalogue
from .flow_training import main as _run_training
from .measurement_model import MeasurementModelBundle, load_measurement_model


@dataclass(frozen=True)
class FlowTrainingConfig:
    """Configuration for one response-aware flow training run."""

    catalogue: Path
    output: Path
    response_target: Path
    coupling_target: Path
    seed: int = 501
    primary_magnitude_max: float = 25.8
    primary_half_light_radius_min: float = 0.5
    max_rows: int = 4_000_000
    epochs: int = 80
    batch_size: int = 8192
    response_weight: float = 450.0
    coupling_weight: float = 500.0
    learning_rate: float = 7.0e-4
    patience: int = 10
    swa_last_k: int = 8
    num_workers: int = 8
    device: Optional[str] = None
    gpu_resident: bool = True
    extra_arguments: Tuple[str, ...] = ()

    @property
    def swa_output(self) -> Path:
        """Checkpoint produced by averaging the final ``swa_last_k`` epochs."""

        output = Path(self.output)
        if "swabase" in output.name:
            return output.with_name(output.name.replace("swabase", "swaavg"))
        return output.with_name(f"{output.stem}_swaavg{output.suffix}")

    @property
    def training_curve(self) -> Path:
        output = Path(self.output)
        return output.with_name(f"{output.stem}_train_curve.npz")

    def validate(self) -> None:
        for path in (self.catalogue, self.response_target, self.coupling_target):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        existing = [
            path for path in (Path(self.output), self.swa_output, self.training_curve)
            if path.exists()
        ]
        if existing:
            rendered = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"refusing to overwrite training artifacts: {rendered}")
        if self.seed <= 0:
            raise ValueError("seed must be positive")
        if self.max_rows < 0:
            raise ValueError("max_rows must be >= 0 (zero means all rows)")

    def to_argv(self) -> List[str]:
        args = [
            "--catalogue", str(self.catalogue),
            "--output", str(self.output),
            "--target-column", "detected",
            "--selection-name", "sextractor_detected",
            "--feature-set", "g0_meas_crowd_conc_szfl_noz",
            "--target-features", "measured_ngmix_g1", "measured_ngmix_g2",
            "measured_mag_auto", "measured_log_flux_radius",
            "--flow-type", "mean_affine",
            "--mean-hidden", "128",
            "--flow-blind-features", "e1_input_p", "e2_input_p",
            "--shear-case", "0.0",
            "--max-rows", str(self.max_rows),
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--hidden-dim", "256",
            "--condition-layers", "3",
            "--n-flows", "10",
            "--lr", str(self.learning_rate),
            "--patience", str(self.patience),
            "--weight-decay", "1e-5",
            "--swa-last-k", str(self.swa_last_k),
            "--seed", str(self.seed),
            "--num-workers", str(self.num_workers),
            "--primary-mag-max", str(self.primary_magnitude_max),
            "--primary-re-min", str(self.primary_half_light_radius_min),
            "--response-weight", str(self.response_weight),
            "--response-delta", "0.02",
            "--response-difference", "central",
            "--response-target-npz", str(self.response_target),
            "--response-error", "absolute",
            "--response-rel-floor", "0.05",
            "--response-bin-ema", "0.0",
            "--coupling-weight", str(self.coupling_weight),
            "--coupling-target-npz", str(self.coupling_target),
        ]
        if self.device:
            args.extend(("--device", self.device))
        if self.gpu_resident:
            args.append("--gpu-resident")
        args.extend(self.extra_arguments)
        return args


@dataclass(frozen=True)
class FlowTrial:
    config: FlowTrainingConfig
    score: float


def train_flow(config: FlowTrainingConfig) -> Path:
    """Train one response-aware flow and return its SWA checkpoint path."""

    config.validate()
    _run_training(config.to_argv())
    if not config.swa_output.is_file():
        raise RuntimeError(f"training did not produce expected SWA model: {config.swa_output}")
    return config.swa_output


def tune_flow(
    base: FlowTrainingConfig,
    candidates: Iterable[dict],
    *,
    validation_catalogue: Catalogue,
    scorer: Callable[[Path, Catalogue], float],
) -> List[FlowTrial]:
    """Run an explicit, auditable flow hyperparameter sweep.

    ``candidates`` contains keyword overrides accepted by ``FlowTrainingConfig``;
    each candidate must set a unique ``output``.  The caller supplies both the
    validation catalogue and scorer so the training/validation firewall remains
    visible.  Trials are returned in ascending score order.
    """

    configs = []
    seen = set()
    for overrides in candidates:
        config = replace(base, **dict(overrides))
        output = Path(config.output).resolve()
        if output in seen:
            raise ValueError(f"duplicate tuning output: {output}")
        seen.add(output)
        configs.append(config)

    # Preflight every trial before starting the first expensive run. This avoids
    # discovering a missing input or occupied output only after earlier trials
    # have already produced artifacts.
    for config in configs:
        config.validate()

    trials = []
    for config in configs:
        checkpoint = train_flow(config)
        score = float(scorer(checkpoint, validation_catalogue))
        trials.append(FlowTrial(config=config, score=score))
    return sorted(trials, key=lambda trial: trial.score)


def load_flow(path: Path, device: str = "cpu") -> MeasurementModelBundle:
    return load_measurement_model(str(path), device=device)


__all__ = [
    "FlowTrainingConfig",
    "FlowTrial",
    "load_flow",
    "train_flow",
    "tune_flow",
]
