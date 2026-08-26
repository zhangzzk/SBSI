"""Archived JAX execution experiment for frozen mean-affine measurement flows.

This module deliberately does not make JAX a runtime dependency of SBSI.  The
conversion is one-way and inference-only: PyTorch remains the checkpoint source
of truth, while a jitted JAX closure is used to test whether XLA can execute the
catalogue likelihood hot path materially faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FrozenMLP:
    """NumPy copy of a linear MLP and its shared hidden activation."""

    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    activation: str


@dataclass(frozen=True)
class FrozenAffineCoupling:
    """Parameters required by one conditional affine coupling inverse."""

    mask: np.ndarray
    scale_limit: float
    network: FrozenMLP


@dataclass(frozen=True)
class FrozenMeanAffineFlow:
    """Framework-neutral snapshot of ``ConditionalMeanFlow``."""

    target_dim: int
    context_dim: int
    keep_indices: np.ndarray
    mean_network: FrozenMLP
    couplings: tuple[FrozenAffineCoupling, ...]


def _activation_name(module: nn.Module) -> str:
    if isinstance(module, nn.SiLU):
        return "silu"
    if isinstance(module, nn.GELU):
        return "gelu"
    if isinstance(module, nn.Tanh):
        return "tanh"
    raise TypeError(f"unsupported JAX flow activation {type(module).__name__}")


def _freeze_mlp(module: nn.Module) -> FrozenMLP:
    sequence = list(module) if isinstance(module, nn.Sequential) else [module]
    layers = []
    activations = []
    for item in sequence:
        if isinstance(item, nn.Linear):
            layers.append(
                (
                    item.weight.detach().cpu().numpy().astype(np.float32, copy=True),
                    item.bias.detach().cpu().numpy().astype(np.float32, copy=True),
                )
            )
        else:
            activations.append(_activation_name(item))
    if not layers:
        raise TypeError("JAX flow conversion requires at least one linear layer")
    if activations and len(set(activations)) != 1:
        raise TypeError("JAX flow conversion requires one shared hidden activation")
    if len(activations) not in (0, len(layers) - 1):
        raise TypeError("JAX flow conversion found an unsupported MLP layout")
    return FrozenMLP(tuple(layers), activations[0] if activations else "identity")


def freeze_mean_affine_flow(model: nn.Module) -> FrozenMeanAffineFlow:
    """Copy a Torch ``ConditionalMeanFlow`` into framework-neutral arrays."""

    required = ("target_dim", "context_dim", "keep_indices", "mean_net", "flow")
    if not all(hasattr(model, name) for name in required):
        raise TypeError("JAX conversion requires a ConditionalMeanFlow-like model")
    if not hasattr(model.flow, "layers"):
        raise TypeError("JAX conversion currently supports only an affine base flow")
    couplings = []
    for layer in model.flow.layers:
        if not all(hasattr(layer, name) for name in ("mask", "scale_limit", "net")):
            raise TypeError("JAX conversion found a non-affine coupling layer")
        couplings.append(
            FrozenAffineCoupling(
                mask=layer.mask.detach().cpu().numpy().astype(np.float32, copy=True),
                scale_limit=float(layer.scale_limit),
                network=_freeze_mlp(layer.net),
            )
        )
    return FrozenMeanAffineFlow(
        target_dim=int(model.target_dim),
        context_dim=int(model.context_dim),
        keep_indices=(
            model.keep_indices.detach().cpu().numpy().astype(np.int32, copy=True)
        ),
        mean_network=_freeze_mlp(model.mean_net),
        couplings=tuple(couplings),
    )


def build_jax_log_prob(
    frozen: FrozenMeanAffineFlow,
    *,
    compile: bool = True,
) -> Callable:
    """Build a JAX log-density function accepting arbitrary leading dimensions."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("JAX is required only for the optional benchmark") from error

    def convert_mlp(mlp: FrozenMLP):
        return tuple((jnp.asarray(weight), jnp.asarray(bias)) for weight, bias in mlp.layers)

    mean_layers = convert_mlp(frozen.mean_network)
    coupling_parameters = tuple(
        (
            jnp.asarray(layer.mask),
            float(layer.scale_limit),
            convert_mlp(layer.network),
            layer.network.activation,
        )
        for layer in frozen.couplings
    )
    keep_indices = jnp.asarray(frozen.keep_indices)

    def activate(values, name):
        if name == "identity":
            return values
        if name == "silu":
            return jax.nn.silu(values)
        if name == "gelu":
            return jax.nn.gelu(values)
        if name == "tanh":
            return jnp.tanh(values)
        raise ValueError(f"unknown frozen activation {name!r}")

    def mlp(values, layers, activation):
        for index, (weight, bias) in enumerate(layers):
            # Catalogue scores and Hessians are differences of nearby log
            # densities.  JAX may otherwise use TF32 on Ampere GPUs, whose
            # error is amplified severely in the flow tails.  Match Torch's
            # default full-fp32 matmul contract before judging speed.
            values = jnp.matmul(
                values,
                weight.T,
                precision=jax.lax.Precision.HIGHEST,
            ) + bias
            if index + 1 < len(layers):
                values = activate(values, activation)
        return values

    def log_prob(target, context):
        mean = mlp(context, mean_layers, frozen.mean_network.activation)
        z = target - mean
        flow_context = jnp.take(context, keep_indices, axis=-1)
        log_determinant = jnp.zeros(z.shape[:-1], dtype=z.dtype)
        for mask, scale_limit, layers, activation in reversed(coupling_parameters):
            masked = z * mask
            parameters = mlp(
                jnp.concatenate((masked, flow_context), axis=-1),
                layers,
                activation,
            )
            shift, raw_log_scale = jnp.split(parameters, 2, axis=-1)
            transform = 1.0 - mask
            shift = shift * transform
            log_scale = jnp.tanh(raw_log_scale) * scale_limit * transform
            z = masked + transform * ((z - shift) * jnp.exp(-log_scale))
            log_determinant = log_determinant - jnp.sum(log_scale, axis=-1)
        log_base = -0.5 * jnp.sum(
            jnp.square(z) + np.float32(np.log(2.0 * np.pi)), axis=-1
        )
        return log_base + log_determinant

    return jax.jit(log_prob) if compile else log_prob


def build_jax_log_prob_from_torch(model: nn.Module, *, compile: bool = True) -> Callable:
    """Freeze a Torch mean-affine flow and return its JAX log-density closure."""

    return build_jax_log_prob(freeze_mean_affine_flow(model), compile=compile)


__all__ = [
    "FrozenAffineCoupling",
    "FrozenMLP",
    "FrozenMeanAffineFlow",
    "build_jax_log_prob",
    "build_jax_log_prob_from_torch",
    "freeze_mean_affine_flow",
]
