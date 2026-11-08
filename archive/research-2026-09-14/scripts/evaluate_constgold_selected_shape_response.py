#!/usr/bin/env python3
"""Use the frozen response evaluator with p(e1,e2|S,x) flow checkpoints.

The base evaluator remains byte-identical to its frozen acceptance protocol.
This small adapter makes the exact factorization explicit: the selected-event
head already contains the magnitude/radius cuts, so the two-output conditional
shape flow is averaged without applying those cuts a second time.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import evaluate_constgold_complete_forward_response as base


SHAPE_NAMES = ("measured_ngmix_g1", "measured_ngmix_g2")
PHYSICAL_NAMES = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_flux_radius",
    "measured_flux_from_mag_auto",
)
_BASE_PREDICTION_SUFFICIENT = base._prediction_sufficient


class _SelectedShapeBundleAdapter:
    """Present selected-shape draws to the unchanged verified reducer path."""

    def __init__(self, bundle):
        self._bundle = bundle
        self.model = type("ConditionalPhysicalDiskFlow", (), {})()
        self.target_transform = SimpleNamespace(target_names=PHYSICAL_NAMES)

    def sample(self, *args, **kwargs):
        shape = np.asarray(self._bundle.sample(*args, **kwargs), dtype=np.float64)
        if shape.ndim != 3 or shape.shape[-1] != 2 or not np.isfinite(shape).all():
            raise ValueError("selected-shape flow returned invalid two-component draws")
        # Positive sentinels satisfy the physical adapter contract. Photometry
        # cuts are disabled below, so these values have no statistical role.
        sentinels = np.ones(shape.shape[:-1] + (2,), dtype=np.float64)
        return np.concatenate((shape, sentinels), axis=-1)


def _prediction_sufficient(bundle, frame, **kwargs):
    names = tuple(bundle.target_transform.target_names)
    if kwargs.get("probability_factorization", "usable_then_cut") != "direct_selected" or names != SHAPE_NAMES:
        return _BASE_PREDICTION_SUFFICIENT(bundle, frame, **kwargs)
    if getattr(getattr(bundle, "model", None).__class__, "__name__", None) != "ConditionalDiskAffineFlow":
        raise ValueError("selected-shape response requires a two-output disk flow")
    if kwargs.get("abs_shape_max") is not None:
        raise ValueError(
            "the selected-shape checkpoint is conditioned on the exact no-|e| event; "
            "an additional shape cut would change the estimand"
        )
    # The measured-catalogue branch still receives the command-line magnitude
    # and radius cuts. Only model-draw re-selection is disabled here.
    return _BASE_PREDICTION_SUFFICIENT(
        _SelectedShapeBundleAdapter(bundle),
        frame,
        **dict(kwargs, mag_max=None, radius_min_arcsec=None),
    )


def main() -> None:
    base._prediction_sufficient = _prediction_sufficient
    base.main()


if __name__ == "__main__":
    main()
