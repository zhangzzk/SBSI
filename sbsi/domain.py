"""Model-support metadata inferred from a trained measurement flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    """Open rectangular support in primary magnitude and half-light radius."""

    magnitude_min: float
    magnitude_max: float
    half_light_radius_min: float
    half_light_radius_max: float

    def mask(self, magnitude, half_light_radius):
        magnitude = np.asarray(magnitude, dtype=float)
        radius = np.asarray(half_light_radius, dtype=float)
        return (
            np.isfinite(magnitude)
            & np.isfinite(radius)
            & (magnitude > self.magnitude_min)
            & (magnitude < self.magnitude_max)
            & (radius > self.half_light_radius_min)
            & (radius < self.half_light_radius_max)
        )

    @classmethod
    def from_flow_metadata(cls, metadata) -> "Domain":
        """Read the training support stored in a flow checkpoint.

        The flow records its training population as one true-magnitude cut,
        ``r_input_p < value``; it sets no size limit.
        """

        cut = (metadata.get("population") or {}).get("truth_analysis_cut") or {}
        if cut.get("column") != "r_input_p" or cut.get("operator") != "<":
            raise ValueError("flow checkpoint does not record its r_input_p < value training cut")
        magnitude_max = float(cut["value"])
        if not np.isfinite(magnitude_max):
            raise ValueError("flow checkpoint has a non-finite magnitude cut")
        return cls(
            magnitude_min=-np.inf,
            magnitude_max=magnitude_max,
            half_light_radius_min=0.0,
            half_light_radius_max=np.inf,
        )


__all__ = ["Domain"]
