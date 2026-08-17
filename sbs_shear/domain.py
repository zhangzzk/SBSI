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
        """Read the training support stored in a flow checkpoint."""

        cuts = metadata.get("selection_cuts")
        if cuts is None or len(cuts) < 4:
            raise ValueError("flow checkpoint does not record primary selection_cuts")
        magnitude = cuts[1]
        radius = cuts[3]
        if len(magnitude) != 2 or len(radius) != 2:
            raise ValueError("flow checkpoint has malformed primary selection_cuts")
        values = [*magnitude, *radius]
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError("flow checkpoint has non-finite primary selection_cuts")
        return cls(
            magnitude_min=float(magnitude[0]),
            magnitude_max=float(magnitude[1]),
            half_light_radius_min=float(radius[0]),
            half_light_radius_max=float(radius[1]),
        )


__all__ = ["Domain"]
