"""Reusable empirical prior over complete catalogue scenes.

The prior atom is one primary galaxy together with the neighbours already
present in the input catalogue.  Neighbour properties are therefore not drawn
independently: selecting an atom selects its full local scene.  A guarded CSR
graph is built once and reused when the atom is viewed at several trial shears.

This module deliberately stops before any model call.  It produces the three
different tables consumed downstream:

* one row per primary with all-neighbour flux summaries for the measurement
  flow;
* one row per primary with a configurable representative in-aperture neighbour
  for the detection classifier (nearest by default);
* one row per primary-neighbour pair for the blending-response emulator.

Keeping these views separate is important: the three models were trained with
different neighbour conventions and apertures.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .catalogue import Catalogue
from .coordinates import (
    ellipticity_from_axis_ratio_angle,
    spin2_angle_from_components,
)
from .forward_catalogue import (
    DEFAULT_CROWDING_RADII_ARCSEC,
    INPUT_CATALOGUE_COLUMNS,
    REQUIRED_CONDITIONS,
    EmulatorPairingConfig,
    rescale_emulator_pairs,
    select_response_pairs,
    validate_input_catalogue,
)
from .preprocessing import rescale
from .shear_map import apply_shear_to_ellipticity


SCENE_STORE_VERSION = 1
SHEAR_TRANSFORM = "intrinsic_ellipticity_only_v1"


def _log_neighbour_impact(
    secondary_magnitude: np.ndarray,
    secondary_size_arcsec: np.ndarray,
    distance_arcsec: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Rank neighbours by flux times ``(size / distance) ** exponent``."""

    magnitude = np.asarray(secondary_magnitude, dtype=float)
    size = np.asarray(secondary_size_arcsec, dtype=float)
    distance = np.asarray(distance_arcsec, dtype=float)
    if not np.isfinite(exponent) or exponent < 0:
        raise ValueError("impact exponent must be finite and non-negative")
    if (
        not np.isfinite(magnitude).all()
        or not np.isfinite(size).all()
        or (size <= 0).any()
        or not np.isfinite(distance).all()
        or (distance <= 0).any()
    ):
        raise ValueError(
            "impact score requires finite magnitudes and positive finite sizes "
            "and separations"
        )
    return (
        -0.4 * np.log(10.0) * magnitude
        + float(exponent) * (np.log(size) - np.log(distance))
    )


@dataclass(frozen=True)
class SceneBuildReport:
    """Auditable counts from constructing a scene prior."""

    n_galaxies: int
    n_directed_edges: int
    n_duplicate_positions: int
    guard_radius_arcsec: float
    group_column: Optional[str]


@dataclass(frozen=True)
class ScenePrior:
    """A finite empirical prior with a guarded, directed neighbour graph.

    ``indptr`` and ``secondary_row`` use CSR layout.  Edge offsets are in
    arcseconds in the same local flat-sky basis used by the existing SBSI
    catalogue preparation.  Every edge is directed, so a physical pair usually
    appears twice with opposite offsets.
    """

    galaxies: pd.DataFrame
    weights: np.ndarray
    indptr: np.ndarray
    secondary_row: np.ndarray
    dx_arcsec: np.ndarray
    dy_arcsec: np.ndarray
    guard_radius_arcsec: float
    group_column: Optional[str] = None
    report: Optional[SceneBuildReport] = None
    metadata: Optional[Mapping] = None

    def __post_init__(self):
        n = len(self.galaxies)
        weights = np.asarray(self.weights, dtype=np.float64)
        indptr = np.asarray(self.indptr, dtype=np.int64)
        secondary = np.asarray(self.secondary_row, dtype=np.int64)
        dx = np.asarray(self.dx_arcsec, dtype=np.float64)
        dy = np.asarray(self.dy_arcsec, dtype=np.float64)
        if n == 0:
            raise ValueError("scene prior is empty")
        if len(weights) != n or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite, non-negative, and one per galaxy")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("at least one prior weight must be positive")
        if indptr.shape != (n + 1,) or indptr[0] != 0 or indptr[-1] != len(secondary):
            raise ValueError("indptr is not a valid CSR pointer array")
        if (np.diff(indptr) < 0).any():
            raise ValueError("indptr must be nondecreasing")
        if len(dx) != len(secondary) or len(dy) != len(secondary):
            raise ValueError("edge arrays must have equal length")
        if ((secondary < 0) | (secondary >= n)).any():
            raise ValueError("secondary_row contains an out-of-range row")
        if not np.isfinite(dx).all() or not np.isfinite(dy).all():
            raise ValueError("edge offsets must be finite")
        if self.guard_radius_arcsec <= 0:
            raise ValueError("guard_radius_arcsec must be positive")
        if self.group_column is not None and self.group_column not in self.galaxies:
            raise KeyError(f"galaxies lack group column {self.group_column!r}")
        object.__setattr__(self, "galaxies", self.galaxies.reset_index(drop=True).copy())
        object.__setattr__(self, "weights", weights / total)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "secondary_row", secondary)
        object.__setattr__(self, "dx_arcsec", dx)
        object.__setattr__(self, "dy_arcsec", dy)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_catalogue(
        cls,
        catalogue: Catalogue,
        *,
        guard_radius_arcsec: float,
        weight_column: Optional[str] = None,
        group_column: Optional[str] = None,
        duplicate_policy: str = "error",
    ) -> "ScenePrior":
        """Process a truth catalogue once into finite prior atoms.

        Exact duplicate positions inside one group are rejected by default.
        They are ambiguous to a self-excluding neighbour finder: a zero-distance
        physical neighbour cannot be distinguished from the primary itself.
        """

        if guard_radius_arcsec <= 0:
            raise ValueError("guard_radius_arcsec must be positive")
        if duplicate_policy not in {"error", "drop"}:
            raise ValueError("duplicate_policy must be 'error' or 'drop'")
        frame = validate_input_catalogue(catalogue)
        if "position_angle" not in frame:
            raise KeyError("scene catalogue is missing column: position_angle")
        if group_column is not None and group_column not in frame:
            raise KeyError(f"scene catalogue lacks group column {group_column!r}")

        duplicate_subset = ["RA", "DEC"]
        duplicated = frame.duplicated(
            subset=([group_column] if group_column is not None else []) + duplicate_subset,
            keep=False,
        )
        n_duplicate_positions = int(duplicated.sum())
        if n_duplicate_positions:
            if duplicate_policy == "error":
                raise ValueError(
                    f"found {n_duplicate_positions} rows at duplicate positions; "
                    "deduplicate explicitly or pass duplicate_policy='drop'"
                )
            frame = frame.drop_duplicates(
                subset=([group_column] if group_column is not None else []) + duplicate_subset,
                keep="first",
            ).reset_index(drop=True)

        if weight_column is None:
            weights = np.ones(len(frame), dtype=np.float64)
        else:
            if weight_column not in frame:
                raise KeyError(f"scene catalogue lacks weight column {weight_column!r}")
            weights = frame[weight_column].to_numpy(dtype=np.float64)

        primary_edge: list[np.ndarray] = []
        secondary_edge: list[np.ndarray] = []
        dx_edge: list[np.ndarray] = []
        dy_edge: list[np.ndarray] = []
        if group_column is None:
            groups = [(None, np.arange(len(frame), dtype=np.int64))]
        else:
            groups = list(frame.groupby(group_column, sort=False, dropna=False).indices.items())

        radius_degrees = guard_radius_arcsec / 3600.0
        for _, rows_value in groups:
            rows = np.asarray(rows_value, dtype=np.int64)
            positions = frame.loc[rows, ["RA", "DEC"]].to_numpy(dtype=np.float64)
            neighbours = KDTree(positions).query_ball_point(
                positions, r=radius_degrees, workers=-1
            )
            for local_primary, local_secondaries_value in enumerate(neighbours):
                local_secondaries = np.asarray(local_secondaries_value, dtype=np.int64)
                local_secondaries = local_secondaries[local_secondaries != local_primary]
                if not len(local_secondaries):
                    continue
                delta = (positions[local_secondaries] - positions[local_primary]) * 3600.0
                distance = np.hypot(delta[:, 0], delta[:, 1])
                order = np.argsort(distance, kind="stable")
                count = len(order)
                primary_edge.append(np.full(count, rows[local_primary], dtype=np.int64))
                secondary_edge.append(rows[local_secondaries[order]])
                dx_edge.append(delta[order, 0])
                dy_edge.append(delta[order, 1])

        if primary_edge:
            primary = np.concatenate(primary_edge)
            secondary = np.concatenate(secondary_edge)
            dx = np.concatenate(dx_edge)
            dy = np.concatenate(dy_edge)
            order = np.lexsort((np.hypot(dx, dy), primary))
            primary, secondary, dx, dy = (
                values[order] for values in (primary, secondary, dx, dy)
            )
            counts = np.bincount(primary, minlength=len(frame))
        else:
            secondary = np.empty(0, dtype=np.int64)
            dx = np.empty(0, dtype=np.float64)
            dy = np.empty(0, dtype=np.float64)
            counts = np.zeros(len(frame), dtype=np.int64)
        indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        report = SceneBuildReport(
            n_galaxies=len(frame),
            n_directed_edges=len(secondary),
            n_duplicate_positions=n_duplicate_positions,
            guard_radius_arcsec=float(guard_radius_arcsec),
            group_column=group_column,
        )
        return cls(
            frame,
            weights,
            indptr,
            secondary,
            dx,
            dy,
            float(guard_radius_arcsec),
            group_column,
            report,
            {},
        )

    def shear(self, g1: float, g2: float, *, kappa: float = 0.0) -> "ShearedScenePrior":
        """Return a model-ready view of the same atoms at a trial shear."""

        return ShearedScenePrior.from_prior(self, g1=g1, g2=g2, kappa=kappa)

    def reweighted(self, weights: np.ndarray, *, metadata: Optional[Mapping] = None) -> "ScenePrior":
        """Reuse the complete scene graph with a new atom-mass vector."""

        merged_metadata = dict(self.metadata or {})
        merged_metadata.update(dict(metadata or {}))
        return ScenePrior(
            self.galaxies,
            weights,
            self.indptr,
            self.secondary_row,
            self.dx_arcsec,
            self.dy_arcsec,
            self.guard_radius_arcsec,
            self.group_column,
            self.report,
            merged_metadata,
        )

    def save(self, path: str | Path, *, metadata: Optional[Mapping] = None) -> None:
        """Write the model-independent scene store to a new or existing directory."""

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.galaxies.to_parquet(root / "galaxies.parquet", index=False)
        np.savez_compressed(
            root / "neighbours.npz",
            weights=self.weights,
            indptr=self.indptr,
            secondary_row=self.secondary_row,
            dx_arcsec=self.dx_arcsec,
            dy_arcsec=self.dy_arcsec,
        )
        saved_metadata = dict(self.metadata or {})
        saved_metadata.update(dict(metadata or {}))
        payload = {
            "version": SCENE_STORE_VERSION,
            "guard_radius_arcsec": self.guard_radius_arcsec,
            "group_column": self.group_column,
            "required_input_columns": list(INPUT_CATALOGUE_COLUMNS),
            "report": None if self.report is None else self.report.__dict__,
            "metadata": saved_metadata,
        }
        (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ScenePrior":
        """Load a scene store created by :meth:`save`."""

        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version") != SCENE_STORE_VERSION:
            raise ValueError(
                f"unsupported scene store version {manifest.get('version')!r}; "
                f"expected {SCENE_STORE_VERSION}"
            )
        arrays = np.load(root / "neighbours.npz")
        report_payload = manifest.get("report")
        report = None if report_payload is None else SceneBuildReport(**report_payload)
        return cls(
            pd.read_parquet(root / "galaxies.parquet"),
            arrays["weights"],
            arrays["indptr"],
            arrays["secondary_row"],
            arrays["dx_arcsec"],
            arrays["dy_arcsec"],
            float(manifest["guard_radius_arcsec"]),
            manifest.get("group_column"),
            report,
            manifest.get("metadata") or {},
        )


@dataclass(frozen=True)
class ShearedScenePrior:
    """One shape-only sheared view of a :class:`ScenePrior`.

    This follows the image simulations used by SBSI: reduced shear changes
    every object's intrinsic ellipticity, while flux, size, sky position, pair
    separation, and neighbour membership remain fixed.
    """

    prior: ScenePrior
    galaxies: pd.DataFrame
    dx_arcsec: np.ndarray
    dy_arcsec: np.ndarray
    g1: float
    g2: float
    kappa: float = 0.0

    @classmethod
    def from_prior(
        cls, prior: ScenePrior, *, g1: float, g2: float, kappa: float = 0.0
    ) -> "ShearedScenePrior":
        g1, g2, kappa = float(g1), float(g2), float(kappa)
        if kappa != 0.0:
            raise ValueError(
                "shape-only SBSI shear does not implement convergence/magnification"
            )
        if np.hypot(g1, g2) >= abs(1.0 - kappa):
            raise ValueError("shear map is singular or parity-flipping")
        frame = prior.galaxies.copy()

        e1, e2 = ellipticity_from_axis_ratio_angle(
            frame["axis_ratio"].to_numpy(dtype=float),
            frame["position_angle"].to_numpy(dtype=float),
        )
        se1, se2 = apply_shear_to_ellipticity(e1, e2, g1, g2)
        eabs = np.hypot(se1, se2)
        if (eabs >= 1.0).any():
            raise ValueError("sheared ellipticity left the physical unit disc")
        frame["axis_ratio"] = (1.0 - eabs) / (1.0 + eabs)
        angle = spin2_angle_from_components(se1, se2)
        raw_angle = frame["position_angle"].to_numpy(dtype=float)
        finite = np.isfinite(raw_angle)
        degrees = bool(finite.any() and np.nanpercentile(np.abs(raw_angle[finite]), 95) > 2 * np.pi * 1.1)
        frame["position_angle"] = np.rad2deg(angle) if degrees else angle

        frame["gamma1"] = g1
        frame["gamma2"] = g2
        frame["shear_component_convention"] = "sky_cos_sin"

        return cls(
            prior,
            frame,
            prior.dx_arcsec.copy(),
            prior.dy_arcsec.copy(),
            g1,
            g2,
            kappa,
        )

    @cached_property
    def distance_arcsec(self) -> np.ndarray:
        return np.hypot(self.dx_arcsec, self.dy_arcsec)

    @cached_property
    def primary_row(self) -> np.ndarray:
        return np.repeat(
            np.arange(len(self.galaxies), dtype=np.int64),
            np.diff(self.prior.indptr),
        )

    def _require_supported_radius(self, radius_arcsec: float) -> None:
        if radius_arcsec <= 0:
            raise ValueError("radius must be positive")
        required_guard = radius_arcsec
        if required_guard > self.prior.guard_radius_arcsec * (1.0 + 1e-12):
            raise ValueError(
                f"radius {radius_arcsec:g} arcsec at |g|={np.hypot(self.g1, self.g2):g} "
                f"needs guard >= {required_guard:g} arcsec; store has "
                f"{self.prior.guard_radius_arcsec:g}"
            )

    def _edge_rows(self, primary_row: int, radius_arcsec: float) -> np.ndarray:
        start = self.prior.indptr[primary_row]
        stop = self.prior.indptr[primary_row + 1]
        rows = np.arange(start, stop, dtype=np.int64)
        rows = rows[self.distance_arcsec[rows] < radius_arcsec]
        if len(rows):
            rows = rows[np.argsort(self.distance_arcsec[rows], kind="stable")]
        return rows

    def _nearest_edges(self, radius_arcsec: float) -> np.ndarray:
        """Nearest transformed edge per primary, or -1 for an isolate."""

        n = len(self.galaxies)
        out = np.full(n, -1, dtype=np.int64)
        valid = np.flatnonzero(self.distance_arcsec < radius_arcsec)
        if not len(valid):
            return out
        order = np.lexsort((self.distance_arcsec[valid], self.primary_row[valid]))
        ordered = valid[order]
        primary = self.primary_row[ordered]
        first = np.r_[True, primary[1:] != primary[:-1]]
        out[primary[first]] = ordered[first]
        return out

    def _impact_edges(self, radius_arcsec: float, exponent: float) -> np.ndarray:
        """Highest-impact edge per primary, or -1 for an isolate.

        The ranking is proportional to

        ``flux_secondary * (Re_secondary / distance)**exponent``.

        It is evaluated in log space; the photometric zero point is common to
        every candidate and therefore cancels from the ranking.  Stable tie
        breaking prefers the smaller distance and then the smaller secondary
        row, making the selected identity reproducible.
        """

        if not np.isfinite(exponent) or exponent < 0:
            raise ValueError("impact exponent must be finite and non-negative")
        required = {"r", "Re"}
        missing = sorted(required - set(self.galaxies.columns))
        if missing:
            raise KeyError(f"impact-ranked neighbours require columns: {missing}")

        n = len(self.galaxies)
        out = np.full(n, -1, dtype=np.int64)
        valid = np.flatnonzero(self.distance_arcsec < radius_arcsec)
        if not len(valid):
            return out
        secondary = self.prior.secondary_row[valid]
        magnitude = self.galaxies["r"].to_numpy(dtype=float)[secondary]
        size = self.galaxies["Re"].to_numpy(dtype=float)[secondary]
        distance = self.distance_arcsec[valid]
        if (
            not np.isfinite(magnitude).all()
            or not np.isfinite(size).all()
            or (size <= 0).any()
            or not np.isfinite(distance).all()
            or (distance <= 0).any()
        ):
            raise ValueError(
                "impact-ranked neighbours require finite magnitudes and positive "
                "finite sizes and separations"
            )
        log_impact = _log_neighbour_impact(magnitude, size, distance, exponent)
        order = np.lexsort(
            (
                secondary,
                distance,
                -log_impact,
                self.primary_row[valid],
            )
        )
        ordered = valid[order]
        primary = self.primary_row[ordered]
        first = np.r_[True, primary[1:] != primary[:-1]]
        out[primary[first]] = ordered[first]
        return out

    def _representative_edges(
        self,
        radius_arcsec: float,
        neighbour_selection: str,
        impact_exponent: float,
    ) -> np.ndarray:
        if neighbour_selection == "nearest":
            return self._nearest_edges(radius_arcsec)
        if neighbour_selection == "impact":
            return self._impact_edges(radius_arcsec, impact_exponent)
        raise ValueError("neighbour_selection must be 'nearest' or 'impact'")

    def object_view(
        self,
        *,
        neighbour_radius_arcsec: float,
        conditions: Mapping[str, float],
        crowding_radii_arcsec: Sequence[float] = DEFAULT_CROWDING_RADII_ARCSEC,
        include_crowding: bool = True,
        neighbour_selection: str = "nearest",
        impact_exponent: float = 2.0,
        primary_indices: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Build a one-row-per-primary, rescaled model table.

        The representative-neighbour block uses ``neighbour_radius_arcsec``.
        ``neighbour_selection='impact'`` ranks candidates by secondary flux
        times ``(Re_secondary / distance)**impact_exponent``.  Crowding fluxes,
        when requested, independently use all edges in the two shells.
        """

        missing = sorted(set(REQUIRED_CONDITIONS) - set(conditions))
        if missing:
            raise KeyError(f"observing conditions are missing: {missing}")
        near, far = (float(v) for v in crowding_radii_arcsec)
        if near <= 0 or far <= near:
            raise ValueError("crowding radii must satisfy 0 < near < far")
        self._require_supported_radius(max(neighbour_radius_arcsec, far if include_crowding else 0.0))

        n_all = len(self.galaxies)
        if primary_indices is None:
            source_rows = np.arange(n_all, dtype=np.int64)
        else:
            source_rows = np.asarray(primary_indices, dtype=np.int64)
            if source_rows.ndim != 1 or (
                (source_rows < 0) | (source_rows >= n_all)
            ).any():
                raise ValueError("primary_indices are invalid")
            if len(np.unique(source_rows)) != len(source_rows):
                raise ValueError("primary_indices must be unique")

        representative_edge_all = self._representative_edges(
            neighbour_radius_arcsec,
            neighbour_selection,
            impact_exponent,
        )
        representative_edge = representative_edge_all[source_rows]
        n = len(source_rows)
        representative_secondary = np.full(n, -1, dtype=np.int64)
        matched = representative_edge >= 0
        representative_secondary[matched] = self.prior.secondary_row[
            representative_edge[matched]
        ]
        primary = self.galaxies.iloc[source_rows].reset_index(drop=True).add_suffix("_input_p")
        primary.insert(0, "primary_row", source_rows)
        if "index" in self.galaxies:
            primary.insert(
                1,
                "input_index",
                self.galaxies.iloc[source_rows]["index"].to_numpy(copy=True),
            )
        else:
            primary.insert(1, "input_index", source_rows)
        if self.prior.group_column is not None:
            primary[self.prior.group_column] = self.galaxies.iloc[source_rows][
                self.prior.group_column
            ].to_numpy(copy=True)

        primary["secondary_row"] = representative_secondary
        primary["neighbored"] = matched
        primary["distance"] = np.nan
        primary["polarization_angle"] = np.nan
        primary["neighbour_log_impact"] = np.nan
        primary.loc[matched, "distance"] = self.distance_arcsec[
            representative_edge[matched]
        ]
        primary.loc[matched, "polarization_angle"] = np.arctan2(
            self.dy_arcsec[representative_edge[matched]],
            self.dx_arcsec[representative_edge[matched]],
        )
        if matched.any():
            sec_mag = self.galaxies["r"].to_numpy(dtype=float)[
                representative_secondary[matched]
            ]
            sec_size = self.galaxies["Re"].to_numpy(dtype=float)[
                representative_secondary[matched]
            ]
            separation = self.distance_arcsec[representative_edge[matched]]
            primary.loc[matched, "neighbour_log_impact"] = _log_neighbour_impact(
                sec_mag, sec_size, separation, impact_exponent
            )

        for column in self.galaxies.columns:
            values = np.full(n, np.nan, dtype=object)
            if matched.any():
                values[matched] = self.galaxies.iloc[
                    representative_secondary[matched]
                ][column].to_numpy()
            primary[f"{column}_input_s"] = values
        for column in self.galaxies.select_dtypes(include=[np.number, "bool"]).columns:
            primary[f"{column}_input_s"] = pd.to_numeric(
                primary[f"{column}_input_s"], errors="coerce"
            )

        e1, e2 = ellipticity_from_axis_ratio_angle(
            self.galaxies["axis_ratio"].to_numpy(dtype=float),
            self.galaxies["position_angle"].to_numpy(dtype=float),
        )
        primary["e1_input_rot0_p"] = e1[source_rows]
        primary["e2_input_rot0_p"] = e2[source_rows]

        if include_crowding:
            zero_point = float(conditions["zero_point"])
            beta = float(conditions["moffat_beta"])
            fwhm = float(conditions["psf_fwhm"])
            psf_size = fwhm * np.sqrt(
                (2 ** (1 / (beta - 1)) - 1) / (2 ** (1 / beta) - 1)
            ) / 2
            aperture_rms = (
                float(conditions["pixel_rms"])
                * (psf_size / float(conditions["pixel_size"])) ** 2
                * np.pi
            )
            flux = 10 ** (-0.4 * (self.galaxies["r"].to_numpy(dtype=float) - zero_point))
            near_flux = np.zeros(n)
            far_flux = np.zeros(n)
            max_flux = np.zeros(n)
            edges = np.flatnonzero(self.distance_arcsec < far)
            if len(edges):
                primary_edges = self.primary_row[edges]
                distances = self.distance_arcsec[edges]
                values = flux[self.prior.secondary_row[edges]]
                is_near = distances < near
                np.add.at(near_flux, primary_edges[is_near], values[is_near])
                np.add.at(far_flux, primary_edges[~is_near], values[~is_near])
                np.maximum.at(max_flux, primary_edges, values)
            primary["nbr_flux_near"] = np.log10(
                1.0 + near_flux[source_rows] / aperture_rms
            )
            primary["nbr_flux_far"] = np.log10(
                1.0 + far_flux[source_rows] / aperture_rms
            )
            primary["nbr_flux_max"] = np.log10(
                1.0 + max_flux[source_rows] / aperture_rms
            )

        return rescale(
            primary.set_index("primary_row", drop=True),
            pixel_rms=float(conditions["pixel_rms"]),
            pixel_size=float(conditions["pixel_size"]),
            zero_mag=float(conditions["zero_point"]),
            psf_fwhm=float(conditions["psf_fwhm"]),
            moffat_beta=float(conditions["moffat_beta"]),
        )

    def flow_view(
        self,
        *,
        conditions: Mapping[str, float],
        neighbour_radius_arcsec: float = DEFAULT_CROWDING_RADII_ARCSEC[1],
        crowding_radii_arcsec: Sequence[float] = DEFAULT_CROWDING_RADII_ARCSEC,
    ) -> pd.DataFrame:
        """One-row flow view with the trained all-neighbour shell summaries."""

        return self.object_view(
            neighbour_radius_arcsec=neighbour_radius_arcsec,
            conditions=conditions,
            crowding_radii_arcsec=crowding_radii_arcsec,
            include_crowding=True,
        )

    def detection_view(
        self,
        *,
        conditions: Mapping[str, float],
        radius_arcsec: float,
        neighbour_selection: str = "nearest",
        impact_exponent: float = 2.0,
        primary_indices: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """One-row classifier view using one representative in-aperture neighbour."""

        return self.object_view(
            neighbour_radius_arcsec=radius_arcsec,
            conditions=conditions,
            include_crowding=False,
            neighbour_selection=neighbour_selection,
            impact_exponent=impact_exponent,
            primary_indices=primary_indices,
        )

    def response_pairs(
        self,
        *,
        config: EmulatorPairingConfig,
        primary_indices: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Training-supported regression pairs for selected primary rows.

        ``primary_indices`` limits only which galaxies act as primaries.  Every
        scene row remains eligible as a secondary, preserving the catalogue's
        complete clustered environment while avoiding a many-pair table for
        zero-mass atoms that can never enter the finite prior sum.
        """

        self._require_supported_radius(config.r_max_arcsec)
        edges = np.flatnonzero(self.distance_arcsec < config.r_max_arcsec)
        if primary_indices is not None:
            selected = np.asarray(primary_indices, dtype=np.int64)
            if selected.ndim != 1 or (
                (selected < 0) | (selected >= len(self.galaxies))
            ).any():
                raise ValueError("response primary indices are invalid")
            if len(selected) and (np.diff(selected) <= 0).any():
                raise ValueError(
                    "response primary indices must be strictly increasing"
                )
            primary_mask = np.zeros(len(self.galaxies), dtype=bool)
            primary_mask[selected] = True
            edges = edges[primary_mask[self.primary_row[edges]]]
        if not len(edges):
            raise ValueError("no neighbours lie inside the response aperture")
        order = np.lexsort((self.distance_arcsec[edges], self.primary_row[edges]))
        edges = edges[order]
        primary = self.primary_row[edges]
        _, starts, counts = np.unique(primary, return_index=True, return_counts=True)
        rank = np.arange(len(edges)) - np.repeat(starts, counts)
        edges = edges[rank < config.k]
        primary = self.primary_row[edges]
        secondary = self.prior.secondary_row[edges]
        pri = self.galaxies.iloc[primary].reset_index(drop=True).add_suffix("_input_p")
        sec = self.galaxies.iloc[secondary].reset_index(drop=True).add_suffix("_input_s")
        pairs = pd.concat((pri, sec), axis=1)
        pairs.insert(0, "secondary_row", secondary)
        pairs.insert(0, "primary_row", primary)
        pairs["distance"] = self.distance_arcsec[edges]
        pairs["polarization_angle"] = np.arctan2(self.dy_arcsec[edges], self.dx_arcsec[edges])
        pairs = select_response_pairs(pairs, config.cuts)
        if pairs.empty:
            raise ValueError("no response pairs remain after emulator training cuts")
        return rescale_emulator_pairs(pairs, config.conditions)


__all__ = [
    "SCENE_STORE_VERSION",
    "SceneBuildReport",
    "ScenePrior",
    "ShearedScenePrior",
]
