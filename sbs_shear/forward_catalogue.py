"""Prepare user input catalogues for forward response inference.

Image simulation and measurement remain BlendEMU responsibilities.  This
module owns the lightweight inference-time transformation from a user galaxy
catalogue to both model views needed for response prediction:

* one row per primary for the measurement-flow self-response;
* one row per primary-neighbour pair for the blending emulator.

The algorithm is adapted from BlendEMU's ``utils.kdt_neighbor_finder``,
``nz_utils.make_reg_features``, and ``data_utils.rescale`` implementations so
SBSI does not delegate catalogue semantics to an opaque model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .catalogue import Catalogue, load_catalogue
from .coordinates import ellipticity_from_axis_ratio_angle


INPUT_CATALOGUE_COLUMNS = (
    "RA",
    "DEC",
    "redshift",
    "r",
    "Re",
    "sersic_n",
    "axis_ratio",
)
PAIR_INDEX_COLUMNS = ("primary_row", "secondary_row")
REQUIRED_CONDITIONS = (
    "pixel_size",
    "zero_point",
    "psf_fwhm",
    "moffat_beta",
    "pixel_rms",
)
DEFAULT_CROWDING_RADII_ARCSEC = (3.0, 7.0)


def _normalise_cuts(cuts: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float], ...]:
    if len(cuts) != 5 or any(len(bounds) != 2 for bounds in cuts):
        raise ValueError("emulator cuts must contain five (min, max) pairs")
    normalised = tuple((float(bounds[0]), float(bounds[1])) for bounds in cuts)
    if any(lower >= upper for lower, upper in normalised):
        raise ValueError("every emulator cut must have min < max")
    return normalised


@dataclass(frozen=True)
class EmulatorPairingConfig:
    """Training-matched settings for one emulator's pair catalogue."""

    cuts: Tuple[Tuple[float, float], ...]
    r_max_arcsec: float
    k: int
    conditions: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "cuts", _normalise_cuts(self.cuts))
        object.__setattr__(self, "conditions", dict(self.conditions))
        if self.r_max_arcsec <= 0:
            raise ValueError("r_max_arcsec must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        missing = sorted(set(REQUIRED_CONDITIONS) - set(self.conditions))
        if missing:
            raise KeyError(f"observing conditions are missing: {missing}")

    @classmethod
    def from_emulator(cls, emulator, task: str = "regression") -> "EmulatorPairingConfig":
        """Recover pairing and observing settings stored with a BlendEMU model."""

        if not hasattr(emulator, "select") or not hasattr(emulator, "conditions"):
            raise TypeError(
                "emulator must expose BlendEMU's model metadata and observing conditions"
            )
        settings = emulator.select.get(task, {})
        cuts = settings.get("cuts")
        r_max = settings.get("r_max")
        k = settings.get("k")
        missing = [name for name, value in (("cuts", cuts), ("r_max", r_max), ("k", k)) if value is None]
        if missing:
            raise ValueError(
                "emulator metadata does not record training-matched "
                f"{', '.join(missing)}; pass EmulatorPairingConfig explicitly"
            )
        return cls(
            cuts=_normalise_cuts(cuts),
            r_max_arcsec=float(r_max),
            k=int(k),
            conditions=emulator.conditions,
        )


@dataclass(frozen=True)
class PreparedForwardCatalogue:
    """Aligned object- and pair-level views derived from one input catalogue.

    ``flow_inputs`` has one row per retained primary and is indexed by the
    original ``primary_row``.  ``emulator_pairs`` may have several rows per
    primary.  Keeping the views separate prevents the flow response from being
    accidentally weighted by neighbour multiplicity.
    """

    flow_inputs: pd.DataFrame
    emulator_pairs: pd.DataFrame

    def __post_init__(self):
        if self.flow_inputs.index.name != "primary_row":
            raise ValueError("flow_inputs must be indexed by primary_row")
        if "primary_row" not in self.emulator_pairs.columns:
            raise KeyError("emulator_pairs lacks primary_row")
        object_rows = self.flow_inputs.index.to_numpy(dtype=np.int64)
        pair_rows = self.emulator_pairs["primary_row"].to_numpy(dtype=np.int64)
        if not np.isin(pair_rows, object_rows).all():
            raise ValueError("emulator_pairs contains a primary absent from flow_inputs")
        if set(object_rows) != set(pair_rows):
            raise ValueError("every flow primary must have at least one emulator pair")


def validate_input_catalogue(catalogue: Catalogue) -> pd.DataFrame:
    """Load and validate the truth-level catalogue used for forward inference."""

    frame = load_catalogue(catalogue)
    missing = sorted(set(INPUT_CATALOGUE_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"input catalogue is missing columns: {missing}")
    if frame.empty:
        raise ValueError("input catalogue is empty")
    positions = frame.loc[:, ["RA", "DEC"]].to_numpy(dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("RA and DEC must be finite")
    return frame.reset_index(drop=True)


def find_neighbours(
    primary_positions: np.ndarray,
    secondary_positions: np.ndarray,
    *,
    r_min: float = 0.0,
    r_max: float,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find up to ``k`` neighbours per primary within the supplied radii.

    Radii use the same units as the two position arrays.  The strict
    ``distance > r_min`` test excludes a galaxy from matching itself when the
    same catalogue is supplied on both sides.
    """

    primary_positions = np.asarray(primary_positions, dtype=float)
    secondary_positions = np.asarray(secondary_positions, dtype=float)
    for name, positions in (
        ("primary", primary_positions),
        ("secondary", secondary_positions),
    ):
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"{name}_positions must have shape (n, 2)")
        if not np.isfinite(positions).all():
            raise ValueError(f"{name}_positions must be finite")
    if len(secondary_positions) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )
    if r_min < 0 or r_max <= r_min:
        raise ValueError("neighbour radii must satisfy 0 <= r_min < r_max")
    if k <= 0:
        raise ValueError("k must be positive")

    distances, indices = KDTree(secondary_positions).query(
        primary_positions,
        k=k,
        distance_upper_bound=r_max,
        workers=-1,
    )
    distances = np.asarray(distances).reshape(len(primary_positions), -1)
    indices = np.asarray(indices).reshape(len(primary_positions), -1)
    valid = (indices != len(secondary_positions)) & (distances > r_min)
    primary_rows = np.broadcast_to(
        np.arange(len(primary_positions), dtype=np.int64)[:, None],
        valid.shape,
    )[valid]
    return primary_rows, indices[valid].astype(np.int64), distances[valid]


def make_pair_catalogue(
    primary_catalogue: Catalogue,
    secondary_catalogue: Optional[Catalogue] = None,
    *,
    r_min_arcsec: float = 0.0,
    r_max_arcsec: float,
    k: int,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Build a primary-neighbour table from one or two input catalogues.

    Sky positions are interpreted as degrees and output ``distance`` is in
    arcseconds.  ``primary_row`` always refers to the row of the loaded primary
    catalogue, so predictions can be reduced and aligned without relying on a
    user index.
    """

    primary = validate_input_catalogue(primary_catalogue)
    secondary = (
        primary.copy()
        if secondary_catalogue is None
        else validate_input_catalogue(secondary_catalogue)
    )
    if group_column is not None:
        for name, frame in (("primary", primary), ("secondary", secondary)):
            if group_column not in frame.columns:
                raise KeyError(f"{name} catalogue lacks group column {group_column!r}")
        grouped = primary.groupby(group_column, sort=False, dropna=False).indices.items()
    else:
        grouped = [(None, np.arange(len(primary), dtype=np.int64))]

    parts = []
    for group, primary_rows in grouped:
        primary_rows = np.asarray(primary_rows, dtype=np.int64)
        if group_column is None:
            secondary_rows = np.arange(len(secondary), dtype=np.int64)
        elif pd.isna(group):
            secondary_rows = np.flatnonzero(secondary[group_column].isna().to_numpy())
        else:
            secondary_rows = np.flatnonzero(
                secondary[group_column].eq(group).to_numpy()
            )
        local_primary, local_secondary, separation = find_neighbours(
            primary.loc[primary_rows, ["RA", "DEC"]].to_numpy(dtype=float),
            secondary.loc[secondary_rows, ["RA", "DEC"]].to_numpy(dtype=float),
            r_min=r_min_arcsec / 3600.0,
            r_max=r_max_arcsec / 3600.0,
            k=k,
        )
        if len(local_primary) == 0:
            continue
        global_primary = primary_rows[local_primary]
        global_secondary = secondary_rows[local_secondary]
        pri = primary.iloc[global_primary].reset_index(drop=True).add_suffix("_input_p")
        sec = secondary.iloc[global_secondary].reset_index(drop=True).add_suffix("_input_s")
        pair = pd.concat((pri, sec), axis=1)
        pair.insert(0, "secondary_row", global_secondary)
        pair.insert(0, "primary_row", global_primary)
        pair["distance"] = separation * 3600.0
        parts.append(pair)

    if not parts:
        columns = [
            *PAIR_INDEX_COLUMNS,
            *(f"{column}_input_p" for column in primary.columns),
            *(f"{column}_input_s" for column in secondary.columns),
            "distance",
        ]
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def select_response_pairs(
    pairs: pd.DataFrame,
    cuts: Sequence[Sequence[float]],
) -> pd.DataFrame:
    """Apply the emulator training cuts to an unscaled pair catalogue."""

    cuts = _normalise_cuts(cuts)
    required = {
        "r_input_s",
        "r_input_p",
        "Re_input_s",
        "Re_input_p",
        "distance",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise KeyError(f"pair catalogue is missing selection columns: {missing}")
    keep = (
        pairs["r_input_s"].between(*cuts[0], inclusive="neither")
        & pairs["r_input_p"].between(*cuts[1], inclusive="neither")
        & pairs["Re_input_s"].between(*cuts[2], inclusive="neither")
        & pairs["Re_input_p"].between(*cuts[3], inclusive="neither")
        & pairs["distance"].between(*cuts[4], inclusive="neither")
    )
    return pairs.loc[keep].reset_index(drop=True)


def rescale_emulator_pairs(
    pairs: pd.DataFrame,
    conditions: Mapping[str, float],
) -> pd.DataFrame:
    """Add the observation-condition-scaled features used by BlendEMU models."""

    missing_conditions = sorted(set(REQUIRED_CONDITIONS) - set(conditions))
    if missing_conditions:
        raise KeyError(f"observing conditions are missing: {missing_conditions}")
    required = {
        "Re_input_p",
        "Re_input_s",
        "r_input_p",
        "r_input_s",
        "distance",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise KeyError(f"pair catalogue is missing rescaling columns: {missing}")

    out = pairs.copy()
    beta = float(conditions["moffat_beta"])
    fwhm = float(conditions["psf_fwhm"])
    psf_size = fwhm * np.sqrt(
        (2 ** (1 / (beta - 1)) - 1) / (2 ** (1 / beta) - 1)
    ) / 2
    pixel_size = float(conditions["pixel_size"])
    pixel_rms = float(conditions["pixel_rms"])
    zero_point = float(conditions["zero_point"])
    aperture_rms = pixel_rms * (psf_size / pixel_size) ** 2 * np.pi

    re_p = out["Re_input_p"].to_numpy(dtype=float)
    re_s = out["Re_input_s"].to_numpy(dtype=float)
    post_re_p = np.sqrt(re_p**2 + psf_size**2)
    post_re_s = np.sqrt(re_s**2 + psf_size**2)
    out["distance_scaled"] = out["distance"] / post_re_p
    out["Re_input_p_scaled"] = re_p / post_re_p
    out["Re_input_s_scaled"] = re_s / post_re_s

    flux_p = 10 ** (-0.4 * (out["r_input_p"] - zero_point))
    flux_s = 10 ** (-0.4 * (out["r_input_s"] - zero_point))
    out["flux_ratio"] = np.log10(flux_p / flux_s)
    out["r_input_p_scaled"] = -2.5 * np.log10(flux_p / aperture_rms) + zero_point
    out["r_input_s_scaled"] = -2.5 * np.log10(flux_s / aperture_rms) + zero_point

    re_sum = re_p + re_s
    distance = out["distance"].to_numpy(dtype=float)
    overlap = np.full(len(out), np.nan, dtype=float)
    valid = np.isfinite(re_sum) & (re_sum > 0) & np.isfinite(distance) & (distance > 0)
    overlap[valid] = np.log10(re_sum[valid] / distance[valid])
    out["log10_re_sum_over_distance"] = overlap
    out["log10_flux_s_over_flux_p"] = -0.4 * (
        out["r_input_s"].to_numpy(dtype=float)
        - out["r_input_p"].to_numpy(dtype=float)
    )
    return out


def _aperture_rms(conditions: Mapping[str, float]) -> float:
    missing = sorted(set(REQUIRED_CONDITIONS) - set(conditions))
    if missing:
        raise KeyError(f"observing conditions are missing: {missing}")
    beta = float(conditions["moffat_beta"])
    fwhm = float(conditions["psf_fwhm"])
    psf_size = fwhm * np.sqrt(
        (2 ** (1 / (beta - 1)) - 1) / (2 ** (1 / beta) - 1)
    ) / 2
    return (
        float(conditions["pixel_rms"])
        * (psf_size / float(conditions["pixel_size"])) ** 2
        * np.pi
    )


def _grouped_row_indices(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    group_column: Optional[str],
):
    if group_column is None:
        yield (
            np.arange(len(primary), dtype=np.int64),
            np.arange(len(secondary), dtype=np.int64),
        )
        return
    for name, frame in (("primary", primary), ("secondary", secondary)):
        if group_column not in frame.columns:
            raise KeyError(f"{name} catalogue lacks group column {group_column!r}")
    for group, primary_rows in primary.groupby(
        group_column, sort=False, dropna=False
    ).indices.items():
        if pd.isna(group):
            secondary_rows = np.flatnonzero(secondary[group_column].isna().to_numpy())
        else:
            secondary_rows = np.flatnonzero(secondary[group_column].eq(group).to_numpy())
        yield np.asarray(primary_rows, dtype=np.int64), secondary_rows.astype(np.int64)


def prepare_flow_inputs(
    primary_catalogue: Catalogue,
    secondary_catalogue: Optional[Catalogue] = None,
    *,
    conditions: Mapping[str, float],
    group_column: Optional[str] = None,
    near_radius_arcsec: float = DEFAULT_CROWDING_RADII_ARCSEC[0],
    far_radius_arcsec: float = DEFAULT_CROWDING_RADII_ARCSEC[1],
) -> pd.DataFrame:
    """Build the one-row-per-primary scene table used by the response flow.

    V3's ``nbr_flux_near``, ``nbr_flux_far``, and ``nbr_flux_max`` features are
    computed from every input neighbour, before the emulator's pair cuts are
    applied.  The near/far defaults reproduce the trained 0--3 and 3--7 arcsec
    shells and remain explicit keyword arguments for other flow definitions.
    """

    if near_radius_arcsec <= 0 or far_radius_arcsec <= near_radius_arcsec:
        raise ValueError("crowding radii must satisfy 0 < near < far")
    primary = validate_input_catalogue(primary_catalogue)
    secondary = (
        primary.copy()
        if secondary_catalogue is None
        else validate_input_catalogue(secondary_catalogue)
    )
    if "position_angle" not in primary.columns:
        raise KeyError("flow input catalogue is missing column: position_angle")
    aperture_rms = _aperture_rms(conditions)
    zero_point = float(conditions["zero_point"])
    secondary_flux = 10 ** (
        -0.4 * (secondary["r"].to_numpy(dtype=float) - zero_point)
    )

    near_flux = np.zeros(len(primary), dtype=float)
    far_flux = np.zeros(len(primary), dtype=float)
    max_flux = np.zeros(len(primary), dtype=float)
    nearest_row = np.full(len(primary), -1, dtype=np.int64)
    nearest_distance = np.full(len(primary), np.nan, dtype=float)
    pair_angle = np.full(len(primary), np.nan, dtype=float)
    far_radius_degrees = far_radius_arcsec / 3600.0

    for primary_rows, secondary_rows in _grouped_row_indices(
        primary, secondary, group_column
    ):
        if len(primary_rows) == 0 or len(secondary_rows) == 0:
            continue
        primary_positions = primary.loc[primary_rows, ["RA", "DEC"]].to_numpy(float)
        secondary_positions = secondary.loc[
            secondary_rows, ["RA", "DEC"]
        ].to_numpy(float)
        neighbours = KDTree(secondary_positions).query_ball_point(
            primary_positions,
            r=far_radius_degrees,
            workers=-1,
        )
        for local_primary, local_secondaries in enumerate(neighbours):
            if not local_secondaries:
                continue
            local_secondaries = np.asarray(local_secondaries, dtype=np.int64)
            delta = secondary_positions[local_secondaries] - primary_positions[local_primary]
            distance_arcsec = np.hypot(delta[:, 0], delta[:, 1]) * 3600.0
            valid = np.isfinite(distance_arcsec) & (distance_arcsec > 0)
            if not valid.any():
                continue
            local_secondaries = local_secondaries[valid]
            delta = delta[valid]
            distance_arcsec = distance_arcsec[valid]
            global_primary = primary_rows[local_primary]
            global_secondaries = secondary_rows[local_secondaries]
            flux = secondary_flux[global_secondaries]
            is_near = distance_arcsec < near_radius_arcsec
            near_flux[global_primary] = flux[is_near].sum()
            far_flux[global_primary] = flux[~is_near].sum()
            max_flux[global_primary] = flux.max(initial=0.0)
            nearest = int(np.argmin(distance_arcsec))
            nearest_row[global_primary] = global_secondaries[nearest]
            nearest_distance[global_primary] = distance_arcsec[nearest]
            pair_angle[global_primary] = np.arctan2(
                delta[nearest, 1], delta[nearest, 0]
            )

    flow = primary.add_suffix("_input_p")
    flow.insert(0, "primary_row", np.arange(len(primary), dtype=np.int64))
    if "index" in primary.columns:
        flow.insert(1, "input_index", primary["index"].to_numpy(copy=True))
    else:
        flow.insert(1, "input_index", np.arange(len(primary), dtype=np.int64))
    if group_column is not None:
        flow[group_column] = primary[group_column].to_numpy(copy=True)

    e1, e2 = ellipticity_from_axis_ratio_angle(
        primary["axis_ratio"].to_numpy(dtype=float),
        primary["position_angle"].to_numpy(dtype=float),
    )
    flow["e1_input_rot0_p"] = e1
    flow["e2_input_rot0_p"] = e2
    flow["secondary_row"] = nearest_row
    flow["distance"] = nearest_distance
    flow["polarization_angle"] = pair_angle
    flow["neighbored"] = nearest_row >= 0

    for column in secondary.columns:
        values = np.full(len(primary), np.nan, dtype=object)
        matched = nearest_row >= 0
        if matched.any():
            values[matched] = secondary.iloc[nearest_row[matched]][column].to_numpy()
        flow[f"{column}_input_s"] = values
    for column in secondary.select_dtypes(include=[np.number, "bool"]).columns:
        flow[f"{column}_input_s"] = pd.to_numeric(
            flow[f"{column}_input_s"], errors="coerce"
        )

    flow["nbr_flux_near"] = np.log10(1.0 + near_flux / aperture_rms)
    flow["nbr_flux_far"] = np.log10(1.0 + far_flux / aperture_rms)
    flow["nbr_flux_max"] = np.log10(1.0 + max_flux / aperture_rms)
    return flow.set_index("primary_row", drop=True)


def prepare_emulator_pairs(
    primary_catalogue: Catalogue,
    secondary_catalogue: Optional[Catalogue] = None,
    *,
    config: EmulatorPairingConfig,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Convert a user input catalogue to a model-ready response pair table."""

    pairs = make_pair_catalogue(
        primary_catalogue,
        secondary_catalogue,
        r_max_arcsec=config.r_max_arcsec,
        k=config.k,
        group_column=group_column,
    )
    pairs = select_response_pairs(pairs, config.cuts)
    if pairs.empty:
        raise ValueError("no primary-neighbour pairs remain after emulator training cuts")
    return rescale_emulator_pairs(pairs, config.conditions)


def prepare_forward_catalogue(
    primary_catalogue: Catalogue,
    secondary_catalogue: Optional[Catalogue] = None,
    *,
    config: EmulatorPairingConfig,
    group_column: Optional[str] = None,
    near_radius_arcsec: float = DEFAULT_CROWDING_RADII_ARCSEC[0],
    far_radius_arcsec: float = DEFAULT_CROWDING_RADII_ARCSEC[1],
) -> PreparedForwardCatalogue:
    """Prepare aligned flow and emulator inputs from one user catalogue."""

    primary = validate_input_catalogue(primary_catalogue)
    secondary = primary if secondary_catalogue is None else secondary_catalogue
    pairs = prepare_emulator_pairs(
        primary,
        secondary,
        config=config,
        group_column=group_column,
    )
    flow = prepare_flow_inputs(
        primary,
        secondary,
        conditions=config.conditions,
        group_column=group_column,
        near_radius_arcsec=near_radius_arcsec,
        far_radius_arcsec=far_radius_arcsec,
    )
    retained = np.sort(pairs["primary_row"].unique().astype(np.int64))
    flow = flow.loc[retained].copy()
    return PreparedForwardCatalogue(flow_inputs=flow, emulator_pairs=pairs)


__all__ = [
    "DEFAULT_CROWDING_RADII_ARCSEC",
    "EmulatorPairingConfig",
    "INPUT_CATALOGUE_COLUMNS",
    "PreparedForwardCatalogue",
    "find_neighbours",
    "make_pair_catalogue",
    "prepare_emulator_pairs",
    "prepare_flow_inputs",
    "prepare_forward_catalogue",
    "rescale_emulator_pairs",
    "select_response_pairs",
    "validate_input_catalogue",
]
