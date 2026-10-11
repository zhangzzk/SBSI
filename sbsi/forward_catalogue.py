"""Prepare user input catalogues for forward response inference.

Image simulation and measurement remain BlendEMU responsibilities.  This
module owns the lightweight inference-time transformation from a user galaxy
catalogue to both model views needed for response prediction:

* one row per primary with the flow and classifier inputs;
* one row per primary-neighbour pair for the blending emulator.

The pair search and scaling are adapted from BlendEMU's
``utils.kdt_neighbor_finder``, ``nz_utils.make_reg_features``, and
``data_utils.rescale`` so SBSI does not delegate catalogue semantics to an
opaque model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .catalogue import Catalogue, load_catalogue
from .coordinates import ellipticity_from_axis_ratio_angle
from .crowding import V39_FEATURES, physical_close_crowding


INPUT_CATALOGUE_COLUMNS = (
    "RA",
    "DEC",
    "r",
    "Re",
    "sersic_n",
    "axis_ratio",
    "position_angle",
)
PAIR_INDEX_COLUMNS = ("primary_row", "secondary_row")
REQUIRED_CONDITIONS = (
    "pixel_size",
    "zero_point",
    "psf_fwhm",
    "moffat_beta",
    "pixel_rms",
)


@dataclass(frozen=True)
class EmulatorPairingConfig:
    """Training-matched neighbour search and observing conditions for one emulator."""

    r_max_arcsec: float
    k: int
    conditions: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "conditions", dict(self.conditions))
        if self.r_max_arcsec <= 0:
            raise ValueError("r_max_arcsec must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        missing = sorted(set(REQUIRED_CONDITIONS) - set(self.conditions))
        if missing:
            raise KeyError(f"observing conditions are missing: {missing}")

    @classmethod
    def from_emulator(cls, emulator) -> "EmulatorPairingConfig":
        """Recover the pairing and observing settings the emulator was loaded with."""

        missing = [name for name in ("r_max_arcsec", "k", "conditions") if not hasattr(emulator, name)]
        if missing:
            raise TypeError(f"emulator does not record its pairing settings: {missing}")
        return cls(
            r_max_arcsec=float(emulator.r_max_arcsec),
            k=int(emulator.k),
            conditions=emulator.conditions,
        )


@dataclass(frozen=True)
class PreparedForwardCatalogue:
    """Aligned object- and pair-level views derived from one input catalogue.

    ``flow_inputs`` has one row per retained primary and is indexed by the
    original ``primary_row``.  ``emulator_pairs`` may have several rows per
    primary, each also carrying the primary's flow inputs.  Keeping the views
    separate prevents the flow response from being accidentally weighted by
    neighbour multiplicity.
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
    out["r_input_p_scaled"] = -2.5 * np.log10(flux_p / aperture_rms) + zero_point
    out["r_input_s_scaled"] = -2.5 * np.log10(flux_s / aperture_rms) + zero_point
    return out


def prepare_flow_inputs(
    primary_catalogue: Catalogue,
    *,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Build the one-row-per-primary table of flow and classifier inputs.

    Adds the eight ``V39_FEATURES`` to the primary's catalogue columns: the
    intrinsic ellipticity, the circularized half-light radius, and the
    flux-weighted crowding of every neighbour within 10 arcsec at 0.5, 1 and
    2 arcsec scales.  Each ``group_column`` value is one independent scene.
    The eight inputs are stored at float32 precision, as in training.
    """

    primary = validate_input_catalogue(primary_catalogue)
    if group_column is None:
        scenes = [np.arange(len(primary), dtype=np.int64)]
    else:
        if group_column not in primary.columns:
            raise KeyError(f"primary catalogue lacks group column {group_column!r}")
        scenes = primary.groupby(group_column, sort=False, dropna=False).indices.values()
    crowding = np.empty((len(primary), 3))
    for rows in scenes:
        crowding[rows] = physical_close_crowding(primary.iloc[rows])

    flow = primary.add_suffix("_input_p")
    flow.insert(0, "primary_row", np.arange(len(primary), dtype=np.int64))
    if "index" in primary.columns:
        flow.insert(1, "input_index", primary["index"].to_numpy(copy=True))
    else:
        flow.insert(1, "input_index", np.arange(len(primary), dtype=np.int64))
    if group_column is not None:
        flow[group_column] = primary[group_column].to_numpy(copy=True)

    q = primary["axis_ratio"].to_numpy(dtype=float)
    flow["e1_input_p"], flow["e2_input_p"] = ellipticity_from_axis_ratio_angle(
        q, primary["position_angle"].to_numpy(dtype=float)
    )
    flow["circularized_Re_input_p"] = primary["Re"].to_numpy(dtype=float) * np.sqrt(q)
    flow[["nbr_close_050", "nbr_close_100", "nbr_close_200"]] = crowding
    features = list(V39_FEATURES)
    flow[features] = flow[features].astype(np.float32).astype(np.float64)
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
    if pairs.empty:
        raise ValueError("no primary has a neighbour within the emulator's pair radius")
    return rescale_emulator_pairs(pairs, config.conditions)


def prepare_forward_catalogue(
    primary_catalogue: Catalogue,
    *,
    config: EmulatorPairingConfig,
    group_column: Optional[str] = None,
) -> PreparedForwardCatalogue:
    """Prepare aligned flow and emulator inputs from one user catalogue.

    Primaries without any neighbour within the emulator's pair radius are
    dropped, as in the pair table.
    """

    primary = validate_input_catalogue(primary_catalogue)
    flow = prepare_flow_inputs(primary, group_column=group_column)
    pairs = prepare_emulator_pairs(primary, config=config, group_column=group_column)
    # The blending emulator acts on draws of each primary's measurement, so
    # every pair row also carries the primary's flow inputs.
    features = list(V39_FEATURES)
    pairs = pairs.drop(columns=[c for c in features if c in pairs.columns])
    pairs = pairs.join(flow[features], on="primary_row")
    retained = np.sort(pairs["primary_row"].unique().astype(np.int64))
    flow = flow.loc[retained].copy()
    return PreparedForwardCatalogue(flow_inputs=flow, emulator_pairs=pairs)


__all__ = [
    "EmulatorPairingConfig",
    "INPUT_CATALOGUE_COLUMNS",
    "PreparedForwardCatalogue",
    "find_neighbours",
    "make_pair_catalogue",
    "prepare_emulator_pairs",
    "prepare_flow_inputs",
    "prepare_forward_catalogue",
    "rescale_emulator_pairs",
    "validate_input_catalogue",
]
