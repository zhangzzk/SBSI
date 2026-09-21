"""Map the prior atoms of one observation in measured space and in truth space.

Two panel rows, both showing the same atoms of the same observation:

* row 1 -- the flow-predicted measurement of every atom, centred on what was
  actually measured for the observation.  This is the space the proposal
  scores in, so it shows where the sampler looked.
* row 2 -- the same atoms on their true (input) properties.  This shows what
  the high-mass atoms physically are.

Marker encodes whether the production draw reached the atom usefully (``x``)
or not (``.``).  Colour encodes the exact posterior mass of the atom.  Mass is
only known for the atoms the exact 24m calculation retained; every other atom
is drawn in grey and carries, in total, the small remainder.

Diagnostic only.  Nothing here changes the target, the model, the cuts or the
prior; the proposal is built exactly as production builds it, and re-flooring
its predicted scatter with ``--floor-percentile`` changes no stored default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Production sampler configuration, read off scripts/run_disk_inference.py.
PRODUCTION_DELTA = 0.1
PRODUCTION_TEMPERATURE = 1.0
PRODUCTION_SEED = 8701
PRODUCTION_DRAWS = 16384
# pipeline_config.proposal.candidates in the run's result.json.  These atoms
# are summed exactly -- but they are NOT removed from the mixture the sampler
# draws from, so a draw landing inside the stratum is spent and contributes
# nothing (catalogue_null._exact_plus_complement).  Their share of the proposal
# is therefore the share of the draw budget that is thrown away.
PRODUCTION_CANDIDATES = 1024
CENTRE_NODE = 0

TARGET_ORDER = [
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_flux_radius",
    "measured_flux_from_mag_auto",
]

TRUTH_COLUMNS = (
    "e1_input_p",
    "e2_input_p",
    "r_input_p",
    "circularized_Re_input_p",
    "nbr_flux_near",
)


def shard_offsets(manifest: dict) -> np.ndarray:
    """Start row of every prior shard in the concatenated atom ordering."""

    counts = [int(s["n_rows"]) for s in manifest["shards"]]
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def gather_truth(manifest: dict, atoms: np.ndarray, columns=TRUTH_COLUMNS):
    """Read ``columns`` of ``flow_zero.parquet`` for the requested atom ids."""

    import pyarrow.parquet as pq

    atoms = np.asarray(atoms, dtype=np.int64)
    offsets = shard_offsets(manifest)
    if offsets[-1] != int(manifest["n_rows"]):
        raise ValueError("shard row counts do not sum to the manifest total")
    if atoms.size and (atoms.min() < 0 or atoms.max() >= offsets[-1]):
        raise ValueError("atom id outside the prior subset")
    out = np.full((atoms.size, len(columns)), np.nan, dtype=np.float64)
    shard_of = np.searchsorted(offsets, atoms, side="right") - 1
    for index, shard in enumerate(manifest["shards"]):
        take = np.flatnonzero(shard_of == index)
        if not take.size:
            continue
        local = atoms[take] - offsets[index]
        table = pq.read_table(
            Path(shard["root"]) / "flow_zero.parquet", columns=list(columns)
        )
        out[take] = np.column_stack(
            [np.asarray(table.column(c), dtype=np.float64)[local] for c in columns]
        )
    if not np.isfinite(out).all():
        raise ValueError("truth lookup left unfilled rows")
    return out


def observation_truth(manifest: dict, truth_row) -> dict:
    """True properties of the observed galaxy, in the atoms' own conventions.

    The mock truth table carries provenance only, so the physical properties
    come from the case input catalogue named in the image-mock manifest.  The
    prior's ``e1_input_p`` / ``circularized_Re_input_p`` are built from axis
    ratio and position angle exactly as ``e1_input_rot0`` and
    ``Re_input * sqrt(axis_ratio)`` are here, which the unit tests pin.

    The returned ellipticity is intrinsic, before the injected shear, matching
    the zero-shear ``flow_zero`` columns the truth panels plot.
    """

    import pandas as pd

    case = int(truth_row["source_case"])
    index = int(truth_row["source_input_index"])
    sources = {int(e["case"]): e for e in manifest["per_case"]}
    if case not in sources:
        raise ValueError(f"case {case} absent from the image-mock manifest")
    path = Path(sources[case]["sources"]["truth"]["path"])
    table = pd.read_feather(path)
    if index < 0 or index >= len(table):
        raise ValueError(f"source_input_index {index} outside {path.name}")
    record = table.iloc[index]
    if int(record["index_input"]) != index:
        raise ValueError("input catalogue is not indexed by source_input_index")
    ratio = float(record["axis_ratio_input"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"axis ratio outside (0, 1]: {ratio}")
    return dict(
        e1=float(record["e1_input_rot0"]),
        e2=float(record["e2_input_rot0"]),
        circularized_Re=float(record["Re_input"]) * np.sqrt(ratio),
        r=float(record["r_input"]),
        case=case,
        source_input_index=index,
        source_truth=str(path),
    )


def draw_classes(probability: np.ndarray, subset: np.ndarray, floor: float):
    """Split the draws by which half of the mixture actually paid for them.

    The proposal is ``q = delta * uniform + (1 - delta) * softmax(score)`` with
    ``delta = 0.1``: a flat 10% spread over the whole catalogue, and 90%
    directed by the proxy ranking.  Every atom's probability is a sum of those
    two contributions, ``floor = delta / n_atoms`` from the flat part and
    ``q - floor`` from the ranking, so each draw can be attributed to whichever
    contribution is larger.

    ``ranked`` is ``q > 2 * floor``, the point where the ranking supplies more
    of the atom's probability than the flat spread does.

    An earlier version cut at ``q > floor`` and called the result "ranked".
    That is the point where the softmax merely fails to underflow in float64,
    about 745 nats below the best atom, which nearly every atom clears; it says
    nothing about the ranking having chosen the atom.
    """

    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("defensive floor must be positive and finite")
    q = probability[subset]
    ranked = q > 2.0 * floor
    return ranked, ~ranked


def score_terms(values: np.ndarray, dispersion: np.ndarray,
                detection: np.ndarray, observed: np.ndarray,
                index: np.ndarray) -> dict:
    """The three additive parts of the proxy score, for named atoms.

    ``s_j = log(Pdet_j) - sum_d log sigma_jd - 0.5 sum_d z_jd^2``.  Splitting
    it says whether the proposal prefers an atom because it fits the
    observation (the quadratic term) or because its predicted scatter is
    narrow, which inflates a *density* without meaning the atom is a better
    explanation of the data.
    """

    index = np.asarray(index, dtype=np.int64)
    mu = np.asarray(values[index], dtype=np.float64)
    sigma = np.asarray(dispersion[index], dtype=np.float64)
    if np.any(sigma <= 0.0):
        raise ValueError("proxy dispersion must be positive")
    z = (np.asarray(observed, dtype=np.float64)[None, :] - mu) / sigma
    residual = np.asarray(observed, dtype=np.float64)[None, :] - mu
    with np.errstate(divide="ignore"):
        log_detection = np.log(np.asarray(detection[index], dtype=np.float64))
    log_dispersion = -np.log(sigma).sum(axis=1)
    quadratic = -0.5 * np.square(z).sum(axis=1)
    return {
        "log_detection": log_detection,
        "log_dispersion": log_dispersion,
        "quadratic": quadratic,
        "score": log_detection + log_dispersion + quadratic,
        "z": z,
        "sigma": sigma,
        "residual": residual,
    }


def summarise_terms(terms: dict, label: str) -> dict:
    """Median of each score part, plus the per-coordinate median |z|."""

    finite = np.isfinite(terms["score"])
    summary = {"group": label, "n": int(terms["score"].size),
               "n_finite": int(finite.sum())}
    if not finite.any():
        return summary
    for key in ("score", "log_detection", "log_dispersion", "quadratic"):
        summary[f"median_{key}"] = float(np.median(terms[key][finite]))
    summary["median_abs_z"] = [
        float(v) for v in np.median(np.abs(terms["z"][finite]), axis=0)
    ]
    # The raw scale behind the standardized miss: an atom can sit under one
    # sigma while being far away in the units of the plot, if its predicted
    # scatter is wide.  Reporting both separates "fits" from "is vague".
    summary["median_sigma"] = [
        float(v) for v in np.median(terms["sigma"][finite], axis=0)
    ]
    summary["median_abs_residual"] = [
        float(v) for v in np.median(np.abs(terms["residual"][finite]), axis=0)
    ]
    return summary


def exact_centre_node(record: dict, row: int):
    """Atom ids and exact posterior weights at the centre stencil node.

    The panel writes the row at the top level of every record; only the two
    rows the probe was pointed at by hand also repeat it under ``exact``.  Both
    are checked when present rather than trusting the file name.
    """

    if int(record["row"]) != int(row):
        raise ValueError(f"exact record is for row {record['row']}, not {row}")
    exact = record["exact"]
    if "row" in exact and int(exact["row"]) != int(row):
        raise ValueError("exact record disagrees with itself about the row")
    heavy = np.asarray(exact["top_atoms"][CENTRE_NODE], dtype=np.int64)
    mass = np.asarray(exact["top_posterior_weights"][CENTRE_NODE], dtype=np.float64)
    if heavy.shape != mass.shape:
        raise ValueError("exact atom ids and weights disagree in length")
    if np.unique(heavy).size != heavy.size:
        raise ValueError("exact atom ids repeat")
    if not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("exact posterior weights must be finite and non-negative")
    return heavy, mass


def zero_point_row(manifest: dict, n_atoms: int) -> int:
    """Row of ``probability.npy`` holding the detection term at zero shear.

    The cache is point-major, ``(n_points, n_atoms)``, and the proposal's base
    weight is the zero-shear detection probability -- the same row the
    comparison job uses.
    """

    points = [list(map(float, p)) for p in manifest["points"]]
    if [0.0, 0.0] not in points:
        raise ValueError("prepared cache has no zero-shear point")
    if int(manifest["n_rows"]) != n_atoms:
        raise ValueError("prepared cache and proposal disagree on atom count")
    return points.index([0.0, 0.0])


def production_draw(probability: np.ndarray, *, seed: int, object_id: int,
                    n_draws: int, exact_stratum: np.ndarray) -> dict:
    """Reproduce the draw production actually runs, for one observation.

    The configured estimator mode is ``tilted_stratified``, which reaches
    ``DefensiveLocalProposal.draw_tilted`` and thence
    ``WholeCatalogueProxy.draw_uniforms_batch``: one fixed-width uniform record
    per draw from a generator seeded by ``(seed, object_id)``, the second
    column used, mapped through the cumulative mixture by inverse-CDF.  That is
    sampling **with replacement**, and the mixture is *not* modified for the
    exact stratum -- a draw landing inside it simply contributes nothing,
    because that stratum is already summed exactly
    (``catalogue_null._exact_plus_complement``).

    An earlier version of this script reproduced ``select_priority_batch``
    instead, which is sampling without replacement and belongs to the
    ``priority_stratified`` mode nothing runs.  The proposal is the same in
    both, so where the mass sits was unaffected, but inclusion probabilities
    and Horvitz-Thompson weights are not quantities this estimator has.
    """

    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    if probability.ndim != 1 or not probability.size:
        raise ValueError("expected one proposal probability per atom")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(object_id)])
    )
    # Fixed-width records keep a smaller draw count an exact prefix of a
    # larger one, which is what makes the nested ladder free; production reads
    # the second column.
    uniforms = rng.random((int(n_draws), 3))[:, 1]
    cumulative = np.cumsum(probability)
    cumulative[-1] = 1.0
    drawn = np.searchsorted(cumulative, uniforms, side="right")
    drawn = np.clip(drawn, None, probability.size - 1).astype(np.int64)

    inside = np.zeros(probability.size, dtype=bool)
    inside[exact_stratum] = True
    wasted = inside[drawn]
    complement = drawn[~wasted]
    unique, multiplicity = np.unique(complement, return_counts=True)
    # With replacement there is no inclusion probability and no threshold; the
    # weight a drawn atom carries is 1 / (M q_j), and coverage -- the chance
    # the atom is seen at all -- is 1 - (1 - q_j)^M.
    coverage = -np.expm1(np.log1p(-np.clip(probability, 0.0, 1.0)) * n_draws)
    return dict(
        drawn=unique,
        multiplicity=multiplicity,
        all_draws=drawn,
        n_wasted=int(wasted.sum()),
        n_usable=int((~wasted).sum()),
        coverage=coverage,
        exact_mass=float(probability[exact_stratum].sum()),
    )


def build_proxy(values: np.ndarray, dispersion: np.ndarray, detection: np.ndarray):
    """Drive the production proxy class through the shim the compare job uses."""

    from sbsi.catalogue_sampling import WholeCatalogueProxy

    n_atoms = len(values)
    shim = SimpleNamespace(
        coordinates=SimpleNamespace(values=values, dispersion=dispersion),
        active_indices=np.arange(n_atoms, dtype=np.int64),
        local_base_weights=detection,
        prior_weights=np.full(n_atoms, 1.0 / n_atoms),
    )
    return WholeCatalogueProxy(shim, torch.device("cpu"), score_dtype=torch.float64)


def centred_measurements(block: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Predicted measurement relative to the observation, one row per atom.

    Shape and size are differenced; flux spans decades, so it is shown as a
    log ratio.  Column order matches ``TARGET_ORDER``.
    """

    block = np.asarray(block, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != 4:
        raise ValueError("expected four measured coordinates per atom")
    if observed[3] <= 0:
        raise ValueError("observed flux must be positive for a log ratio")
    return np.column_stack(
        [
            block[:, 0] - observed[0],
            block[:, 1] - observed[1],
            block[:, 2] - observed[2],
            np.log10(np.clip(block[:, 3], 1e-12, None) / observed[3]),
        ]
    )


def panel_limits(primary: np.ndarray, background: np.ndarray, *, span=99.0, pad=0.08):
    """Limits that keep the mass-carrying atoms visible against the bulk."""

    tail = (100.0 - span) / 2.0
    low = min(float(np.percentile(background, tail)), float(np.min(primary)))
    high = max(float(np.percentile(background, 100.0 - tail)), float(np.max(primary)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(background)), float(np.max(background))
        if high <= low:
            low, high = low - 0.5, high + 0.5
    margin = pad * (high - low)
    return low - margin, high + margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--exact-dir", type=Path, required=True)
    parser.add_argument("--row", type=int, default=142230)
    parser.add_argument("--background", type=int, default=60000)
    parser.add_argument("--background-seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-percentile", type=float, default=None,
                        help="re-floor the proposal's predicted scatter at "
                             "this percentile of the atoms' own scatter "
                             "before building the mixture; omit to use the "
                             "table as shipped (floored at the 1st percentile)")
    parser.add_argument("--fractional-floor", action="append", default=None,
                        metavar="TARGET",
                        help="treat this coordinate's scatter as "
                             "multiplicative when flooring, so the floor binds "
                             "equally at every brightness; repeatable")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    from sbsi.catalogue_closure import MockCatalogue
    from sbsi.catalogue_sampling import ProposalCoordinateTable
    from sbsi.disk_inference_store import file_hash, write_json

    prepared = args.run / "disk_assembled_v1"
    subset = args.run / "prior_subset24m"
    args.output.mkdir(parents=True)

    coords = ProposalCoordinateTable.load(prepared / "proposal")
    if list(coords.target_names) != TARGET_ORDER:
        raise ValueError(f"unexpected proposal target order: {coords.target_names}")
    n_atoms = len(coords.values)
    manifest = json.loads((subset / "manifest.json").read_text())
    if int(manifest["n_rows"]) != n_atoms:
        raise ValueError("proposal table and prior subset disagree on atom count")

    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    probability_row = zero_point_row(prepared_manifest, n_atoms)
    detection = np.load(prepared / "probability.npy", mmap_mode="r")
    if detection.ndim != 2 or detection.shape[1] != n_atoms:
        raise ValueError(f"detection probability shape {detection.shape}")
    detection = np.asarray(detection[probability_row], dtype=np.float64)

    mock = MockCatalogue.load(args.run / "input")
    observed = mock.measurements.iloc[args.row][TARGET_ORDER].to_numpy(dtype=np.float64)
    mock_manifest = json.loads((args.run / "input" / "image_mock_manifest.json").read_text())
    true_observation = observation_truth(mock_manifest, mock.truth.iloc[args.row])

    # The floor is applied after the per-atom summary, so raising it needs no
    # flow evaluation: the cached table can be re-floored in place.
    fractional_targets = tuple(args.fractional_floor or ())
    if args.floor_percentile is not None:
        coords = coords.with_dispersion_floor(
            percentile=args.floor_percentile,
            fractional_targets=fractional_targets,
        )
    elif fractional_targets:
        raise ValueError("--fractional-floor needs --floor-percentile")

    proxy = build_proxy(coords.values, coords.dispersion, detection)
    mixture = proxy.mixture(
        observed[None, :],
        delta=PRODUCTION_DELTA,
        temperature=PRODUCTION_TEMPERATURE,
    )[0]
    probability = mixture.numpy()
    # Production sums the top `candidates` atoms exactly.  It does not remove
    # them from the mixture it draws from: a draw that lands inside the stratum
    # simply contributes nothing, so the stratum's share of `q` is the share of
    # the budget that is thrown away.
    exact_stratum = np.argsort(probability)[::-1][:PRODUCTION_CANDIDATES]
    floor = PRODUCTION_DELTA / n_atoms
    uniform = 1.0 / n_atoms

    draw = production_draw(
        probability,
        seed=PRODUCTION_SEED,
        object_id=args.row,
        n_draws=PRODUCTION_DRAWS,
        exact_stratum=exact_stratum,
    )
    drawn = draw["drawn"]
    coverage = draw["coverage"]

    exact_files = sorted(args.exact_dir.glob(f"worker_*/row_{args.row}_exact.json"))
    if len(exact_files) != 1:
        raise ValueError(f"expected one exact record for row {args.row}")
    heavy, mass = exact_centre_node(
        json.loads(exact_files[0].read_text()), args.row
    )

    rng = np.random.default_rng(args.background_seed)
    background = rng.choice(n_atoms, size=args.background, replace=False)
    drawn_set = np.zeros(n_atoms, dtype=bool)
    drawn_set[drawn] = True
    exact_set = np.zeros(n_atoms, dtype=bool)
    exact_set[exact_stratum] = True
    # The exact stratum is neither raced nor grey bulk: it is the part of the
    # proposal the estimator sums with certainty, and on these rows it holds
    # most of the probability.  Earlier versions plotted only the raced atoms,
    # so the atoms the ranking likes best -- the ones expected to sit on the
    # observation -- were invisible, and the ranking looked worse than it is.
    background = background[~(drawn_set | exact_set)[background]]

    exact_shown = exact_stratum[np.argsort(probability[exact_stratum])[::-1]]
    everything = np.concatenate([heavy, drawn, exact_shown, background])
    truth = gather_truth(manifest, everything)
    # `n_unique` counts the distinct atoms the draw reached usefully; the
    # budget is PRODUCTION_DRAWS and the two are not interchangeable.
    n_heavy, n_unique, n_exact = heavy.size, drawn.size, exact_shown.size
    truth_heavy = truth[:n_heavy]
    truth_drawn = truth[n_heavy : n_heavy + n_unique]
    truth_exact = truth[n_heavy + n_unique : n_heavy + n_unique + n_exact]
    truth_background = truth[n_heavy + n_unique + n_exact :]

    measured_heavy = centred_measurements(coords.values[heavy], observed)
    measured_drawn = centred_measurements(coords.values[drawn], observed)
    measured_exact = centred_measurements(coords.values[exact_shown], observed)
    measured_background = centred_measurements(coords.values[background], observed)
    heavy_drawn = drawn_set[heavy]
    heavy_exact = exact_set[heavy]
    # A mass atom inside the exact stratum is summed with certainty; one that
    # is neither summed nor drawn is the only kind the estimator truly misses.
    heavy_reached = heavy_drawn | heavy_exact

    # Production draws from one mixture, so which half of it paid for a draw has
    # to be inferred from the atom's own probability.
    ranked, flat = draw_classes(probability, drawn, floor)

    # Where the proposal's own preference sits, independent of the draw.
    catalogue_ranked = np.flatnonzero(probability > 2.0 * floor)
    order = np.argsort(probability[catalogue_ranked])[::-1]
    catalogue_ranked = catalogue_ranked[order]
    top_scoring = catalogue_ranked[: min(64, catalogue_ranked.size)]
    flat_sample = rng.choice(drawn[flat], size=min(4096, int(flat.sum())),
                             replace=False) if flat.any() else np.empty(0, np.int64)
    groups = [
        ("top 64 by proposal probability", top_scoring),
        ("all atoms the ranking dominates", catalogue_ranked),
        ("mass-carrying atoms (exact, centre node)", heavy),
        ("drawn from ranked atoms", drawn[ranked]),
        ("drawn uniformly (sample)", flat_sample),
    ]
    decomposition = [
        summarise_terms(
            score_terms(coords.values, coords.dispersion, detection, observed, idx),
            label,
        )
        for label, idx in groups if idx.size
    ]

    # Does the ranking sit on the observation or beside it?  Rank bands answer
    # that without plotting 24 million atoms.  Band 1 is the exact stratum;
    # later bands are progressively deeper slices of the same ordering.  A
    # ranking that is centred has a mean near zero in every measured
    # coordinate for its leading band; one that is offset does not.
    rank_order = np.argsort(probability)[::-1]
    band_edges = [0, PRODUCTION_CANDIDATES, 8192, 65536, 524288, 4194304,
                  n_atoms]
    rank_bands = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        band = rank_order[lo:hi]
        block = centred_measurements(coords.values[band], observed)
        weight = probability[band]
        total = float(weight.sum())
        rank_bands.append(dict(
            rank_from=int(lo + 1),
            rank_to=int(hi),
            n_atoms=int(band.size),
            probability_mass=total,
            mean=[float(v) for v in block.mean(axis=0)],
            median=[float(v) for v in np.median(block, axis=0)],
            std=[float(v) for v in block.std(axis=0)],
            probability_weighted_mean=[
                float(v) for v in
                (block * weight[:, None]).sum(axis=0) / max(total, 1e-300)
            ],
        ))
        del block, weight

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    norm = LogNorm(vmin=max(float(mass.min()), 1e-6), vmax=float(mass.max()))

    # Every unsampled atom is the same small dot; only colour distinguishes an
    # atom whose mass we know from the grey bulk whose mass we never computed.
    unsampled_size = 26.0

    specs = [
        (
            axes[0][0],
            measured_background[:, [0, 1]],
            measured_drawn[:, [0, 1]],
            measured_exact[:, [0, 1]],
            measured_heavy[:, [0, 1]],
            r"predicted $g_1$ $-$ measured $g_1$",
            r"predicted $g_2$ $-$ measured $g_2$",
            "Panel 1a   measured space: shape",
            (0.0, 0.0),
        ),
        (
            axes[0][1],
            measured_background[:, [2, 3]],
            measured_drawn[:, [2, 3]],
            measured_exact[:, [2, 3]],
            measured_heavy[:, [2, 3]],
            r"predicted $R_{\rm flux}$ $-$ measured $R_{\rm flux}$   [pix]",
            r"$\log_{10}$(predicted flux / measured flux)",
            "Panel 1b   measured space: size and brightness",
            (0.0, 0.0),
        ),
        (
            axes[1][0],
            truth_background[:, [0, 1]],
            truth_drawn[:, [0, 1]],
            truth_exact[:, [0, 1]],
            truth_heavy[:, [0, 1]],
            r"true $e_1$ (intrinsic)",
            r"true $e_2$ (intrinsic)",
            "Panel 2a   truth space: intrinsic shape",
            (true_observation["e1"], true_observation["e2"]),
        ),
        (
            axes[1][1],
            truth_background[:, [3, 2]],
            truth_drawn[:, [3, 2]],
            truth_exact[:, [3, 2]],
            truth_heavy[:, [3, 2]],
            r"true circularized $R_e$   [arcsec]",
            r"true magnitude $r$",
            "Panel 2b   truth space: size and brightness",
            (true_observation["circularized_Re"], true_observation["r"]),
        ),
    ]

    handle = None
    for axis, bulk, sampled, summed, top, xlabel, ylabel, title, marker_at \
            in specs:
        # Unsampled bulk: mass unknown, so plain grey at the common size.
        axis.scatter(
            bulk[:, 0], bulk[:, 1], s=unsampled_size, c="0.86", marker=".",
            linewidths=0, alpha=0.45, rasterized=True, zorder=1,
        )
        # The draw, split into the part the proxy ranked and the part only the
        # defensive uniform component reached, at their true relative counts.
        # Both are kept translucent so the mass-carrying atoms stay readable
        # through what is, on these rows, a very crowded draw.
        axis.scatter(
            sampled[flat, 0], sampled[flat, 1], s=8.0,
            c="0.62", marker="x", linewidths=0.4, alpha=0.30,
            rasterized=True, zorder=2,
        )
        axis.scatter(
            sampled[ranked, 0], sampled[ranked, 1], s=30.0,
            c="tab:blue", marker="x", linewidths=1.0, alpha=0.65,
            rasterized=True, zorder=4,
        )
        # The part of the proposal that is summed with certainty.  It costs no
        # draw, so it is not a cross; it is what the ranking most prefers.
        axis.scatter(
            summed[:, 0], summed[:, 1], s=14.0,
            c="tab:orange", marker="o", linewidths=0.0, alpha=0.55,
            rasterized=True, zorder=3,
        )
        for mask, is_drawn in ((~heavy_reached, False), (heavy_reached, True)):
            if not mask.any():
                continue
            # Unsampled mass atoms keep the common dot size; a hairline edge
            # makes them findable without making them bigger.
            handle = axis.scatter(
                top[mask, 0], top[mask, 1], c=mass[mask], cmap="plasma", norm=norm,
                s=150.0 if is_drawn else unsampled_size,
                marker="x" if is_drawn else ".",
                linewidths=2.0 if is_drawn else 0.6,
                zorder=5 if is_drawn else 4,
                **({} if is_drawn else {"edgecolors": "black"}),
            )
        axis.axhline(marker_at[1], color="tab:red", lw=0.9, ls="--", zorder=6)
        axis.axvline(marker_at[0], color="tab:red", lw=0.9, ls="--", zorder=6)
        # The observation's own marker must stay inside the frame too.
        axis.set_xlim(*panel_limits(np.append(top[:, 0], marker_at[0]), bulk[:, 0]))
        axis.set_ylim(*panel_limits(np.append(top[:, 1], marker_at[1]), bulk[:, 1]))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=11)
    axes[1][1].invert_yaxis()

    subsample = n_atoms / max(background.size, 1)
    # Both shares are shares of probability, so the draw counts they buy vary
    # row by row.
    share = ("the flat 10% of probability", "the ranked 90% of probability")
    legend = [
        Line2D([], [], ls="", marker=".", color="0.86", ms=7,
               label=f"not drawn ({background.size:,} shown, 1 in {subsample:,.0f})"),
        Line2D([], [], ls="", marker="x", color="0.62", ms=6,
               label=f"drawn uniformly \u2014 {share[0]} "
                     f"({int(flat.sum()):,})"),
        Line2D([], [], ls="", marker="x", color="tab:blue", ms=8, mew=1.4,
               label=f"drawn from ranked atoms \u2014 {share[1]} "
                     f"({int(ranked.sum()):,})"),
        Line2D([], [], ls="", marker="o", color="tab:orange", ms=5,
               label=f"summed exactly "
                     f"({PRODUCTION_CANDIDATES:,} atoms, "
                     f"{draw['exact_mass']:.1%} of the proposal, "
                     f"{draw['n_wasted'] / PRODUCTION_DRAWS:.0%} of the "
                     f"draw budget wasted here)"),
        Line2D([], [], ls="", marker=".", color="black", ms=7,
               label="carries posterior mass, MISSED"),
        Line2D([], [], ls="", marker="x", color="black", ms=11, mew=2,
               label=f"carries posterior mass, reached "
                     f"({int(heavy_exact.sum())} summed exactly, "
                     f"{int(heavy_drawn.sum())} drawn)"),
        Line2D([], [], ls="--", color="tab:red", lw=0.9,
               label="the observation (measured above, true below)"),
    ]
    axes[0][0].legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)

    bar = fig.colorbar(handle, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("exact posterior mass of the atom")

    captured = float(mass.sum())
    reached = float(mass[heavy_reached].sum())
    sampler_label = (
        f"tilted_stratified: {PRODUCTION_DRAWS:,} draws with replacement, "
        f"{draw['n_usable']:,} usable on {n_unique:,} distinct atoms "
        f"({draw['n_wasted']:,} landed in the exact stratum and carry nothing)"
    )
    score_label = (
        "proposal scatter as shipped (floored at the 1st percentile)"
        if args.floor_percentile is None
        else f"proposal scatter re-floored at the "
             f"{args.floor_percentile:g}th percentile"
             + (f", fractional on {', '.join(fractional_targets)}"
                if fractional_targets else "")
    )
    fig.suptitle(
        f"Row {args.row}: measured mag "
        f"{mock.measurements.iloc[args.row]['measured_mag_auto']:.2f}"
        f"   |   {int(heavy_reached.sum())} of {n_heavy} mass-carrying atoms reached"
        f" ({int(heavy_exact.sum())} summed exactly + {int(heavy_drawn.sum())} drawn)"
        f"   |   {reached / captured:.1%} of the captured mass reached"
        f"\n{sampler_label}"
        f"\n{score_label}",
        fontsize=12,
    )
    suffix = ("" if args.floor_percentile is None
              else f"_floor{args.floor_percentile:g}")
    if fractional_targets:
        suffix += "_frac"
    figure_path = args.output / f"proposal_atom_map_row{args.row}{suffix}.png"
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    report = dict(
        status="diagnostic_only",
        row=args.row,
        n_atoms=int(n_atoms),
        observed={name: float(v) for name, v in zip(TARGET_ORDER, observed)},
        measured_mag_auto=float(mock.measurements.iloc[args.row]["measured_mag_auto"]),
        proposal=dict(
            delta=PRODUCTION_DELTA,
            temperature=PRODUCTION_TEMPERATURE,
            seed=PRODUCTION_SEED,
            n_draws=int(PRODUCTION_DRAWS),
            n_unique_usable_draws=int(n_unique),
            max_probability=float(probability.max()),
            top_atom=int(np.argmax(probability)),
            defensive_floor=float(floor),
            uniform_share=float(uniform),
            ranked_share_of_probability=1.0 - PRODUCTION_DELTA,
            flat_share_of_probability=PRODUCTION_DELTA,
            # Over the distinct reached atoms, not over the draws.
            n_unique_from_ranked=int(ranked.sum()),
            n_unique_uniform=int(flat.sum()),
            n_ranked_in_catalogue=int(catalogue_ranked.size),
            probability_mass_on_ranked=float(probability[catalogue_ranked].sum()),
            n_above_flat_share_in_catalogue=int((probability > uniform).sum()),
        ),
        dispersion_floor=dict(
            percentile=(None if args.floor_percentile is None
                        else float(args.floor_percentile)),
            fractional_targets=list(fractional_targets),
            shipped_percentile=1.0,
            note="the proposal divides each residual by the atom's own "
                 "predicted scatter, so a floor on that scatter sets how "
                 "tightly a vague atom may be judged; a fractional floor "
                 "ranks a coordinate on sigma/|x| so it binds at every "
                 "brightness",
            median_dispersion={
                name: float(v) for name, v in
                zip(TARGET_ORDER, np.median(coords.dispersion, axis=0))
            },
        ),
        rank_bands=dict(
            columns=list(TARGET_ORDER[:3]) + ["log10_flux_ratio"],
            note="predicted minus measured; flux as a log10 ratio",
            bands=rank_bands,
        ),
        exact_stratum=dict(
            n_atoms=int(PRODUCTION_CANDIDATES),
            probability_mass=float(probability[exact_stratum].sum()),
            ranked_mass_removed=float(
                probability[exact_stratum].sum()
                - PRODUCTION_CANDIDATES * floor
            ),
            min_probability=float(probability[exact_stratum].min()),
            n_mass_atoms_inside=int(heavy_exact.sum()),
            mass_inside=float(mass[heavy_exact].sum()),
        ),
        sampler=dict(
            mode="tilted_stratified",
            replacement=True,
            note="draw_tilted -> draw_uniforms_batch: inverse-CDF from the "
                 "mixture, with replacement.  The exact stratum is not removed "
                 "from the mixture, so a draw landing inside it contributes "
                 "nothing (catalogue_null._exact_plus_complement)",
            n_draws=int(PRODUCTION_DRAWS),
            n_wasted_in_exact_stratum=int(draw["n_wasted"]),
            n_usable=int(draw["n_usable"]),
            wasted_fraction=float(draw["n_wasted"] / PRODUCTION_DRAWS),
            n_unique_usable=int(n_unique),
            worst_weight_on_a_drawn_mass_atom=(
                float((1.0 / np.maximum(
                    PRODUCTION_DRAWS
                    * probability[heavy][heavy_drawn & ~heavy_exact],
                    1.0e-300)).max())
                if (heavy_drawn & ~heavy_exact).any() else 0.0
            ),
        ),
        score_decomposition=decomposition,
        true_observation=true_observation,
        exact=dict(
            node=CENTRE_NODE,
            n_atoms=int(n_heavy),
            captured_mass=captured,
            n_drawn=int(heavy_drawn.sum()),
            n_summed_exactly=int(heavy_exact.sum()),
            n_reached=int(heavy_reached.sum()),
            reached_mass=reached,
            atoms=[
                dict(
                    atom=int(a),
                    mass=float(w),
                    proposal_probability=float(probability[a]),
                    # With replacement there is no inclusion probability.  The
                    # comparable quantity is coverage: the chance this atom is
                    # drawn at least once in the whole budget.
                    coverage=float(coverage[a]),
                    drawn=bool(d),
                    summed_exactly=bool(e),
                )
                for a, w, d, e in zip(heavy, mass, heavy_drawn, heavy_exact)
            ],
        ),
        background=dict(n_shown=int(background.size), seed=args.background_seed),
        figure=dict(path=figure_path.name, sha256=file_hash(figure_path)),
        source=dict(
            proposal_coordinates=file_hash(prepared / "proposal" / "coordinates.npz"),
            exact_record=file_hash(exact_files[0]),
            prior_manifest=file_hash(subset / "manifest.json"),
        ),
        note=(
            "Reproduces the configured tilted_stratified draw: the exact "
            "stratum is summed with certainty but is NOT removed from the "
            "mixture, so draws landing inside it are simply discarded.  The "
            "drawn set reported here is the usable complement.  Diagnostic "
            "only; re-flooring the proposal here changes no production default."
        ),
    )
    write_json(args.output / "report.json", report)
    print(f"FIGURE {figure_path}", flush=True)
    print(
        f"REACHED {int(heavy_reached.sum())}/{n_heavy} captured={captured:.4f} "
        f"reached={reached:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
