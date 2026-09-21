"""Map the prior atoms of one observation in measured space and in truth space.

Two panel rows, both showing the same atoms of the same observation:

* row 1 -- the flow-predicted measurement of every atom, centred on what was
  actually measured for the observation.  This is the space the proposal
  scores in, so it shows where the sampler looked.
* row 2 -- the same atoms on their true (input) properties.  This shows what
  the high-mass atoms physically are.

Marker encodes whether the production priority race drew the atom (``x``) or
not (``.``).  Colour encodes the exact posterior mass of the atom.  Mass is
only known for the atoms the exact 24m calculation retained; every other atom
is drawn in grey and carries, in total, the small remainder.

Diagnostic only.  Nothing here changes the target, the model, the cuts or the
prior; the proposal is evaluated exactly as production builds it.
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


def priority_race(mixture: torch.Tensor, *, seed: int, object_id: int, n_select: int):
    """Reproduce the production without-replacement draw for one observation.

    Mirrors ``DefensiveLocalProposal.select_priority_batch``: one uniform per
    atom from a generator seeded by ``(seed, object_id)``, key ``q_j / u_j``,
    keep the ``n_select`` largest.  No exact stratum is excluded here, so this
    is the race as the proposal alone would run it.
    """

    from sbsi.catalogue_sampling import _priority_row_seed

    if n_select <= 0 or n_select + 1 > mixture.numel():
        raise ValueError("priority sampling needs more atoms than draws")
    generator = torch.Generator(device=mixture.device)
    generator.manual_seed(_priority_row_seed(int(seed), int(object_id)))
    tiny = float(np.finfo(np.float64).tiny)
    uniform = torch.rand(
        mixture.numel(),
        generator=generator,
        device=mixture.device,
        dtype=torch.float64,
    ).clamp_min(tiny)
    top = torch.topk(mixture / uniform, n_select + 1, sorted=True)
    threshold = top.values[n_select]
    if not torch.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError(f"priority threshold not positive finite: {threshold}")
    return top.indices[:n_select].cpu().numpy().astype(np.int64)


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

    proxy = build_proxy(coords.values, coords.dispersion, detection)
    mixture = proxy.mixture(
        observed[None, :],
        delta=PRODUCTION_DELTA,
        temperature=PRODUCTION_TEMPERATURE,
    )[0]
    drawn = priority_race(
        mixture,
        seed=PRODUCTION_SEED,
        object_id=args.row,
        n_select=PRODUCTION_DRAWS,
    )
    probability = mixture.numpy()

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
    background = background[~drawn_set[background]]

    everything = np.concatenate([heavy, drawn, background])
    truth = gather_truth(manifest, everything)
    n_heavy, n_drawn = heavy.size, drawn.size
    truth_heavy = truth[:n_heavy]
    truth_drawn = truth[n_heavy : n_heavy + n_drawn]
    truth_background = truth[n_heavy + n_drawn :]

    measured_heavy = centred_measurements(coords.values[heavy], observed)
    measured_drawn = centred_measurements(coords.values[drawn], observed)
    measured_background = centred_measurements(coords.values[background], observed)
    heavy_drawn = drawn_set[heavy]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    norm = LogNorm(vmin=max(float(mass.min()), 1e-6), vmax=float(mass.max()))

    specs = [
        (
            axes[0][0],
            measured_background[:, [0, 1]],
            measured_drawn[:, [0, 1]],
            measured_heavy[:, [0, 1]],
            r"predicted $g_1$ $-$ measured $g_1$",
            r"predicted $g_2$ $-$ measured $g_2$",
            "Panel 1a   measured space: shape",
            True,
        ),
        (
            axes[0][1],
            measured_background[:, [2, 3]],
            measured_drawn[:, [2, 3]],
            measured_heavy[:, [2, 3]],
            r"predicted $R_{\rm flux}$ $-$ measured $R_{\rm flux}$   [pix]",
            r"$\log_{10}$(predicted flux / measured flux)",
            "Panel 1b   measured space: size and brightness",
            True,
        ),
        (
            axes[1][0],
            truth_background[:, [0, 1]],
            truth_drawn[:, [0, 1]],
            truth_heavy[:, [0, 1]],
            r"true $e_1$ (intrinsic)",
            r"true $e_2$ (intrinsic)",
            "Panel 2a   truth space: intrinsic shape",
            False,
        ),
        (
            axes[1][1],
            truth_background[:, [3, 2]],
            truth_drawn[:, [3, 2]],
            truth_heavy[:, [3, 2]],
            r"true circularized $R_e$   [arcsec]",
            r"true magnitude $r$",
            "Panel 2b   truth space: size and brightness",
            False,
        ),
    ]

    handle = None
    for axis, bulk, sampled, top, xlabel, ylabel, title, centred in specs:
        axis.scatter(
            bulk[:, 0], bulk[:, 1], s=1.0, c="0.82", marker=".",
            linewidths=0, rasterized=True, zorder=1,
        )
        axis.scatter(
            sampled[:, 0], sampled[:, 1], s=9.0, c="0.35", marker="x",
            linewidths=0.5, rasterized=True, zorder=2,
        )
        for mask, marker in ((~heavy_drawn, "."), (heavy_drawn, "x")):
            if not mask.any():
                continue
            is_drawn = marker == "x"
            # 'x' is an unfilled marker, so it takes no edgecolor at all.
            style = {} if is_drawn else {"edgecolors": "black"}
            handle = axis.scatter(
                top[mask, 0], top[mask, 1], c=mass[mask], cmap="plasma", norm=norm,
                s=150.0 if is_drawn else 190.0, marker=marker,
                linewidths=2.0 if is_drawn else 0.8,
                zorder=4 if is_drawn else 3,
                **style,
            )
        if centred:
            axis.axhline(0.0, color="tab:red", lw=0.8, ls="--", zorder=5)
            axis.axvline(0.0, color="tab:red", lw=0.8, ls="--", zorder=5)
        axis.set_xlim(*panel_limits(top[:, 0], bulk[:, 0]))
        axis.set_ylim(*panel_limits(top[:, 1], bulk[:, 1]))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=11)
    axes[1][1].invert_yaxis()

    legend = [
        Line2D([], [], ls="", marker=".", color="0.82", ms=9,
               label=f"prior atom, not drawn ({background.size:,} of {n_atoms:,} shown)"),
        Line2D([], [], ls="", marker="x", color="0.35", ms=7,
               label=f"drawn by the proposal ({n_drawn:,} draws)"),
        Line2D([], [], ls="", marker=".", color="black", mfc="none", ms=12,
               label="carries posterior mass, NOT drawn"),
        Line2D([], [], ls="", marker="x", color="black", ms=11, mew=2,
               label="carries posterior mass, drawn"),
    ]
    axes[0][0].legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)

    bar = fig.colorbar(handle, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("exact posterior mass of the atom")

    captured = float(mass.sum())
    reached = float(mass[heavy_drawn].sum())
    fig.suptitle(
        f"Row {args.row}: measured mag "
        f"{mock.measurements.iloc[args.row]['measured_mag_auto']:.2f}"
        f"   |   {int(heavy_drawn.sum())} of {n_heavy} mass-carrying atoms drawn"
        f"   |   {reached / captured:.1%} of the captured mass reached by the draw",
        fontsize=13,
    )
    figure_path = args.output / f"proposal_atom_map_row{args.row}.png"
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
            n_draws=int(n_drawn),
            n_unique_drawn=int(np.unique(drawn).size),
            max_probability=float(probability.max()),
            top_atom=int(np.argmax(probability)),
            defensive_floor=float(PRODUCTION_DELTA / n_atoms),
        ),
        exact=dict(
            node=CENTRE_NODE,
            n_atoms=int(n_heavy),
            captured_mass=captured,
            n_drawn=int(heavy_drawn.sum()),
            drawn_mass=reached,
            atoms=[
                dict(
                    atom=int(a),
                    mass=float(w),
                    proposal_probability=float(probability[a]),
                    expected_draws=float(probability[a] * PRODUCTION_DRAWS),
                    drawn=bool(d),
                )
                for a, w, d in zip(heavy, mass, heavy_drawn)
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
            "Priority race reproduced from the production mixture without "
            "excluding the tilted-stratified exact stratum, so the drawn set is "
            "the proposal's own race, not the estimator's final bookkeeping."
        ),
    )
    write_json(args.output / "report.json", report)
    print(f"FIGURE {figure_path}", flush=True)
    print(
        f"DRAWN {int(heavy_drawn.sum())}/{n_heavy} captured={captured:.4f} "
        f"reached={reached:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
