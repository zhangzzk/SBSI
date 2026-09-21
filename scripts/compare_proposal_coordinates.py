#!/usr/bin/env python
"""Re-rank known-important atoms under alternative proposal coordinates.

Diagnostic only.  This changes nothing about the target: the proposal ``q``
is the only object varied here, every likelihood contribution still carries
the exact ``pi_j / q_j`` correction, so each variant is an equally valid
(differently precise) estimator.  No model, cut, prior weight or production
setting is touched, and no new flow evaluation is performed -- the cached
proposal coordinate table and the exact 24m atom weights already on disk
supply everything.

For each observation in the frozen worst-32 exact panel this reports, per
variant, how much proposal mass reaches the atoms that actually carry the
exact centre-node posterior, and the importance-sampling second moment those
atoms contribute.

The ``prod_4d`` variant must reproduce the recorded production proposal
probabilities; that reproduction is asserted, not assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

# Production epsilon for the defensive prior component (q = delta*pi + (1-delta)*softmax).
PRODUCTION_DELTA = 0.1
# Draw budget the current production estimator uses per observation.
PRODUCTION_DRAWS = 16384
# Index of the centre node within the nine-node stencil (centre,+x,-x,+y,-y,++,+-,-+,--).
CENTRE_NODE = 0


def variant_dispersion(dispersion: np.ndarray, inflation: dict[int, float]) -> np.ndarray:
    """Scale selected coordinate dispersions, leaving the rest untouched."""

    scaled = np.array(dispersion, dtype=np.float64, copy=True)
    for column, factor in inflation.items():
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("dispersion inflation must be finite and positive")
        scaled[:, column] *= float(factor)
    return scaled


def coverage_metrics(
    weights: np.ndarray,
    probabilities: np.ndarray,
    *,
    floor: float,
    n_draws: int = PRODUCTION_DRAWS,
) -> dict:
    """Summarize how a proposal covers atoms of known exact posterior weight.

    ``weights`` are exact centre-node posterior weights over all 24m atoms,
    restricted to the retained top atoms, so they sum to the capture fraction
    rather than to one.  ``second_moment`` is therefore a lower bound on the
    full ``sum_j w_j^2 / q_j``, and ``ess_upper_bound`` an upper bound on the
    achievable effective sample size.  Neither is a production acceptance
    number.
    """

    weights = np.asarray(weights, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if weights.shape != probabilities.shape or weights.ndim != 1:
        raise ValueError("aligned one-dimensional weights and probabilities required")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("exact posterior weights must be finite and non-negative")
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("proposal probabilities must be finite and positive")
    second_moment = float(np.sum(np.square(weights) / probabilities))
    at_floor = probabilities <= floor * (1.0 + 1.0e-9)
    return {
        "capture": float(weights.sum()),
        "proposal_mass_on_top_atoms": float(probabilities.sum()),
        "max_proposal_probability": float(probabilities.max()),
        "min_proposal_probability": float(probabilities.min()),
        "n_at_defensive_floor": int(at_floor.sum()),
        "n_atoms": int(len(weights)),
        "expected_hits_top_atoms": float(n_draws * probabilities.sum()),
        "expected_hits_excluding_largest": float(
            n_draws * (probabilities.sum() - probabilities.max())
        ),
        "second_moment": second_moment,
        # None rather than inf: the report is written with allow_nan=False.
        "ess_upper_bound": float(n_draws / second_moment) if second_moment > 0 else None,
    }


FOUR_D = (0, 1, 2, 3)
SIZE_FLUX_2D = (2, 3)


def component(columns, temperature=1.0, weight=1.0, inflation=None) -> dict:
    """One local proposal component: coordinates, dispersion, temperature."""

    return {
        "columns": tuple(columns),
        "temperature": float(temperature),
        "weight": float(weight),
        "inflation": dict(inflation or {}),
    }


def build_variants() -> list[dict]:
    """Proposal variants compared here.  All are proposal-only changes.

    A variant is a weighted mixture of local components, combined with the
    production defensive prior as
    ``q = delta*pi + (1-delta) * sum_c weight_c * softmax(score_c / T_c)``.
    The single-component case reproduces the production proposal exactly.
    """

    return [
        {
            "name": "prod_4d",
            "components": [component(FOUR_D)],
            "note": "production: g1,g2,flux_radius,flux_from_mag_auto at T=1",
        },
        {
            "name": "rf_2d",
            "components": [component(SIZE_FLUX_2D)],
            "note": "owner proposal: rank on size and brightness only",
        },
        {
            "name": "prod_4d_T2",
            "components": [component(FOUR_D, temperature=2.0)],
            "note": "production coordinates, tempered softmax",
        },
        {
            "name": "prod_4d_T3",
            "components": [component(FOUR_D, temperature=3.0)],
            "note": "production coordinates, tempered further",
        },
        {
            "name": "prod_4d_T4",
            "components": [component(FOUR_D, temperature=4.0)],
            "note": "production coordinates, more strongly tempered",
        },
        {
            "name": "rf_2d_T2",
            "components": [component(SIZE_FLUX_2D, temperature=2.0)],
            "note": "size and brightness only, tempered",
        },
        {
            "name": "mix_4d_rf2d",
            "components": [
                component(FOUR_D, weight=0.5),
                component(SIZE_FLUX_2D, weight=0.5),
            ],
            "note": "equal mixture of the production and two-dimensional proposals",
        },
        {
            "name": "mix_4d_rf2d_T2",
            "components": [
                component(FOUR_D, temperature=2.0, weight=0.5),
                component(SIZE_FLUX_2D, temperature=2.0, weight=0.5),
            ],
            "note": "equal mixture, both components tempered",
        },
        {
            "name": "shape_sigma_x3",
            "components": [component(FOUR_D, inflation={0: 3.0, 1: 3.0})],
            "note": "keep shape coordinates but widen their Gaussian threefold",
        },
    ]


def check_variants(variants: list[dict]) -> None:
    """Component weights must form a convex combination."""

    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    for variant in variants:
        if not variant["components"]:
            raise ValueError(f"{variant['name']}: at least one component required")
        total = sum(c["weight"] for c in variant["components"])
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"{variant['name']}: component weights sum to {total}, not one"
            )
        for item in variant["components"]:
            if item["weight"] <= 0 or item["temperature"] <= 0:
                raise ValueError(f"{variant['name']}: positive weight and temperature required")


def local_component(proxy, observations: np.ndarray, temperature: float):
    """Undo the defensive blend to recover ``softmax(score / temperature)``.

    ``mixture`` is the only place the production proxy forms a proposal, so
    driving it and inverting the affine defensive blend keeps this diagnostic
    on exactly the production score path rather than a parallel copy of it.
    """

    mixture = proxy.mixture(
        observations, delta=PRODUCTION_DELTA, temperature=temperature
    )
    mixture -= PRODUCTION_DELTA * proxy.prior.unsqueeze(0)
    mixture /= 1.0 - PRODUCTION_DELTA
    return mixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--exact-dir", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing comparison")
    args.output.mkdir(parents=True)

    from sbsi.catalogue_closure import MockCatalogue
    from sbsi.catalogue_sampling import ProposalCoordinateTable, WholeCatalogueProxy
    from sbsi.disk_inference_store import file_hash, write_json

    started = time.monotonic()
    prepared_root = args.run / "disk_assembled_v1"
    prepared = json.loads((prepared_root / "manifest.json").read_text())
    if prepared["status"] != "complete" or prepared["pilot_only"]:
        raise ValueError("complete production preparation required")
    for name, expected in prepared["output_sha256"].items():
        if file_hash(prepared_root / name) != expected:
            raise ValueError(f"prepared hash mismatch: {name}")

    # The proxy path is imported from the shared checkout; pin the two modules
    # whose behaviour this diagnostic depends on to what the exact panel ran.
    import sbsi.catalogue_sampling as sampling_module
    import sbsi.disk_inference_store as store_module

    implementation = {
        "sbsi/catalogue_sampling.py": file_hash(Path(sampling_module.__file__)),
        "sbsi/disk_inference_store.py": file_hash(Path(store_module.__file__)),
    }

    coords = ProposalCoordinateTable.load(prepared_root / "proposal")
    n_atoms = len(coords.values)
    prior_weights = np.full(n_atoms, 1.0 / n_atoms, dtype=np.float64)
    probability = np.load(prepared_root / "probability.npy", mmap_mode="r")
    points = [tuple(point) for point in prepared["points"]]
    if (0.0, 0.0) not in points:
        raise ValueError("zero-shear detection probabilities required")
    local_base_weights = np.asarray(
        probability[points.index((0.0, 0.0))], dtype=np.float64
    )
    if local_base_weights.shape != (n_atoms,):
        raise ValueError("detection probabilities misaligned with proposal coordinates")
    active_indices = np.flatnonzero(prior_weights > 0).astype(np.int64)
    floor = PRODUCTION_DELTA / n_atoms

    mock = MockCatalogue.load(args.run / "input")
    target_names = list(coords.target_names)
    if target_names != [
        "measured_ngmix_g1",
        "measured_ngmix_g2",
        "measured_flux_radius",
        "measured_flux_from_mag_auto",
    ]:
        raise ValueError(f"unexpected proposal target order: {target_names}")

    exact_paths = sorted(args.exact_dir.glob("worker_*/row_*_exact.json"))
    if not exact_paths:
        raise ValueError("no exact panel rows found")
    panel = []
    for path in exact_paths:
        payload = json.loads(path.read_text())
        exact = payload["exact"]
        row = int(payload["row"])
        atoms = np.asarray(exact["top_atoms"][CENTRE_NODE], dtype=np.int64)
        weights = np.asarray(
            exact["top_posterior_weights"][CENTRE_NODE], dtype=np.float64
        )
        if atoms.shape != weights.shape:
            raise ValueError(f"row {row}: atom/weight mismatch")
        panel.append(
            {
                "row": row,
                "atoms": atoms,
                "weights": weights,
                "source": str(path),
                "source_sha256": file_hash(path),
            }
        )
    rows = np.array([entry["row"] for entry in panel], dtype=np.int64)
    observed = mock.measurements.iloc[rows][target_names].to_numpy(dtype=np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("observed proposal coordinates must be finite")

    variants = build_variants()
    check_variants(variants)

    proxy_cache: dict[tuple, object] = {}

    def get_proxy(columns: tuple, inflation: dict):
        key = (columns, tuple(sorted(inflation.items())))
        if key not in proxy_cache:
            dispersion = variant_dispersion(coords.dispersion, inflation)
            shim = SimpleNamespace(
                coordinates=SimpleNamespace(
                    values=np.ascontiguousarray(coords.values[:, list(columns)]),
                    dispersion=np.ascontiguousarray(dispersion[:, list(columns)]),
                ),
                active_indices=active_indices,
                local_base_weights=local_base_weights,
                prior_weights=prior_weights,
            )
            proxy_cache[key] = WholeCatalogueProxy(
                shim, torch.device("cpu"), score_dtype=torch.float64
            )
        return proxy_cache[key]

    results: dict[str, dict] = {}
    per_atom_detail: dict[str, dict] = {}
    prior_row = torch.as_tensor(prior_weights, dtype=torch.float64)
    for variant in variants:
        variant_rows = {}
        for start in range(0, len(panel), args.row_chunk):
            stop = min(start + args.row_chunk, len(panel))
            block = None
            for item in variant["components"]:
                columns = item["columns"]
                proxy = get_proxy(columns, item["inflation"])
                local = local_component(
                    proxy,
                    observed[start:stop][:, list(columns)],
                    item["temperature"],
                )
                local *= item["weight"]
                block = local if block is None else block + local
                del local
            # Restore the production defensive prior around the mixed local part.
            block *= 1.0 - PRODUCTION_DELTA
            block += PRODUCTION_DELTA * prior_row.unsqueeze(0)
            for offset in range(stop - start):
                entry = panel[start + offset]
                q = block[offset].numpy()[entry["atoms"]]
                variant_rows[entry["row"]] = coverage_metrics(
                    entry["weights"], q, floor=floor
                )
                if entry["row"] == 142230:
                    per_atom_detail[variant["name"]] = {
                        "atoms": entry["atoms"].tolist(),
                        "proposal_probability": q.tolist(),
                        "exact_posterior_weight": entry["weights"].tolist(),
                    }
            del block
        results[variant["name"]] = {
            "definition": {
                "components": [
                    {
                        "target_names": [target_names[c] for c in item["columns"]],
                        "temperature": item["temperature"],
                        "weight": item["weight"],
                        "dispersion_inflation": {
                            target_names[int(c)]: float(f)
                            for c, f in item["inflation"].items()
                        },
                    }
                    for item in variant["components"]
                ],
                "delta": PRODUCTION_DELTA,
                "note": variant["note"],
            },
            "rows": variant_rows,
        }
        print(f"VARIANT_COMPLETE {variant['name']} seconds={time.monotonic()-started:.1f}", flush=True)

    # Hard reproduction check against the recorded production proposal.
    retrieval = json.loads(args.retrieval.read_text())
    recorded_row = int(retrieval["row"])
    recorded = {
        int(item["atom"]): float(item["proposal_probability"])
        for item in retrieval["atoms"]
    }
    detail = per_atom_detail.get("prod_4d")
    if detail is None or recorded_row != 142230:
        raise ValueError("production reproduction check requires row 142230 detail")
    reproduced = dict(zip(detail["atoms"], detail["proposal_probability"]))
    missing = sorted(set(recorded) - set(reproduced))
    common = sorted(set(recorded) & set(reproduced))
    if not common:
        raise ValueError("no shared atoms between recorded and reproduced proposals")
    recorded_values = np.array([recorded[a] for a in common])
    reproduced_values = np.array([reproduced[a] for a in common])
    relative = np.abs(reproduced_values - recorded_values) / recorded_values
    reproduction = {
        "row": recorded_row,
        "n_compared": len(common),
        "n_recorded_not_reproduced": len(missing),
        "max_relative_difference": float(relative.max()),
        "median_relative_difference": float(np.median(relative)),
    }
    print("PRODUCTION_REPRODUCTION " + json.dumps(reproduction), flush=True)
    if relative.max() > 1.0e-6:
        raise ValueError(
            "prod_4d does not reproduce the recorded production proposal "
            f"(max relative difference {relative.max():.3e})"
        )

    report = {
        "status": "diagnostic_only",
        "production_changes": False,
        "selection": (
            "frozen worst-32 exact curvature panel; selected by production "
            "negative-curvature contribution, not random or held out"
        ),
        "delta": PRODUCTION_DELTA,
        "n_draws": PRODUCTION_DRAWS,
        "centre_node": CENTRE_NODE,
        "n_atoms": int(n_atoms),
        "defensive_floor": floor,
        "implementation_sha256": implementation,
        "preparation_sha256": file_hash(prepared_root / "manifest.json"),
        "retrieval_sha256": file_hash(args.retrieval),
        "coordinates_sha256": file_hash(prepared_root / "proposal" / "coordinates.npz"),
        "panel": [
            {k: v for k, v in entry.items() if k not in ("atoms", "weights")}
            for entry in panel
        ],
        "production_reproduction": reproduction,
        "variants": results,
        "row_142230_detail": per_atom_detail,
        "seconds": time.monotonic() - started,
    }
    write_json(args.output / "report.json", report)

    # Compact stdout summary: pooled over the panel, per variant.
    print()
    print(
        f"{'variant':>18} {'median ESS<=':>13} {'median q(top)':>14} "
        f"{'median q_max':>13} {'mean #floor':>12}"
    )
    for name, payload in results.items():
        ess = np.array([r["ess_upper_bound"] for r in payload["rows"].values()])
        qtop = np.array([r["proposal_mass_on_top_atoms"] for r in payload["rows"].values()])
        qmax = np.array([r["max_proposal_probability"] for r in payload["rows"].values()])
        nfloor = np.array([r["n_at_defensive_floor"] for r in payload["rows"].values()])
        print(
            f"{name:>18} {np.median(ess):>13.1f} {np.median(qtop):>14.4f} "
            f"{np.median(qmax):>13.4f} {nfloor.mean():>12.2f}"
        )
    print(f"\nCOMPARISON_COMPLETE seconds={time.monotonic()-started:.1f}")


if __name__ == "__main__":
    main()
