"""Repeat the reliable legacy Gaussian-toy grids with isolated component scenes.

The renderer, ngmix measurement, noise seeds, shear amplitude, and configuration
grids match ``archive/toy_blend_decompose.py`` and
``archive/toy_close_pair_scan.py``.  In addition to replaying their contextual
decomposition, this runner measures every neighbour from an isolated
primary--neighbour pair and measures the primary both with and without the
other galaxies present.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import galsim
import numpy as np
import pandas as pd

from archive.toy_blend_linearity import PIX, _PSF, measure


def galaxy(flux: float, x: float = 0.0, y: float = 0.0, hlr: float = 0.4) -> dict:
    """Construct the circular Gaussian dictionary used by the legacy toys."""
    return {
        "hlr": float(hlr),
        "flux": float(flux),
        "e1": 0.0,
        "e2": 0.0,
        "x": float(x),
        "y": float(y),
    }


def neighbour(distance: float, angle_deg: float, flux: float, hlr: float = 0.4) -> dict:
    angle = np.deg2rad(float(angle_deg))
    return galaxy(
        flux=flux,
        x=float(distance) * np.cos(angle),
        y=float(distance) * np.sin(angle),
        hlr=hlr,
    )


def sweep_configs() -> list[dict]:
    """The exact 3 flux regimes x 4 layouts in toy_decomp_sweep.out."""
    layouts = [
        ("1 nbr @1.2\"", [(1.2, 40.0)]),
        ("2 nbrs @1.2\"", [(1.2, 40.0), (1.2, 200.0)]),
        (
            "4 nbrs @1.2\"",
            [(1.2, angle) for angle in (20.0, 110.0, 200.0, 290.0)],
        ),
        (
            "4 nbrs mixed dist 1-2.5\"",
            list(zip((1.0, 1.5, 2.0, 2.5), (20.0, 110.0, 200.0, 290.0))),
        ),
    ]
    regimes = ((800.0, "faint"), (2000.0, "brighter"), (500.0, "very_faint"))
    configs = []
    for flux, regime in regimes:
        for layout, positions in layouts:
            configs.append({
                "suite": "sweep",
                "group": regime,
                "label": f"{regime}: {layout}",
                "target": galaxy(flux),
                "neighbours": [neighbour(d, a, flux) for d, a in positions],
                "legacy_nreal": 300,
            })
    return _number_configs(configs)


def closepair_configs() -> list[dict]:
    """The exact 26 configurations in the legacy close-pair scan."""
    configs = []
    for distance in (0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0):
        configs.append({
            "suite": "closepair",
            "group": "equal_flux_pair",
            "label": f'equal-flux pair @ {distance:.1f}\"',
            "target": galaxy(1000.0),
            "neighbours": [neighbour(distance, 40.0, 1000.0)],
            "legacy_nreal": 400,
        })
    for distance in (0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0):
        configs.append({
            "suite": "closepair",
            "group": "faint_neighbour_pair",
            "label": f'faint-neighbour pair @ {distance:.1f}\"',
            "target": galaxy(1000.0),
            "neighbours": [neighbour(distance, 40.0, 250.0)],
            "legacy_nreal": 400,
        })
    for distance in (0.5, 0.7, 1.0, 1.5):
        configs.append({
            "suite": "closepair",
            "group": "faint_target_pair",
            "label": f'faint-target pair @ {distance:.1f}\"',
            "target": galaxy(500.0),
            "neighbours": [neighbour(distance, 40.0, 500.0)],
            "legacy_nreal": 400,
        })
    for count in (1, 2, 4, 8):
        angles = np.linspace(0.0, 360.0, count, endpoint=False)
        configs.append({
            "suite": "closepair",
            "group": "faint_crowding",
            "label": f'{count} faint neighbours @ 1.0\"',
            "target": galaxy(1000.0),
            "neighbours": [neighbour(1.0, angle, 250.0) for angle in angles],
            "legacy_nreal": 400,
        })
    for count in (1, 2, 4, 8):
        angles = np.linspace(0.0, 360.0, count, endpoint=False)
        configs.append({
            "suite": "closepair",
            "group": "very_faint_close_crowding",
            "label": f'{count} very-faint neighbours @ 0.8\"',
            "target": galaxy(1000.0),
            "neighbours": [neighbour(0.8, angle, 120.0) for angle in angles],
            "legacy_nreal": 400,
        })
    return _number_configs(configs)


def _number_configs(configs: list[dict]) -> list[dict]:
    for config_id, config in enumerate(configs):
        config["config_id"] = int(config_id)
    return configs


def get_configs(suite: str) -> list[dict]:
    if suite == "sweep":
        return sweep_configs()
    if suite == "closepair":
        return closepair_configs()
    raise ValueError(f"unknown suite {suite}")


def flag(source: dict, sheared: bool) -> dict:
    result = dict(source)
    result["shear"] = bool(sheared)
    return result


def render_legacy(
    sources: list[dict],
    g1: float,
    g2: float,
    stamp: int = 96,
    shear_pos: bool = True,
) -> np.ndarray:
    """Render byte-for-byte like the corrected legacy decomposition toy."""
    image = galsim.ImageF(int(stamp), int(stamp), scale=PIX)
    for source in sources:
        obj = galsim.Gaussian(
            half_light_radius=source["hlr"], flux=source["flux"]
        )
        if source.get("e1", 0.0) or source.get("e2", 0.0):
            obj = obj.shear(
                g1=source.get("e1", 0.0), g2=source.get("e2", 0.0)
            )
        x, y = source["x"], source["y"]
        if source.get("shear", True):
            obj = obj.shear(g1=g1, g2=g2)
            if shear_pos:
                x, y = x * (1.0 + g1) + y * g2, y * (1.0 - g1) + x * g2
        galsim.Convolve([obj, _PSF]).drawImage(
            image=image,
            add_to_image=True,
            offset=(x / PIX, y / PIX),
        )
    return image.array


def response_draws(
    sources: list[dict],
    g: float,
    sky: float,
    nreal: int,
    stamp: int = 96,
) -> np.ndarray:
    """Legacy g1 response with its exact common-noise and fitter seeds."""
    plus = render_legacy(sources, +g, 0.0, stamp=stamp)
    minus = render_legacy(sources, -g, 0.0, stamp=stamp)
    values = []
    for realization in range(int(nreal)):
        noise = np.random.RandomState(realization).normal(0.0, sky, plus.shape)
        eplus = measure(plus + noise, realization)
        eminus = measure(minus + noise, realization)
        values.append((eplus[0] - eminus[0]) / (2.0 * g))
    return np.asarray(values, dtype=float)


def assemble_draws(
    self_context: np.ndarray,
    neighbour_context: list[np.ndarray],
    self_isolated: np.ndarray,
    neighbour_pairs: list[np.ndarray],
    full: np.ndarray,
) -> pd.DataFrame:
    """Assemble the three exact closures from matched response vectors."""
    arrays = [self_context, self_isolated, full, *neighbour_context, *neighbour_pairs]
    lengths = {len(np.asarray(value)) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("all matched response vectors must have the same length")
    context_sum = np.sum(np.stack(neighbour_context), axis=0)
    pair_sum = np.sum(np.stack(neighbour_pairs), axis=0)
    frame = pd.DataFrame({
        "realization": np.arange(len(full), dtype=int),
        "R_self_context": self_context,
        "R_neighbour_context_sum": context_sum,
        "R_self_isolated": self_isolated,
        "R_neighbour_pair_sum": pair_sum,
        "R_full": full,
    })
    for index, values in enumerate(neighbour_context):
        frame[f"R_neighbour_context_{index}"] = values
    for index, values in enumerate(neighbour_pairs):
        frame[f"R_neighbour_pair_{index}"] = values
    frame["R_linear_context"] = (
        frame.R_self_context + frame.R_neighbour_context_sum
    )
    frame["R_linear_pair_contextself"] = (
        frame.R_self_context + frame.R_neighbour_pair_sum
    )
    frame["R_linear_isolated"] = (
        frame.R_self_isolated + frame.R_neighbour_pair_sum
    )
    frame["excess_context"] = frame.R_full - frame.R_linear_context
    frame["excess_pair_contextself"] = (
        frame.R_full - frame.R_linear_pair_contextself
    )
    frame["excess_isolated"] = frame.R_full - frame.R_linear_isolated
    frame["delta_neighbour_context_pair"] = (
        frame.R_neighbour_context_sum - frame.R_neighbour_pair_sum
    )
    frame["delta_self_context_isolated"] = (
        frame.R_self_context - frame.R_self_isolated
    )
    return frame


def vector_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sem_matched": float(values.std(ddof=0) / np.sqrt(len(values))),
    }


def run_config(
    config: dict,
    g: float,
    sky: float,
    nreal: int,
    stamp: int,
) -> tuple[pd.DataFrame, dict]:
    target = config["target"]
    neighbours = config["neighbours"]
    full_sources = [target, *neighbours]
    self_context = response_draws(
        [flag(target, True), *[flag(item, False) for item in neighbours]],
        g, sky, nreal, stamp,
    )
    neighbour_context = []
    for active in range(len(neighbours)):
        neighbour_context.append(response_draws(
            [flag(target, False), *[
                flag(item, index == active)
                for index, item in enumerate(neighbours)
            ]],
            g, sky, nreal, stamp,
        ))
    self_isolated = response_draws([flag(target, True)], g, sky, nreal, stamp)
    neighbour_pairs = [
        response_draws([flag(target, False), flag(item, True)], g, sky, nreal, stamp)
        for item in neighbours
    ]
    full = response_draws(
        [flag(item, True) for item in full_sources], g, sky, nreal, stamp
    )
    frame = assemble_draws(
        self_context, neighbour_context, self_isolated, neighbour_pairs, full
    )
    frame.insert(0, "n_neighbours", len(neighbours))
    frame.insert(0, "label", config["label"])
    frame.insert(0, "group", config["group"])
    frame.insert(0, "config_id", config["config_id"])
    frame.insert(0, "suite", config["suite"])

    response_columns = [
        "R_self_context", "R_neighbour_context_sum", "R_self_isolated",
        "R_neighbour_pair_sum", "R_full", "R_linear_context",
        "R_linear_pair_contextself", "R_linear_isolated", "excess_context",
        "excess_pair_contextself", "excess_isolated",
        "delta_neighbour_context_pair", "delta_self_context_isolated",
    ]
    summaries = {
        column: vector_summary(frame[column].to_numpy(float))
        for column in response_columns
    }
    sem = lambda values: float(np.asarray(values).std(ddof=0) / np.sqrt(nreal))
    context_component_sem = [sem(self_context), *map(sem, neighbour_context)]
    pair_contextself_component_sem = [sem(self_context), *map(sem, neighbour_pairs)]
    isolated_component_sem = [sem(self_isolated), *map(sem, neighbour_pairs)]
    summaries["excess_context"]["sem_legacy_quadrature"] = float(np.sqrt(
        sem(full) ** 2 + np.sum(np.square(context_component_sem))
    ))
    summaries["excess_pair_contextself"]["sem_legacy_quadrature"] = float(np.sqrt(
        sem(full) ** 2 + np.sum(np.square(pair_contextself_component_sem))
    ))
    summaries["excess_isolated"]["sem_legacy_quadrature"] = float(np.sqrt(
        sem(full) ** 2 + np.sum(np.square(isolated_component_sem))
    ))
    audit = {
        "suite": config["suite"],
        "config_id": int(config["config_id"]),
        "group": config["group"],
        "label": config["label"],
        "n_neighbours": int(len(neighbours)),
        "n_galaxies_full_scene": int(len(neighbours) + 1),
        "nreal": int(nreal),
        "legacy_nreal": int(config["legacy_nreal"]),
        "g": float(g),
        "sky": float(sky),
        "stamp": int(stamp),
        "pixel_scale": float(PIX),
        "primary": target,
        "neighbours": neighbours,
        "same_noise_and_fit_seed_all_counterfactuals": True,
        "positions_sheared_for_active_sources": True,
        "profiles": "circular Gaussian",
        "summaries": summaries,
        "max_identity_replay_abs": float(max(
            np.abs(
                frame.excess_pair_contextself
                - frame.excess_context
                - frame.delta_neighbour_context_pair
            ).max(),
            np.abs(
                frame.excess_isolated
                - frame.excess_pair_contextself
                - frame.delta_self_context_isolated
            ).max(),
        )),
    }
    return frame, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("sweep", "closepair"), required=True)
    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--nreal", type=int)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--sky", type=float, default=10.0)
    parser.add_argument("--stamp", type=int, default=96)
    parser.add_argument("--output-draws", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    for output in (args.output_draws, args.output_json):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")
    configs = get_configs(args.suite)
    if not 0 <= args.config_id < len(configs):
        raise IndexError(f"config id {args.config_id} outside {len(configs)} configs")
    config = configs[args.config_id]
    nreal = int(args.nreal or config["legacy_nreal"])
    if nreal < 1 or args.g <= 0 or args.sky <= 0:
        raise ValueError("nreal, g, and sky must be positive")
    frame, audit = run_config(config, args.g, args.sky, nreal, args.stamp)
    if audit["max_identity_replay_abs"] > 1.0e-12:
        raise RuntimeError(f"decomposition identity failed: {audit}")
    Path(args.output_draws).parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(args.output_draws)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "suite": audit["suite"],
        "config_id": audit["config_id"],
        "label": audit["label"],
        "nreal": audit["nreal"],
        "n_neighbours": audit["n_neighbours"],
        "excess_context": audit["summaries"]["excess_context"],
        "excess_pair_contextself": audit["summaries"]["excess_pair_contextself"],
        "excess_isolated": audit["summaries"]["excess_isolated"],
        "max_identity_replay_abs": audit["max_identity_replay_abs"],
    }, indent=2, sort_keys=True))
    print("LEGACY_TOY_NO_CONTEXT_CONFIG_DONE", flush=True)


if __name__ == "__main__":
    main()
