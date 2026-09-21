#!/usr/bin/env python3
"""Estimate the V3.5-like response change for FLUX_RADIUS >= 0.5 arcsec.

The expensive 16 x 64 QMC integration is already cached for the magnitude-only
and 0.75-arcsec selections.  Only one raw flow draw per object was retained.
Use that draw to estimate the *paired* response change from magnitude-only to a
radius cut, then anchor the change to the cached magnitude-only response.  The
same construction at 0.75 arcsec is an empirical validation against its exact
cached QMC result.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from sbsi.selection_model import load_selection_model


CASES = tuple(range(120, 140))
MAGNITUDE_CUT = 2
MAGNITUDE_RADIUS_CUT = 6
MAG_LIMIT = 25.8
RADII_PIXELS = (2.5, 3.75)
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPLICATES = 10_000
BASE_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "coherent_u_crowd_ensemble_v1"
)
SELECTION_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "detection_circularized_repair_v1/selection_replication_c90_139_v1"
)
MOCK_ROOT = SELECTION_ROOT / "mock"
CANDIDATE_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_coherent_u_flow_rblend_ablation_v2"
)
MODEL_PATHS = tuple(
    CANDIDATE_ROOT / f"models/flow8_rblend/seed{seed}/selected.pt"
    for seed in (20260913, 20260914, 20260915)
)
MODEL_HASHES = (
    "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
    "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
    "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
)


def digest(path: str | Path) -> str:
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preserving existing result: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def actual_statistics(base, data: dict, radius_pixels: float | None) -> np.ndarray:
    measured = data["values"]
    keep = data["U"].astype(bool) & (measured[..., 2] < MAG_LIMIT)
    if radius_pixels is not None:
        keep &= measured[..., 3] >= np.log(radius_pixels)
    return np.asarray(
        [
            base.sufficient(keep, data["y"]),
            base.sufficient(keep, data["es"]),
            base.sufficient(keep, data["y"] - data["es"]),
        ]
    )


def raw_statistics(base, parent: np.ndarray, raw: np.ndarray, es: np.ndarray,
                   radius_pixels: float | None) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = 30.0 - 2.5 * np.log10(raw[..., 3])
    keep = magnitude < MAG_LIMIT
    if radius_pixels is not None:
        keep &= raw[..., 2] >= radius_pixels
    q = keep.astype(np.float64)
    u = q[..., None] * raw[..., :2]
    return base.model_statistics(parent, q, u, es)[:3]


def response_entry(base, actual_r: np.ndarray, model_r: np.ndarray) -> dict:
    denominator = model_r[:, 0, 0]
    m = 100.0 * (actual_r[:, 0, 0] / denominator - 1.0)
    result = {
        "actual_R11": base.describe(actual_r[:, 0, 0]),
        "model_R11": base.describe(denominator),
        "m_percent": base.describe(m),
        "component_contribution_to_m_pp": {},
    }
    for index, name in ((1, "truth"), (2, "residual")):
        delta = model_r[:, index, 0] - actual_r[:, index, 0]
        result["component_contribution_to_m_pp"][name] = base.describe(
            -100.0 * delta / denominator
        )
    return result


def run(output: Path, device: str) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the response screen must run through Slurm")
    if tuple(digest(path) for path in MODEL_PATHS) != MODEL_HASHES:
        raise RuntimeError("V3.5-like classifier hash mismatch")

    base = load_module(BASE_ROOT / "worker.py", "radius050_cached_base")
    config = base.verify()
    candidate = load_module(
        Path(__file__).with_name("train_minimal_coherent_u_rblend_ablation.py"),
        "radius050_candidate_source",
    )
    protocol = json.loads((CANDIDATE_ROOT / "protocol.json").read_text())
    members = [load_selection_model(path, device=device) for path in MODEL_PATHS]

    actual = {"magnitude": [], "radius_2.5": [], "radius_3.75": []}
    exact = {
        "classifier": {"magnitude": [], "radius_3.75": []},
        "oracle": {"magnitude": [], "radius_3.75": []},
    }
    raw_stats = {
        parent: {name: [] for name in actual}
        for parent in ("classifier", "oracle")
    }
    counts = []

    for case in CASES:
        data, _, _, moments, _ = base.source_case(config, case, with_moments=True)
        ids, frames, usable, _ = candidate.verified_case(protocol, case)
        np.testing.assert_array_equal(ids, data["ids"])
        np.testing.assert_array_equal(usable, data["U"])
        probabilities = np.mean(
            np.stack(
                [
                    np.stack(
                        [member.predict_proba(frame, batch_size=65_536) for frame in frames]
                    )
                    for member in members
                ]
            ),
            axis=0,
        )

        mock_path = MOCK_ROOT / "parts" / f"case{case:03d}" / "mock_replica000.npz"
        completed_path = mock_path.with_name("completed.json")
        completed = json.loads(completed_path.read_text())
        if completed["hashes"].get(str(mock_path)) != digest(mock_path):
            raise RuntimeError(f"retained raw-flow hash mismatch: case={case}")
        with np.load(mock_path) as stored:
            np.testing.assert_array_equal(stored["ids"], ids)
            raw = stored["measurement"].astype(np.float64)

        actual_case = {
            "magnitude": actual_statistics(base, data, None),
            "radius_2.5": actual_statistics(base, data, 2.5),
            "radius_3.75": actual_statistics(base, data, 3.75),
        }
        np.testing.assert_array_equal(
            data["U"].astype(bool) & data["C"][MAGNITUDE_CUT].astype(bool),
            data["U"].astype(bool) & (data["values"][..., 2] < MAG_LIMIT),
        )
        np.testing.assert_array_equal(
            data["U"].astype(bool) & data["C"][MAGNITUDE_RADIUS_CUT].astype(bool),
            data["U"].astype(bool)
            & (data["values"][..., 2] < MAG_LIMIT)
            & (data["values"][..., 3] >= np.log(3.75)),
        )
        for name, value in actual_case.items():
            actual[name].append(value)

        for parent_name, parent in (("classifier", probabilities), ("oracle", data["U"])):
            exact[parent_name]["magnitude"].append(
                base.model_statistics(
                    parent,
                    moments["q"][MAGNITUDE_CUT],
                    moments["u"][MAGNITUDE_CUT],
                    data["es"],
                )[:3]
            )
            exact[parent_name]["radius_3.75"].append(
                base.model_statistics(
                    parent,
                    moments["q"][MAGNITUDE_RADIUS_CUT],
                    moments["u"][MAGNITUDE_RADIUS_CUT],
                    data["es"],
                )[:3]
            )
            raw_stats[parent_name]["magnitude"].append(
                raw_statistics(base, parent, raw, data["es"], None)
            )
            raw_stats[parent_name]["radius_2.5"].append(
                raw_statistics(base, parent, raw, data["es"], 2.5)
            )
            raw_stats[parent_name]["radius_3.75"].append(
                raw_statistics(base, parent, raw, data["es"], 3.75)
            )

        n_mag = actual_case["magnitude"][0, :, 0]
        counts.append(
            {
                "case": case,
                "usable_magnitude": n_mag.tolist(),
                "retained_radius_2.5": (
                    actual_case["radius_2.5"][0, :, 0] / n_mag
                ).tolist(),
                "retained_radius_3.75": (
                    actual_case["radius_3.75"][0, :, 0] / n_mag
                ).tolist(),
            }
        )
        print(f"RADIUS050_EXISTING_CASE case={case}", flush=True)

    actual = {name: np.asarray(value) for name, value in actual.items()}
    exact = {
        parent: {name: np.asarray(value) for name, value in values.items()}
        for parent, values in exact.items()
    }
    raw_stats = {
        parent: {name: np.asarray(value) for name, value in values.items()}
        for parent, values in raw_stats.items()
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.vstack(
        [
            np.ones((1, len(CASES)), dtype=int),
            rng.multinomial(
                len(CASES),
                np.full(len(CASES), 1.0 / len(CASES)),
                BOOTSTRAP_REPLICATES,
            ),
        ]
    )
    actual_r = {name: base.responses(value, weights) for name, value in actual.items()}
    result = {"classifier": {}, "oracle": {}}
    validation = {}
    for parent in result:
        exact_r = {
            name: base.responses(value, weights) for name, value in exact[parent].items()
        }
        raw_r = {
            name: base.responses(value, weights) for name, value in raw_stats[parent].items()
        }
        anchored = {
            name: exact_r["magnitude"] + raw_r[name] - raw_r["magnitude"]
            for name in ("radius_2.5", "radius_3.75")
        }
        result[parent]["magnitude_only_exact_qmc"] = response_entry(
            base, actual_r["magnitude"], exact_r["magnitude"]
        )
        result[parent]["radius_0.5_arcsec_anchored_one_draw"] = response_entry(
            base, actual_r["radius_2.5"], anchored["radius_2.5"]
        )
        result[parent]["radius_0.75_arcsec_exact_qmc"] = response_entry(
            base, actual_r["radius_3.75"], exact_r["radius_3.75"]
        )
        result[parent]["radius_0.75_arcsec_anchored_one_draw"] = response_entry(
            base, actual_r["radius_3.75"], anchored["radius_3.75"]
        )
        validation[parent] = {
            "anchored_minus_exact_R11_at_0.75": base.describe(
                anchored["radius_3.75"][:, 0, 0]
                - exact_r["radius_3.75"][:, 0, 0]
            ),
            "anchored_minus_exact_m_pp_at_0.75": float(
                result[parent]["radius_0.75_arcsec_anchored_one_draw"]["m_percent"]["value"]
                - result[parent]["radius_0.75_arcsec_exact_qmc"]["m_percent"]["value"]
            ),
        }

    total_mag = np.sum([row["usable_magnitude"] for row in counts], axis=0)
    total_25 = np.sum(
        [np.asarray(row["usable_magnitude"]) * row["retained_radius_2.5"] for row in counts],
        axis=0,
    )
    total_375 = np.sum(
        [np.asarray(row["usable_magnitude"]) * row["retained_radius_3.75"] for row in counts],
        axis=0,
    )
    payload = {
        "analysis": "V3.5-like measured-radius sensitivity from existing catalogues",
        "screen_only_not_promotion": True,
        "cases": list(CASES),
        "population": "18<r<25.8, 0.37<Re<1.5; MAG_AUTO<25.8; no measured-|e| cut",
        "pixel_scale_arcsec": 0.2,
        "result": result,
        "one_draw_anchor_validation": validation,
        "actual_retention_after_usability_and_magnitude": {
            "radius_0.5_arcsec": (total_25 / total_mag).tolist(),
            "radius_0.75_arcsec": (total_375 / total_mag).tolist(),
            "leg_counts_before_radius": total_mag.astype(int).tolist(),
        },
        "method": {
            "exact_cached_qmc": "mean of 16 seeds x 64 joint-flow draws, with CRN between shear legs",
            "radius_0.5": "paired one-retained-draw response change from magnitude-only, anchored to exact magnitude-only QMC",
            "validation": "apply the identical anchored estimator at 0.75 arcsec and compare to its exact cached QMC result",
            "no_image_simulation_or_measurement": True,
        },
        "provenance": {
            "base_worker": str(BASE_ROOT / "worker.py"),
            "base_worker_sha256": digest(BASE_ROOT / "worker.py"),
            "candidate_models": [
                {"path": str(path), "sha256": value}
                for path, value in zip(MODEL_PATHS, MODEL_HASHES)
            ],
            "mock_manifest": str(MOCK_ROOT / "manifest.json"),
            "mock_manifest_sha256": digest(MOCK_ROOT / "manifest.json"),
        },
        "bootstrap": {
            "unit": "simulation case",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "limitations": [
            "The 0.5-arcsec model-cut increment uses one retained flow draw per object; its 0.75-arcsec reproduction error is the empirical precision check.",
            "Cases120-139 are a reused one-axis development screen, not a fresh acceptance set.",
            "SExtractor FLUX_RADIUS is a convolved half-light-radius measurement, not T or a deterministic transform of intrinsic Re.",
        ],
    }
    write_json(output, payload)
    print(f"RADIUS050_EXISTING_COMPLETE output={output}", flush=True)


def selftest() -> None:
    exact = np.arange(24, dtype=float).reshape(2, 3, 4)
    raw_base = exact + 10
    raw_cut = raw_base - 0.25
    np.testing.assert_allclose(exact + raw_cut - raw_base, exact - 0.25)
    print("RADIUS050_EXISTING_SELFTEST_PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "run"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.mode == "selftest":
        selftest()
    else:
        if arguments.output is None:
            parser.error("--output is required")
        run(arguments.output.resolve(), arguments.device)
