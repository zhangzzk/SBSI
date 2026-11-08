"""Exhaust one original four-row group's A/B likelihood over the frozen grid."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(os.environ["SBSI_EXACT_PROFILE_ROOT"])
SOURCE = Path(os.environ["SBSI_INDEPENDENT_ROOT"])
sys.path.insert(0, str(SOURCE))
import run_v39_independent_profiles as base  # noqa: E402

np, torch, study, save = base.np, base.torch, base.study, base.save


def protocol():
    p = study.read(ROOT / "protocol.json")
    study.verify(__file__, p["script_sha256"])
    for path, expected in p["sources"].items():
        study.verify(path, expected)
    base.protocol()
    return p


def worker(index):
    p = protocol()
    source_protocol = base.protocol()
    priors = study.read(SOURCE / "priors.json")
    group = study.read(SOURCE / "cohort.json")["groups"][p["group"]]
    source_group = SOURCE / f"groups/group_{p['group']:03d}"
    for name, expected in group["input_sha256"].items():
        study.verify(source_group / "input" / name, expected)
    sample = study.MockCatalogue.load(source_group / "input")
    likelihood = base.load_likelihood()
    raw = sample.measurements[list(likelihood.target_names)].to_numpy(float)
    center = source_protocol["points"].index(source_protocol["center"])
    assigned = [i for i in range(len(source_protocol["points"])) if i % 4 == index and i != center]
    order = [center, *assigned]
    points = [source_protocol["points"][i] for i in order]
    values = np.full((2, len(order), 4), -np.inf)
    n = len(likelihood.cache.prior.weights)
    started = time.monotonic()
    with base.GroupResponseMemo(likelihood, raw):
        for pi, point in enumerate(points):
            for start in range(0, n, 16384):
                atoms = np.arange(start, min(start + 16384, n))
                terms = base.full_block_terms(likelihood, sample.measurements, point, atoms).cpu().numpy()
                assert not np.isnan(terms).any() and not np.isposinf(terms).any()
                for arm, (lo, hi) in enumerate(priors["ranges"]):
                    active = (atoms >= lo) & (atoms < hi)
                    if active.any():
                        values[arm, pi] = np.logaddexp(
                            values[arm, pi], np.logaddexp.reduce(terms[:, active], axis=1)
                        )
                if start // 16384 % 128 == 0:
                    save(
                        ROOT / f"progress_{index}.json",
                        dict(
                            point=pi,
                            total_points=len(order),
                            atoms=int(atoms[-1]) + 1,
                            elapsed_seconds=time.monotonic() - started,
                        ),
                    )
                    print("EXACT_PROFILE_SCAN", index, pi, int(atoms[-1]) + 1, n, flush=True)
            values[:, pi] -= np.log(priors["masses"])[:, None]
            if pi == 0:
                with np.load(source_group / "profile.npz") as old:
                    expected = old["exact_center_log_numerator"]
                np.testing.assert_allclose(values[:, pi], expected, rtol=0, atol=1.0e-7)
                save(
                    ROOT / f"replay_{index}.json",
                    dict(
                        passed=True,
                        maximum_error=float(abs(values[:, pi] - expected).max()),
                        tolerance=1.0e-7,
                    ),
                )
            np.savez_compressed(
                ROOT / f"partial_{index}.npz",
                point_indices=order[: pi + 1],
                log_numerator=values[:, : pi + 1],
            )
            print("EXACT_PROFILE_POINT", index, pi + 1, len(order), time.monotonic() - started, flush=True)
    dest = ROOT / f"worker_{index}.npz"
    np.savez_compressed(dest, point_indices=order, log_numerator=values)
    save(
        ROOT / f"worker_{index}.json",
        dict(
            status="complete",
            group=p["group"],
            protocol_sha256=study.file_hash(ROOT / "protocol.json"),
            output_sha256=study.file_hash(dest),
            elapsed_seconds=time.monotonic() - started,
        ),
    )


def combine():
    p = protocol()
    source = base.protocol()
    identity = study.file_hash(ROOT / "protocol.json")
    npoints = len(source["points"])
    center = source["points"].index(source["center"])
    values = np.full((2, npoints, 4), np.nan)
    seen = np.zeros(npoints, int)
    for index in range(4):
        record = study.read(ROOT / f"worker_{index}.json")
        assert record["status"] == "complete" and record["protocol_sha256"] == identity
        study.verify(ROOT / f"worker_{index}.npz", record["output_sha256"])
        with np.load(ROOT / f"worker_{index}.npz") as f:
            for j, point in enumerate(f["point_indices"]):
                if seen[point]:
                    assert point == center
                    np.testing.assert_allclose(
                        values[:, point], f["log_numerator"][:, j], rtol=0, atol=1.0e-7
                    )
                values[:, point] = f["log_numerator"][:, j]
                seen[point] += 1
    assert np.isfinite(values).all() and seen[center] == 4 and (seen[np.arange(npoints) != center] == 1).all()
    np.savez_compressed(ROOT / "exact_profile.npz", points=source["points"], log_numerator=values)
    group = study.read(SOURCE / "cohort.json")["groups"][p["group"]]
    objects = study.read(SOURCE / "audit/audit.json")["objects"][group["sample"]]
    with np.load(SOURCE / f"audit/{group['sample']}_rows.npz") as data:
        numerator = data["numerator"].copy()
        normal = data["primary_normalization"]
    indices = [i for i, obj in enumerate(objects) if obj["group"] == p["group"]]
    before = numerator.copy()
    exact = values - values[:, center : center + 1]
    for i in indices:
        slot = objects[i]["slot"]
        numerator[..., i] = exact[:, None, None, :, slot]
    totals = numerator.sum(-1) - source["n_observations"] * normal[:, None, None, :]
    primary = totals[:, :, -1].mean(1)
    reports = []
    for arm in range(2):
        # All grid nodes are retained in the numerical check for this repair.
        error = float(abs(totals[arm] - primary[arm]).max())
        previous = next(
            r
            for r in study.read(SOURCE / "result.json")["reports"]
            if r["sample"] == group["sample"] and r["prior"] == "AB"[arm]
        )
        reports.append(
            dict(
                sample=group["sample"],
                prior="AB"[arm],
                centered_log_likelihood=primary[arm].tolist(),
                profiles=totals[arm].tolist(),
                numerical_log_profile_max_difference=error,
                numerical_pass=error <= source["profile_log_tolerance"],
                normalization_pass=previous["normalization_pass"],
                normalization_log_profile_max_difference=previous["normalization_log_profile_max_difference"],
                grid_peak=source["points"][int(primary[arm].argmax())],
                peak_on_boundary=any(
                    abs(source["points"][int(primary[arm].argmax())][d] - source["center"][d])
                    >= 0.03 - 1.0e-12
                    for d in range(2)
                ),
                maximum_change_from_sampled_primary=float(
                    abs(primary[arm] - previous["centered_log_likelihood"]).max()
                ),
                maximum_original_group_error=float(
                    abs(before[arm, ..., indices] - numerator[arm, ..., indices]).max()
                ),
                status="coarse_grid_only_no_accepted_shear_estimate",
            )
        )
    save(
        ROOT / "result.json",
        dict(
            status="complete",
            protocol_sha256=identity,
            points=source["points"],
            group=p["group"],
            repaired_rows=[objects[i]["row"] for i in indices],
            reports=reports,
            limitation="One original four-row group evaluated exhaustively; other124 rows and both prior normalizers unchanged. Finite-grid diagnostic only; no0.3% claim.",
        ),
    )
    print("EXACT_PROFILE_COMPLETE", json.dumps(reports), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["worker", "combine"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        worker(int(os.environ["SLURM_ARRAY_TASK_ID"])) if args.stage == "worker" else combine()
    except Exception:
        save(
            ROOT / f"failure_{args.stage}_{os.environ.get('SLURM_ARRAY_TASK_ID', 'cpu')}.json",
            dict(traceback=traceback.format_exc()),
        )
        raise
