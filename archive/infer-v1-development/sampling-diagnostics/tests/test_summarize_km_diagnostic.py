"""Archived regression tests for the completed K/M diagnostic."""

import hashlib
import json

import numpy as np

from scripts.summarize_km_diagnostic import main


def _write_arm(root, k, seed, first, second):
    arm = root / "results" / f"k{k}_e0p1_p{seed}_m512_65536"
    arm.mkdir(parents=True)
    moments = arm / "one_step_moments.npz"
    np.savez_compressed(moments, unique_counts=np.array([3, 4, 5]))
    digest = hashlib.sha256(moments.read_bytes()).hexdigest()
    identity = {"same": "identity"}
    payload = {
        "config": {
            "proposal_candidates": k,
            "proposal_seed": seed,
            "proposal_epsilon": 0.1,
            "proposal_flow_samples": 128,
            "proposal_statistic": "mean",
            "proposal_dispersion_statistic": "std",
            "proposal_prefilter_candidates": 65536,
            "retain_full_ladder": True,
            "adaptive_draw_ladder": [512, 65536],
        },
        "full_ladder": {
            "512": {"estimate": list(first)},
            "65536": {"estimate": list(second)},
        },
        "result": {
            "elapsed_seconds": 12.0,
            "flow_evaluations": 100,
            "mean_unique_fraction": 0.5,
        },
        "initial_center": identity,
        "observation_partition": {
            "start": 0,
            "stop": 20000,
            "n_partition": 20000,
            "n_total": 100000,
        },
        "mock_input_sha256": identity,
        "proposal_cache_sha256": identity,
        "one_step_moments": {"path": moments.name, "sha256": digest},
    }
    (arm / "result.json").write_text(json.dumps(payload))


def test_km_summary_selects_smallest_paired_stable_k_and_m(tmp_path):
    for k, offset in ((8192, 0.0), (32768, 2.0e-5), (65536, 3.0e-5)):
        _write_arm(tmp_path, k, 8701, (0.01 + offset, 0.0), (0.01004 + offset, 0.0))
    main(["--root", str(tmp_path), "--mode", "select"])
    assert (tmp_path / "selected_k.txt").read_text().strip() == "8192"

    _write_arm(tmp_path, 8192, 8702, (0.01001, 0.0), (0.01005, 0.0))
    main(["--root", str(tmp_path), "--mode", "final"])
    report = json.loads((tmp_path / "final_summary.json").read_text())
    assert report["selected_k_for_second_seed"] == 8192
    assert report["seed_endpoint_passes_tolerance"]
    assert report["smallest_m_passing_nested_tail_and_seed_gate"] == 512
    assert report["material_m_reduction_supported"]
