"""Per-separation-shell blending terms: partition, merge, and joint fit.

``R_blend`` is a signed sum over the 20 nearest neighbours inside r_max = 10
arcsec, and on the ConstGold domain that cap binds for nearly every primary
(mean 16 pairs, 33% at exactly the maximum).  A two-term fit therefore forces a
touching pair and one five arcseconds away to share a single multiplicative
error.  Splitting the sum into separation shells and giving each its own
coefficient is only meaningful if three things hold, which is what these tests
pin down:

1. the shells really are a partition -- they add back to ``R_blend`` exactly,
   at build time and again at merge time;
2. the merge cannot silently concatenate blocks cut on different edges;
3. the joint fit recovers per-shell errors that were planted, and reports a
   condition number that exposes the case where it cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_constgold_fixed_g0_blend_lookup import (
    shell_bounds,
    shell_column_names,
)
from scripts.diagnose_constgold_response_split import fit_joint, summarize
from scripts.merge_constgold_fixed_g0_blend_lookups import main as merge_main

H = 0.02


def test_shell_bounds_cover_the_aperture_without_gaps():
    bounds = shell_bounds((2.0, 4.0, 6.0), 10.0)
    assert bounds == ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 10.0))
    #  Half-open shells tile [0, r_max) exactly once, which is what makes the
    #  reconstruction identity in the builder an identity.
    assert bounds[0][0] == 0.0 and bounds[-1][1] == 10.0
    for lower, upper in zip(bounds, bounds[1:]):
        assert lower[1] == upper[0]


def test_shell_bounds_rejects_edges_outside_the_aperture():
    with pytest.raises(ValueError, match="strictly inside"):
        shell_bounds((0.0, 4.0), 10.0)
    with pytest.raises(ValueError, match="strictly inside"):
        shell_bounds((4.0, 10.0), 10.0)
    with pytest.raises(ValueError, match="repeated"):
        shell_bounds((4.0, 4.0), 10.0)


def test_no_shell_edges_means_no_shell_columns():
    assert shell_bounds((), 10.0) == ()
    assert shell_column_names(()) == ()


def test_shell_column_names_are_filesystem_and_feather_safe():
    names = shell_column_names(shell_bounds((1.5, 4.0), 10.0))
    assert names == ("R_blend_r0_1p5", "R_blend_r1p5_4", "R_blend_r4_10")
    assert all("." not in name for name in names)


def shell_block(path, cases, names, *, rows=4, break_partition=False):
    """A lookup block whose shells add to R_blend, unless told to break it."""
    frames = []
    for case in cases:
        shells = {
            name: np.linspace(-0.01, 0.02, rows) * (index + 1) / len(names)
            for index, name in enumerate(names)
        }
        total = np.sum(list(shells.values()), axis=0)
        absolute = np.sum([np.abs(v) for v in shells.values()], axis=0)
        if break_partition:
            #  Move R_blend away from its shells while keeping R_blend_abs a
            #  valid bound, so the partition check is what fails and not the
            #  pre-existing absolute-sum check.
            total = total + 0.5
            absolute = absolute + 0.5
        data = {
            "case": np.full(rows, case, dtype=np.int32),
            "input_index": np.arange(rows, dtype=np.int64),
            "R_blend": total,
            "R_blend_abs": absolute,
            "blend_pairs": np.full(rows, len(names), dtype=np.int64),
        }
        for name, values in shells.items():
            data[name] = values
            data[f"{name}_abs"] = np.abs(values)
        frames.append(pd.DataFrame(data))
    pd.concat(frames, ignore_index=True).to_feather(path)
    return path


NAMES = ("R_blend_r0_2", "R_blend_r2_4", "R_blend_r4_10")


def test_shell_columns_survive_the_merge(tmp_path):
    a = shell_block(tmp_path / "a.feather", [40, 41], NAMES)
    b = shell_block(tmp_path / "b.feather", [42], NAMES)
    out = tmp_path / "merged.feather"
    merge_main(
        ["--input", str(a), "--input", str(b), "--output", str(out)]
        + [arg for case in (40, 41, 42) for arg in ("--case", str(case))]
    )
    merged = pd.read_feather(out)
    for name in NAMES:
        assert name in merged.columns
        assert f"{name}_abs" in merged.columns
    rebuilt = merged[list(NAMES)].to_numpy(float).sum(axis=1)
    assert rebuilt == pytest.approx(merged["R_blend"].to_numpy(float), abs=1e-12)


def test_merge_refuses_blocks_whose_shells_do_not_reconstruct_r_blend(tmp_path):
    a = shell_block(tmp_path / "a.feather", [40], NAMES)
    b = shell_block(tmp_path / "b.feather", [41], NAMES, break_partition=True)
    out = tmp_path / "merged.feather"
    with pytest.raises(ValueError, match="do not reconstruct R_blend"):
        merge_main(
            ["--input", str(a), "--input", str(b), "--output", str(out)]
            + [arg for case in (40, 41) for arg in ("--case", str(case))]
        )


def test_merge_drops_shells_a_block_lacks_rather_than_mixing_edges(tmp_path):
    #  Different shell edges produce differently named columns, so a block cut
    #  on other edges simply has none in common and the shells are dropped.
    #  That is the safe outcome: a partial set would silently stop summing to
    #  R_blend.
    a = shell_block(tmp_path / "a.feather", [40], NAMES)
    b = shell_block(tmp_path / "b.feather", [41],
                    ("R_blend_r0_3", "R_blend_r3_10"))
    out = tmp_path / "merged.feather"
    merge_main(
        ["--input", str(a), "--input", str(b), "--output", str(out)]
        + [arg for case in (40, 41) for arg in ("--case", str(case))]
    )
    merged = pd.read_feather(out)
    assert not [name for name in merged.columns if name.startswith("R_blend_r")]


def stats(value, n):
    return {"numerator": np.array([value * n, 0.0]), "denominator": float(n)}


def planted(flow, shells, measured, n=100):
    entry = {
        "measured": {"plus": stats(measured * H, n), "minus": stats(-measured * H, n)},
        "flow": {"plus": stats(flow * H, n), "minus": stats(-flow * H, n)},
    }
    for label, value in shells.items():
        entry[f"blend::{label}"] = {
            "plus": stats(value * H, n), "minus": stats(-value * H, n)}
    return entry


def test_joint_fit_recovers_a_per_shell_error_the_two_term_fit_averages_away():
    #  Plant the situation the ConstGold profile suggests: the near shell is
    #  modelled well and the far shell is badly over-predicted.  A single
    #  blending coefficient can only report the blend of the two, while the
    #  joint fit must separate them.
    truth = {"near": 1.00, "far": 0.40}
    flow_error = 1.02
    keys = tuple(f"bin{index:02d}" for index in range(6))
    #  Near and far vary independently across bins, which is the condition the
    #  real shells were checked to satisfy (per-object correlation <= 0.02).
    near = [0.02, 0.30, 0.02, 0.30, 0.16, 0.16]
    far = [0.05, 0.05, 0.50, 0.50, 0.05, 0.50]
    per_case = {}
    for case in range(5):
        entry = {}
        totals = {"flow": 0.0, "near": 0.0, "far": 0.0, "measured": 0.0}
        for index, key in enumerate(keys):
            shells = {"near": near[index], "far": far[index]}
            measured = flow_error * 0.48 + sum(
                truth[name] * value for name, value in shells.items())
            entry[key] = planted(0.48, shells, measured)
            totals["flow"] += 0.48
            totals["near"] += shells["near"]
            totals["far"] += shells["far"]
            totals["measured"] += measured
        entry["all"] = planted(
            totals["flow"] / len(keys),
            {"near": totals["near"] / len(keys), "far": totals["far"] / len(keys)},
            totals["measured"] / len(keys), n=100 * len(keys))
        per_case[str(case)] = entry
    groups = {key: [key] for key in keys}
    groups["all"] = list(keys)

    out = summarize(per_case, range(5), keys, groups, keys, ["near", "far"],
                    h=H, replicates=20, seed=5, joint=True)
    joint = out["joint_fit"]
    assert joint is not None
    assert joint["joint_flow_fractional_error"]["value"] == pytest.approx(0.02)
    assert joint["joint_fractional_error::near"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert joint["joint_fractional_error::far"]["value"] == pytest.approx(-0.60)
    #  A planted model with no noise must be fitted exactly.
    assert joint["joint_weighted_residual"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert joint["joint_condition_number"]["value"] < 100.0


def test_joint_fit_is_absent_unless_asked_for():
    keys = ("bin00", "bin01")
    per_case = {
        str(case): {
            "bin00": planted(0.48, {"a": 0.1}, 0.5),
            "bin01": planted(0.48, {"a": 0.5}, 1.0),
            "all": planted(0.48, {"a": 0.3}, 0.75, n=200),
        }
        for case in range(3)
    }
    groups = {key: [key] for key in keys}
    groups["all"] = list(keys)
    out = summarize(per_case, range(3), keys, groups, keys, ["a"],
                    h=H, replicates=5, seed=1)
    assert out["joint_fit"] is None


def test_collinear_components_are_flagged_by_the_condition_number():
    #  Two shells that always move together cannot be separated.  The
    #  coefficients still come back, so the condition number is the only thing
    #  standing between a reader and a confident meaningless split.
    flow = np.full(6, 0.48)
    near = np.array([0.02, 0.10, 0.18, 0.26, 0.34, 0.42])
    independent = np.array([0.05, 0.50, 0.05, 0.50, 0.30, 0.10])
    collinear = 2.0 * near
    weights = np.full(6, 100.0)

    good = fit_joint(flow, np.column_stack((near, independent)),
                     0.02 * flow - 0.30 * independent, weights, ["near", "far"])
    assert np.isfinite(good["joint_condition_number"])
    assert good["joint_condition_number"] < 100.0
    assert good["joint_fractional_error::far"] == pytest.approx(-0.30)

    bad = fit_joint(flow, np.column_stack((near, collinear)),
                    0.02 * flow - 0.30 * collinear, weights, ["near", "far"])
    #  lstsq still returns coefficients here; the condition number is the only
    #  thing that says the near/far split behind them is not determined.
    assert bad["joint_condition_number"] > 1.0e6
