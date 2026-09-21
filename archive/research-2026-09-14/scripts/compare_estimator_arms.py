#!/usr/bin/env python
"""Compare common-random-number paired inference arms across a draw ladder.

A screen submits two or more runs that differ in exactly one declared numerical
choice -- the estimator mode, the candidate pool -- on one frozen mock with one
proposal seed.  Reading them means three things, and each is easy to get wrong
by hand:

1. Confirming the arms really are paired.  If anything but the declared choice
   differs, the comparison measures that instead, so every identity is checked
   and the run refuses rather than reporting a number.
2. Differencing the arms with the pairing kept.  The arms share objects and
   draws, so their estimates are correlated and the unpaired standard errors
   badly overstate the uncertainty on their difference.  The paired error comes
   from the per-object influence functions.
3. Reading the weight diagnostics on the right column.  ESS as a fraction of
   the draw budget does not compare across estimator modes; the relative
   standard error does.  See doc/CATALOGUE_PRIOR.md#weight-diagnostics.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


# Identity that must agree for any comparison to mean anything.  `pipeline_config`
# is handled separately, because that is where the declared difference lives.
STRICT_IDENTITY = (
    "likelihood_release",
    "likelihood_config_sha256",
    "likelihood_component_sha256",
    "initial_center",
    "injected_shear",
    "observation_partition",
    "model_sha256",
    "model_cache_sha256",
    "scene_sha256",
    "proposal_cache_sha256",
    "mock_input_sha256",
    "implementation_sha256",
)

COMPONENT_INDEX = {"g1": 0, "g2": 1}


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(label: str, root: str | Path) -> dict:
    """Load one arm's result payload and its verified per-object moments."""

    root = Path(root).resolve()
    payload = json.loads((root / "result.json").read_text())
    moment_path = root / payload["one_step_moments"]["path"]
    if _sha256(moment_path) != payload["one_step_moments"]["sha256"]:
        raise RuntimeError(f"moment hash mismatch in {root}")
    arrays = np.load(moment_path)
    if "ladder_score" not in arrays:
        raise RuntimeError(
            f"{root} did not retain the full ladder; there is nothing to compare "
            "across draw counts"
        )
    ladder = tuple(int(value) for value in payload["result"]["draw_ladder"])
    if len(ladder) != len(arrays["ladder_score"]):
        raise RuntimeError(f"ladder length disagrees with retained moments in {root}")
    return {
        "label": label,
        "root": root,
        "payload": payload,
        "center": np.asarray(payload["result"]["center"], dtype=np.float64),
        "ladder": ladder,
        "ladder_score": np.asarray(arrays["ladder_score"], dtype=np.float64),
        "ladder_information": np.asarray(arrays["ladder_information"], dtype=np.float64),
        "estimator_mode": payload["result"].get("estimator_mode", "mixture"),
        "pipeline_release": payload.get("pipeline_release"),
        "weight_diagnostics": payload["result"].get("weight_diagnostics"),
    }


def check_pairing(arms, allowed_config_keys) -> dict:
    """Reject arms that differ anywhere but the declared choice.

    Returns the keys of `pipeline_config` that actually differ, so a screen that
    silently overrode nothing is as visible as one that overrode too much.
    """

    reference = arms[0]
    allowed = set(allowed_config_keys)
    differing = set()
    for arm in arms[1:]:
        mismatched = [
            name
            for name in STRICT_IDENTITY
            if json.dumps(reference["payload"].get(name), sort_keys=True)
            != json.dumps(arm["payload"].get(name), sort_keys=True)
        ]
        if mismatched:
            raise RuntimeError(
                f"{arm['label']} is not paired with {reference['label']}: "
                f"{sorted(mismatched)} differ"
            )
        if len(arm["ladder_score"]) != len(reference["ladder_score"]) or arm[
            "ladder"
        ] != reference["ladder"]:
            raise RuntimeError(f"{arm['label']} uses a different draw ladder")
        if arm["ladder_score"].shape != reference["ladder_score"].shape:
            raise RuntimeError(f"{arm['label']} covers a different set of objects")
        left = reference["payload"]["pipeline_config"]
        right = arm["payload"]["pipeline_config"]
        for name in sorted(set(left) | set(right)):
            if json.dumps(left.get(name), sort_keys=True) != json.dumps(
                right.get(name), sort_keys=True
            ):
                differing.add(name)
    outside = sorted(differing - allowed)
    if outside:
        raise RuntimeError(
            f"arms differ in undeclared configuration keys {outside}; pass "
            "--allow-differing for each key the screen varies on purpose"
        )
    return {
        "differing_config_keys": sorted(differing),
        "allowed_config_keys": sorted(allowed),
        "n_objects": int(reference["ladder_score"].shape[1]),
    }


def rung_solutions(arm) -> list:
    """One-step estimate and per-object influence function at every rung.

    The influence function is what makes any later difference paired: the
    estimate is a smooth function of per-object sums, so differences of
    influence functions give the correct standard error on differences of
    estimates whether the difference is between rungs or between arms.
    """

    solutions = []
    for rung_index, n_draws in enumerate(arm["ladder"]):
        score = arm["ladder_score"][rung_index]
        information = arm["ladder_information"][rung_index]
        if information.ndim == 2:
            information = np.stack(
                [np.diag(row) for row in information]
            )  # diagonal-information runs
        information_sum = information.sum(axis=0, dtype=np.float64)
        information_sum = 0.5 * (information_sum + information_sum.T)
        eigenvalues = np.linalg.eigvalsh(information_sum)
        entry = {
            "draws": int(n_draws),
            "information_eigenvalues": eigenvalues.tolist(),
        }
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
            entry.update({"estimate": None, "influence": None})
            solutions.append(entry)
            continue
        step = np.linalg.solve(information_sum, score.sum(axis=0, dtype=np.float64))
        residual = score - np.einsum("nij,j->ni", information, step)
        influence = np.linalg.solve(information_sum / len(score), residual.T).T
        entry["estimate"] = arm["center"] + step
        entry["influence"] = influence
        entry["robust_standard_error"] = np.std(influence, axis=0, ddof=1) / np.sqrt(
            len(score)
        )
        solutions.append(entry)
    return solutions


def _paired(left, right, component: int):
    """Shift, paired standard error, and pull between two solved rungs."""

    if left.get("influence") is None or right.get("influence") is None:
        return None
    shift = float(right["estimate"][component] - left["estimate"][component])
    difference = right["influence"][:, component] - left["influence"][:, component]
    error = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
    return {
        "shift": shift,
        "paired_standard_error": error,
        "pull": (shift / error) if error > 0 else None,
    }


def ladder_report(arm, component: int) -> dict:
    """Per-rung estimate with the paired drift against the previous rung."""

    solutions = rung_solutions(arm)
    rows = []
    for index, entry in enumerate(solutions):
        row = {
            "draws": entry["draws"],
            "estimate": (
                None if entry.get("estimate") is None else float(entry["estimate"][component])
            ),
            "robust_standard_error": (
                None
                if entry.get("estimate") is None
                else float(entry["robust_standard_error"][component])
            ),
            "previous_rung": (
                None if index == 0 else _paired(solutions[index - 1], entry, component)
            ),
        }
        rows.append(row)
    return {
        "rows": rows,
        "solutions": solutions,
        # The whole-ladder drift is the screen's headline: a converged estimator
        # has no systematic climb from the first rung to the last.
        "ladder_drift": _paired(solutions[0], solutions[-1], component),
    }


def _format_value(value, width=10, digits=6):
    return " " * width if value is None else f"{value:+.{digits}f}".rjust(width)


def _format_pull(entry):
    """Render a pull, capping the display rather than wrecking the column.

    A paired error can be numerically zero -- two rungs that happen to move
    every object by the identical amount -- and the ratio then carries no
    information beyond "much larger than any threshold".  The uncapped value
    still goes to the JSON report.
    """

    if entry is None or entry.get("pull") is None:
        return "        "
    pull = entry["pull"]
    if not np.isfinite(pull):
        return "     inf"
    if abs(pull) > 9999:
        return ("  >+9999" if pull > 0 else "  <-9999")
    return f"{pull:+.2f}".rjust(8)


def _pull_text(entry):
    if entry is None or entry.get("pull") is None:
        return "n/a"
    return _format_pull(entry).strip()


def render_ladders(reports, component_name: str) -> str:
    """One row per rung, one block per arm, in the order given."""

    lines = [f"Draw ladder, {component_name} (paired errors against the previous rung)"]
    for label, report in reports.items():
        lines.append(f"  {label}")
        lines.append(
            "      M          estimate      robust se        shift    paired se     pull"
        )
        for row in report["rows"]:
            previous = row["previous_rung"] or {}
            lines.append(
                "  %7d  %s  %s  %s  %s %s"
                % (
                    row["draws"],
                    _format_value(row["estimate"], 15),
                    _format_value(row["robust_standard_error"], 13),
                    _format_value(previous.get("shift"), 12),
                    _format_value(previous.get("paired_standard_error"), 12),
                    _format_pull(row["previous_rung"]),
                )
            )
        drift = report["ladder_drift"]
        if drift is not None:
            lines.append(
                "    first rung to last: %+.6f +- %.6f (pull %s)"
                % (
                    drift["shift"],
                    drift["paired_standard_error"],
                    _pull_text(drift),
                )
            )
    return "\n".join(lines)


def cross_arm_report(reports, reference_label: str, component: int) -> dict:
    """Difference between arms at each rung, keeping the common draws paired."""

    reference = reports[reference_label]["solutions"]
    out = {}
    for label, report in reports.items():
        if label == reference_label:
            continue
        rows = []
        for index, entry in enumerate(report["solutions"]):
            paired = _paired(reference[index], entry, component)
            rows.append({"draws": entry["draws"], **(paired or {})})
        out[label] = rows
    return out


def render_cross_arm(cross, reference_label: str, component_name: str) -> str:
    lines = [
        f"Arm difference at each rung, {component_name} "
        f"(relative to {reference_label}, paired)"
    ]
    for label, rows in cross.items():
        lines.append(f"  {label} minus {reference_label}")
        lines.append("      M         difference    paired se     pull")
        for row in rows:
            lines.append(
                "  %7d  %s  %s %s"
                % (
                    row["draws"],
                    _format_value(row.get("shift"), 15),
                    _format_value(row.get("paired_standard_error"), 12),
                    _format_pull(row),
                )
            )
    return "\n".join(lines)


def render_weights(arms) -> str:
    """The weight diagnostics, read on the column that compares."""

    lines = [
        "Weight diagnostics (zero-shear view).  Compare arms on relative se,",
        "NOT on ESS/M: an exactly summed stratum carries no variance and no",
        "draws, so it lowers ESS/M while improving the estimate.",
    ]
    for arm in arms:
        rows = arm["weight_diagnostics"]
        lines.append(f"  {arm['label']}  (estimator_mode={arm['estimator_mode']})")
        if not rows:
            lines.append("    none recorded (run predates the diagnostics)")
            continue
        lines.append(
            "      M      ESS/M p50    rel se p50    rel se p90     k p50   k>thr   undef"
        )
        for row in rows:
            khat = row.get("pareto_k_percentiles")
            above = row.get("pareto_k_above_threshold")
            lines.append(
                "  %7d  %11.4f  %12.6f  %12.6f  %8s  %6s  %6d"
                % (
                    row["draws"],
                    row["ess_fraction_percentiles"][2],
                    row["relative_standard_error_percentiles"][2],
                    row["relative_standard_error_percentiles"][3],
                    "n/a" if khat is None else f"{khat[2]:+.3f}",
                    "n/a" if above is None else f"{above:.2f}",
                    row["pareto_k_undefined"],
                )
            )
        final = rows[-1]
        threshold = final.get("pareto_k_threshold")
        above = final.get("pareto_k_above_threshold")
        if above is not None and threshold is not None:
            lines.append(
                "    at M=%d, %.0f%% of fitted objects sit above k=%.2f%s"
                % (
                    final["draws"],
                    100.0 * above,
                    threshold,
                    (
                        " -- infinite weight variance, so no root-M rate exists "
                        "and more draws cannot fix this arm"
                        if above > 0.5
                        else ""
                    ),
                )
            )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=DIRECTORY",
        help="one screen arm; the first is the reference the others differ from",
    )
    parser.add_argument("--component", choices=("g1", "g2"), default="g1")
    parser.add_argument(
        "--allow-differing",
        action="append",
        default=None,
        help=(
            "a pipeline_config key the screen varies on purpose "
            "(default: estimator)"
        ),
    )
    parser.add_argument("--output", default=None, help="also write the report as JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    arms = []
    for item in args.arm:
        if "=" not in item:
            raise SystemExit(f"--arm expects LABEL=DIRECTORY, got {item}")
        label, _, root = item.partition("=")
        arms.append(load_arm(label, root))
    allowed = args.allow_differing if args.allow_differing is not None else ["estimator"]
    pairing = check_pairing(arms, allowed)

    component = COMPONENT_INDEX[args.component]
    reports = {arm["label"]: ladder_report(arm, component) for arm in arms}
    cross = cross_arm_report(reports, arms[0]["label"], component)

    print(
        "Paired on %d objects; pipeline_config differs in %s"
        % (
            pairing["n_objects"],
            pairing["differing_config_keys"] or "nothing at all",
        )
    )
    for arm in arms:
        print(f"  {arm['label']}: {arm['root']}  [{arm['pipeline_release']}]")
    print()
    print(render_ladders(reports, args.component))
    print()
    print(render_cross_arm(cross, arms[0]["label"], args.component))
    print()
    print(render_weights(arms))

    if args.output:
        payload = {
            "component": args.component,
            "pairing": pairing,
            "arms": {
                arm["label"]: {
                    "root": str(arm["root"]),
                    "pipeline_release": arm["pipeline_release"],
                    "estimator_mode": arm["estimator_mode"],
                    "ladder": reports[arm["label"]]["rows"],
                    "ladder_drift": reports[arm["label"]]["ladder_drift"],
                    "weight_diagnostics": arm["weight_diagnostics"],
                }
                for arm in arms
            },
            "arm_difference": cross,
            "reference": arms[0]["label"],
        }
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
