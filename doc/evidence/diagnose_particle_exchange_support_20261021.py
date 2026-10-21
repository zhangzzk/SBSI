"""Read-only CPU support audit of native target/neighbour exchange proposals.

Run with imports explicitly pinned to --source, a frozen sampler source tree.
No likelihood evaluations, sampler reruns, population cuts, or target changes
are made. Conditional type support probabilities describe the saved particles,
not independent posterior samples or an acceptance-probability estimate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from check_galsbi_continuous_prior import build_prior
from sbsi.galsbi_prior import ExactMemo, upper_gamma
from sbsi.galsbi_scene_prior import GalSBISceneBatch
from sbsi.galsbi_target_coordinates import GalSBITargetCoordinates
from sbsi.scene_exchange import numerical_over_analytic, propose_native_exchange
import sbsi.scene_exchange as exchange_source


LEVELS = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
REPLAY_SEEDS = [2026102190, 2026102191, 2026102192, 2026102193]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def distribution(values, weights=None):
    values = np.asarray(values).reshape(-1)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights).reshape(-1)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    result = dict(total_count=len(values), finite_positive_weight_count=int(finite.sum()),
                  finite_weight=float(weights[finite].sum()))
    if finite.any():
        value, weight = values[finite], weights[finite]
        order = np.argsort(value)
        value, weight = value[order], weight[order]
        cumulative = np.cumsum(weight)/weight.sum()
        indices = np.minimum(np.searchsorted(cumulative, LEVELS, side="left"), len(value)-1)
        result.update(mean=float(np.sum(value*weight)/weight.sum()),
                      quantile_levels=LEVELS, quantiles=value[indices].tolist())
    return result


def native_distributions(data):
    return {key: distribution(data[key]) for key in ("z", "M_B", "m_r", "R_kpc", "n", "a")}


def support_details(prior, data):
    """Classify the production correction's support failure without replacing it."""
    analytic = prior.log_intensity(data)
    correction = numerical_over_analytic(prior, data)
    flags = {name: np.zeros(len(analytic), bool) for name in (
        "analytic_invalid", "y_below_table", "y_above_table", "cdf_slope_nonpositive",
        "inverse_value_above_logmax", "other_nonfinite_correction")}
    flags["analytic_invalid"] = ~np.isfinite(analytic)
    for kind, name in enumerate(prior.types):
        ix = np.flatnonzero((data["type"] == kind) & np.isfinite(analytic))
        if not len(ix):
            continue
        table = prior.tables[name]
        z, magnitude = data["z"][ix], data["M_B"][ix]
        zi = np.clip(np.searchsorted(table["z"], z, side="right")-1, 0, len(table["z"])-2)
        qz = np.diff(table["cdf"])[zi]/np.diff(table["z"])[zi]
        mstar, _, alpha = prior.luminosity_parameters(z, name)
        y = mstar-magnitude
        yy, tt = table["inverse_y"][::-1], table["log_g"][::-1]
        yi = np.clip(np.searchsorted(yy, y, side="right")-1, 0, len(yy)-2)
        slope = np.diff(tt)[yi]/np.diff(yy)[yi]
        value = tt[yi]+slope*(y-yy[yi])
        cap = np.minimum(prior.magnitude_cap(z), prior.p["lum_fct_m_max"])
        with np.errstate(divide="ignore"):
            logmax = np.log(upper_gamma(alpha+1, 10**(0.4*(mstar-cap))))
        flags["y_below_table"][ix] = y < yy[0]
        flags["y_above_table"][ix] = y > yy[-1]
        flags["cdf_slope_nonpositive"][ix] = ~(qz > 0)
        flags["inverse_value_above_logmax"][ix] = ~(value <= logmax)
    explained = np.logical_or.reduce([flag for name, flag in flags.items() if name != "other_nonfinite_correction"])
    flags["other_nonfinite_correction"] = ~np.isfinite(correction) & ~explained
    return analytic, correction, flags


def input_receipt(path):
    protocol = json.loads((path/"protocol.json").read_text())
    result = json.loads((path/"result.json").read_text())
    hashes = {name: digest(path/name) for name in ("protocol.json", "particles.npz", "scenes.npz", "result.json")}
    if (result.get("status") != "completed"
            or result.get("protocol_sha256") != hashes["protocol.json"]
            or result.get("particles_sha256") != hashes["particles.npz"]
            or result.get("scenes_sha256") != hashes["scenes.npz"]
            or result.get("observations") != protocol["observations"]):
        raise ValueError(f"inconsistent or incomplete saved particle run: {path}")
    return protocol, dict(path=str(path), sha256=hashes, seed=protocol["seed"],
                         observations=protocol["observations"], particles=protocol["particles"])


def inspect_run(path, protocol, prior, coordinates):
    with np.load(path/"particles.npz", allow_pickle=False) as arrays:
        x = arrays["coordinates"].reshape(-1, 10)
        saved_logp = arrays["log_prior"].reshape(-1)
    with np.load(path/"scenes.npz", allow_pickle=False) as arrays:
        targets = {key[7:]: arrays[key] for key in arrays.files if key.startswith("target_")}
        neighbours = {key[10:]: arrays[key] for key in arrays.files if key.startswith("neighbour_")}
        scene = GalSBISceneBatch(targets, neighbours, arrays["indptr"], arrays["offset_arcsec"])
    encoded = coordinates.encode(scene.targets)
    coordinate_error = float(np.max(np.abs(encoded-x)))
    if coordinate_error > 2e-9:
        raise ValueError(f"saved native/coordinate replay failed: {coordinate_error}")
    logp, _, probabilities = coordinates.log_mixture_prior(x)
    if not np.array_equal(np.isfinite(logp), np.isfinite(saved_logp)):
        raise ValueError("saved analytic target support changed")
    prior_error = float(np.max(np.abs(logp-saved_logp)))
    if prior_error > 1e-6:
        raise ValueError(f"saved analytic target density replay failed: {prior_error}")
    if not np.isfinite(logp).all():
        raise ValueError("this diagnostic requires final positive-target-mass particles")
    target_cases = []
    for kind in range(len(prior.types)):
        mark = {key: value.copy() for key, value in scene.targets.items()}
        mark["type"][:] = kind
        target_cases.append(support_details(prior, mark))
    target_finite = np.column_stack([np.isfinite(case[1]) for case in target_cases])
    target_support_mass = np.sum(probabilities*target_finite, axis=1)
    neighbour_lp, neighbour_correction, neighbour_flags = support_details(prior, scene.neighbours)
    native_owner = scene.owner
    count = protocol["particles"]
    object_owner = np.repeat(np.arange(len(protocol["observations"])), count)
    neighbour_object = object_owner[native_owner]
    records = []
    for index, oid in enumerate(protocol["observations"]):
        rows = np.flatnonzero(object_owner == index)
        inside = neighbour_object == index
        record = dict(observation=oid, particles=len(rows), target_native=native_distributions(
            {key: value[rows] for key, value in scene.targets.items()}),
            target_type_probabilities_mean=probabilities[rows].mean(axis=0).tolist(),
            target_numerical_support_conditional_type_mass=distribution(target_support_mass[rows]),
            target_zero_conditional_support_mass_count=int((target_support_mass[rows] == 0).sum()),
            target_all_types_numerically_invalid_count=int((~target_finite[rows].any(axis=1)).sum()),
            target_by_type=[])
        for kind, (lp, correction, flags) in enumerate(target_cases):
            weight = probabilities[rows, kind]
            record["target_by_type"].append(dict(type=prior.types[kind],
                analytic_support_count=int(np.isfinite(lp[rows]).sum()),
                numerical_and_analytic_support_count=int(np.isfinite(correction[rows]).sum()),
                conditional_type_mass=float(weight.sum()),
                numerical_support_conditional_type_mass=float(weight[np.isfinite(correction[rows])].sum()),
                failure_counts={name: int(flag[rows].sum()) for name, flag in flags.items()},
                failure_conditional_type_mass={name: float(weight[flag[rows]].sum()) for name, flag in flags.items()},
                log_q_over_p=distribution(correction[rows], weight),
                analytic_log_intensity=distribution(lp[rows], weight)))
        correction_matrix = np.column_stack([case[1][rows] for case in target_cases])
        record["target_log_q_over_p_conditional_type_weighted"] = distribution(correction_matrix, probabilities[rows])
        record["neighbours"] = dict(count=int(inside.sum()),
            analytic_support_count=int(np.isfinite(neighbour_lp[inside]).sum()),
            numerical_and_analytic_support_count=int(np.isfinite(neighbour_correction[inside]).sum()),
            failure_counts={name: int(flag[inside].sum()) for name, flag in neighbour_flags.items()},
            log_q_over_p=distribution(neighbour_correction[inside]),
            native=native_distributions({key: value[inside] for key, value in scene.neighbours.items()}),
            count_per_scene=distribution(scene.counts[rows]))
        records.append(record)
    replays = []
    for mode in ("fixed", "relocated"):
        settings = {} if mode == "fixed" else dict(relocation_scale=0.1, aperture_arcsec=10.0, central_weight=0.9)
        for seed in REPLAY_SEEDS:
            proposed, correction, active = propose_native_exchange(
                prior, scene, probabilities, np.random.default_rng(seed), radius_scale=1.0, **settings)
            new_lp, new_qp, _ = support_details(prior, proposed.targets)
            replays.append(dict(mode=mode, seed=seed, observations=[dict(
                observation=oid, active_count=int(active[object_owner == index].sum()),
                finite_native_correction_count=int((active & np.isfinite(correction))[object_owner == index].sum()),
                incoming_neighbour_analytic_support_count=int((active & np.isfinite(new_lp))[object_owner == index].sum()),
                incoming_neighbour_numerical_and_analytic_support_count=int((active & np.isfinite(new_qp))[object_owner == index].sum()),
                finite_log_correction=distribution(correction[(object_owner == index) & active]))
                for index, oid in enumerate(protocol["observations"])]))
    return dict(run=str(path), seed=protocol["seed"], coordinate_replay_max=coordinate_error,
                analytic_prior_replay_max=prior_error, observations=records, native_proposal_replays=replays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = args.source.resolve()/"sbsi/scene_exchange.py"
    if Path(exchange_source.__file__).resolve() != expected:
        raise ValueError("exchange implementation import is not the frozen requested source")
    args.output.mkdir(parents=True)
    source_files = ["sbsi/scene_exchange.py", "sbsi/galsbi_prior.py", "sbsi/galsbi_target_coordinates.py",
                    "sbsi/galsbi_scene_prior.py", "scripts/check_galsbi_continuous_prior.py"]
    inputs = [input_receipt(path.resolve()) for path in args.runs]
    identity = lambda p: {key: p[key] for key in ("observations", "reference_protocol_sha256", "observation_sha256")}
    if any(identity(protocol) != identity(inputs[0][0]) for protocol, _ in inputs):
        raise ValueError("diagnostic runs use different observations or references")
    manifest = dict(script_sha256=digest(__file__), source=str(args.source.resolve()),
                    source_sha256={name: digest(args.source/name) for name in source_files},
                    inputs=[receipt for _, receipt in inputs], replay_seeds=REPLAY_SEEDS,
                    scope="Frozen final states; analytic/numerical support and native proposal correction only. No K or full acceptance probability.")
    (args.output/"protocol.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    started = time.monotonic()
    galaxy, cosmo, _ = build_prior()
    distance = ExactMemo(lambda z: cosmo.background.dist_ang_a(a=1/(1+z)))
    coordinates = GalSBITargetCoordinates(galaxy, distance)
    report = dict(protocol_sha256=digest(args.output/"protocol.json"), table_support={
        name: dict(y_min=float(table["inverse_y"].min()), y_max=float(table["inverse_y"].max()),
                   redshift_max=float(table["z"].max()), zero_redshift_cdf_intervals=int((np.diff(table["cdf"]) == 0).sum()))
        for name, table in galaxy.tables.items()}, runs=[])
    for path, (protocol, _) in zip(args.runs, inputs):
        result = inspect_run(path.resolve(), protocol, galaxy, coordinates)
        report["runs"].append(result)
        (args.output/"progress.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print("EXCHANGE_SUPPORT", path.name, [(x["observation"], x["target_zero_conditional_support_mass_count"],
              x["target_numerical_support_conditional_type_mass"]["mean"]) for x in result["observations"]], flush=True)
    report["elapsed_seconds"] = time.monotonic()-started
    report["status"] = "completed"
    (args.output/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("EXCHANGE_SUPPORT_COMPLETE", args.output, report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
