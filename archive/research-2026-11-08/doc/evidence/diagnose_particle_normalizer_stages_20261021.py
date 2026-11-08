"""Read-only stage/coordinate diagnosis of independent particle populations.

Use a frozen input manifest and run numeric analysis through the scheduler.
Stage covariance allocations and coordinate associations are descriptive, not
causal effects. Resampled particles are never treated as independent replicates.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata


COORDINATES = ["log_z", "m_r", "sed_alr0", "sed_alr1", "sed_alr2", "sed_alr3",
               "log_circularized_radius", "sersic_logit", "distortion_logit"]
BETA_BOUNDARIES = [0.0, 0.01, 0.1, 0.5, 1.0]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def normalizer_summary(logz):
    finite = np.isfinite(logz)
    weights = np.zeros(len(logz))
    if finite.any():
        weights[finite] = np.exp(logz[finite]-np.max(logz[finite]))
    total = weights.sum()
    if not total:
        return dict(extinct=len(logz), run_weight_ess=0.0, relative_mcse=None)
    normalized = weights/total
    return dict(extinct=int((~finite).sum()), run_weight_ess=float(1/np.sum(normalized**2)),
                max_run_weight=float(normalized.max()), run_weights=normalized.tolist(),
                relative_mcse=float(weights.std(ddof=1)/np.sqrt(len(weights))/weights.mean()) if len(weights)>1 else None,
                log_mean_normalizer=float(np.max(logz[finite])+np.log(weights.mean())))


def association(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = dict(paired_runs=len(x), pearson=None, spearman=None)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return result
    result.update(pearson=float(np.corrcoef(x, y)[0, 1]),
                  spearman=float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]))
    if len(x) >= 4:
        loo = []
        for i in range(len(x)):
            xx, yy = np.delete(x, i), np.delete(y, i)
            if xx.std() > 0 and yy.std() > 0:
                loo.append(float(np.corrcoef(xx, yy)[0, 1]))
        result["leave_one_run_out_pearson_range"] = [min(loo), max(loo)] if loo else None
    return result


def summarize_distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()),
                sd=float(values.std(ddof=1)) if len(values)>1 else None,
                minimum=float(values.min()), median=float(np.median(values)), maximum=float(values.max()))


def summarize_acceptance(stage_windows, oi, particles):
    """Missing historical stage counters stay missing, never inferred from totals."""
    available = [all("acceptance" in stage for stage in window) for window in stage_windows]
    result = dict(available_by_run=available, acceptance=None)
    if not all(available):
        return result
    blocks = sorted(stage_windows[0][0]["acceptance"])
    acceptance = {}
    for block in blocks:
        attempted = np.array([sum(stage["acceptance"][block]["attempted"][oi] for stage in window)
                              for window in stage_windows])
        accepted = np.array([sum(stage["acceptance"][block]["accepted"][oi] for stage in window)
                             for window in stage_windows])
        acceptance[block] = dict(attempted=int(attempted.sum()), accepted=int(accepted.sum()),
            fraction=float(accepted.sum()/attempted.sum()) if attempted.sum() else None,
            fractions_by_run=finite_list(np.divide(accepted, attempted,
                out=np.full(len(stage_windows), np.nan), where=attempted>0)),
            mean_accepted_moves_per_particle=float(accepted.mean()/particles))
    result["acceptance"] = acceptance
    return result


def read_run(receipt):
    path = Path(receipt["path"])
    for name, expected in receipt["sha256"].items():
        if digest(path/name) != expected:
            raise ValueError(f"frozen input changed: {path/name}")
    protocol = json.loads((path/"protocol.json").read_text())
    result = json.loads((path/"result.json").read_text())
    progress = json.loads((path/"progress.json").read_text())
    if (result["status"] != "completed" or result["protocol_sha256"] != receipt["sha256"]["protocol.json"]
            or result["particles_sha256"] != receipt["sha256"]["particles.npz"]
            or result["observations"] != protocol["observations"]):
        raise ValueError("completed result disagrees with its frozen inputs")
    stages = progress["stages"]
    if (len(stages) != protocol["stages"]
            or [stage["stage"] for stage in stages] != list(range(1, protocol["stages"]+1))
            or not np.array_equal([stage["beta"] for stage in stages], protocol["betas"][1:])
            or stages[-1]["log_normalizer"] != result["log_normalizer"]):
        raise ValueError("completed stage trace is inconsistent")
    if all("acceptance" in stage for stage in stages):
        for block, totals in result["acceptance"].items():
            for counter in ("attempted", "accepted"):
                if not np.array_equal(np.sum([stage["acceptance"][block][counter] for stage in stages], axis=0), totals[counter]):
                    raise ValueError("stage acceptance counters disagree with final totals")
    with np.load(path/"particles.npz", allow_pickle=False) as stored:
        coordinates = stored["coordinates"]
    if coordinates.shape != (len(protocol["observations"]), protocol["particles"], 10):
        raise ValueError("coordinate array disagrees with protocol")
    return dict(path=str(path), protocol=protocol, result=result, stages=stages, coordinates=coordinates)


def coordinate_summary(x):
    if not np.isfinite(x).all():
        raise ValueError("nonfinite surviving particle coordinates")
    sed = softmax(np.column_stack((x[:, 2:6], np.zeros(len(x)))), axis=1)
    extras = {"redshift": np.exp(x[:, 0]), "circularized_radius": np.exp(x[:, 6]),
              "phase_sine": np.sin(x[:, 9]), "phase_cosine": np.cos(x[:, 9]),
              **{f"sed_fraction{i}": sed[:, i] for i in range(5)}}
    features = {**{name: x[:, i] for i, name in enumerate(COORDINATES)}, **extras}
    complex_mean = np.exp(1j*x[:, 9]).mean()
    return dict(features={name: dict(mean=float(value.mean()), sd=float(value.std(ddof=1)),
                                      quantiles=np.quantile(value, [.1, .5, .9]).tolist())
                          for name, value in features.items()},
                phase=dict(circular_mean=float(np.angle(complex_mean) % (2*np.pi)),
                           resultant=float(abs(complex_mean)), circular_variance=float(1-abs(complex_mean))))


def analyze_group(name, records, partial):
    runs = [read_run(record) for record in records]
    first = runs[0]["protocol"]
    keys = ["observations", "particles", "stages", "moves", "betas", "reference_protocol_sha256", "observation_sha256"]
    if any(any(run["protocol"][key] != first[key] for key in keys) for run in runs):
        raise ValueError("different settings or target within one replicate group")
    seeds = [run["protocol"]["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("replicate RNG seeds are not independent")
    betas = np.asarray(first["betas"])
    logz = np.array([[[(-np.inf if value is None else value) for value in stage["log_normalizer"]]
                      for stage in run["stages"]] for run in runs], dtype=float)
    logz = np.concatenate([np.zeros((len(runs), 1, logz.shape[2])), logz], axis=1)
    result = dict(name=name, partial_reference=partial, completed_replicas=len(runs), seeds=seeds,
                  particles=first["particles"], stages=first["stages"], moves=first["moves"],
                  beta_boundaries=BETA_BOUNDARIES, observations=[],
                  sampling_seconds=summarize_distribution([run["result"]["timing_seconds"]["sampling"] for run in runs]))
    indices = [int(np.searchsorted(betas, edge, side="right")-1) for edge in BETA_BOUNDARIES]
    for oi, oid in enumerate(first["observations"]):
        final = logz[:, -1, oi]
        cumulative = logz[:, :, oi]
        finite_runs = np.isfinite(final)
        item = dict(observation=oid, final_log_normalizer_by_run=finite_list(final),
                    final_log_normalizer_sd=float(final[finite_runs].std(ddof=1)) if finite_runs.sum()>1 else None,
                    final_normalizer=normalizer_summary(final), stage_trace=[], variance_windows=[],
                    initial_valid=[run["result"]["initial_valid"][oi] for run in runs], coordinate_runs=[])
        item["initial_valid_vs_final_logZ"] = association(item["initial_valid"], final)
        item["all_stage_aggregate_acceptance"] = summarize_acceptance(
            [[run["result"]] for run in runs], oi, first["particles"])["acceptance"]
        for si, beta in enumerate(betas[1:]):
            stage_values = cumulative[:, si+1]
            phases = [run["stages"][si] for run in runs]
            acceptance = summarize_acceptance([[stage] for stage in phases], oi, first["particles"])
            item["stage_trace"].append(dict(stage=si+1, beta=float(beta),
                cumulative_logZ_sd=summarize_distribution(stage_values).get("sd"),
                cumulative_logZ_vs_final=association(stage_values, final),
                incremental_logZ_sd=summarize_distribution(stage_values-cumulative[:, si]).get("sd"),
                normalizer=normalizer_summary(stage_values),
                pre_resampling_ess=summarize_distribution([stage["pre_resampling_ess"][oi] for stage in phases]),
                initial_ancestor_ess=summarize_distribution([stage["initial_ancestor_ess"][oi] for stage in phases]),
                distinct_initial_ancestors=summarize_distribution([stage["distinct_initial_ancestors"][oi] for stage in phases]),
                acceptance=acceptance["acceptance"], acceptance_available_by_run=acceptance["available_by_run"]))
        increments = np.column_stack([cumulative[finite_runs, right]-cumulative[finite_runs, left]
                                      for left, right in zip(indices[:-1], indices[1:])])
        covariance = np.cov(increments, rowvar=False, ddof=1) if finite_runs.sum()>1 else None
        variance = np.var(final[finite_runs], ddof=1) if finite_runs.sum()>1 else None
        item["window_increment_covariance_survivors_only"] = covariance.tolist() if covariance is not None else None
        for wi, (left, right) in enumerate(zip(indices[:-1], indices[1:])):
            window = dict(stage_start_exclusive=left, stage_end_inclusive=right,
                          beta_start=float(betas[left]), beta_end=float(betas[right]),
                          logZ_increment_by_run=finite_list(cumulative[:, right]-cumulative[:, left]),
                          increment_vs_final=association(cumulative[:, right]-cumulative[:, left], final),
                          covariance_share_of_final_logZ_variance=float(covariance[wi].sum()/variance) if variance else None)
            acceptance = summarize_acceptance([run["stages"][left:right] for run in runs], oi, first["particles"])
            window.update(mutation_acceptance=acceptance["acceptance"],
                          acceptance_available_by_run=acceptance["available_by_run"])
            item["variance_windows"].append(window)
        for ri, run in enumerate(runs):
            item["coordinate_runs"].append(dict(run=Path(run["path"]).name, seed=seeds[ri],
                log_normalizer=float(final[ri]) if finite_runs[ri] else None,
                summary=coordinate_summary(run["coordinates"][oi]) if finite_runs[ri] else None))
        feature_names = list(next(record["summary"]["features"] for record in item["coordinate_runs"] if record["summary"] is not None)) if finite_runs.any() else []
        associations = {}
        for feature in feature_names:
            feature_means = np.array([record["summary"]["features"][feature]["mean"] if record["summary"] is not None else np.nan for record in item["coordinate_runs"]])
            feature_sd = np.array([record["summary"]["features"][feature]["sd"] if record["summary"] is not None else np.nan for record in item["coordinate_runs"]])
            between = float(np.nanstd(feature_means, ddof=1)) if finite_runs.sum()>1 else None
            within = float(np.sqrt(np.nanmean(feature_sd**2))) if finite_runs.any() else None
            associations[feature] = dict(mean_by_run=finite_list(feature_means), within_run_sd_by_run=finite_list(feature_sd),
                between_run_mean_sd=between, rms_within_run_sd=within,
                between_mean_sd_over_within_sd=between/within if between is not None and within else None,
                relation_to_final_logZ=association(feature_means, final))
        item["coordinate_dispersion_and_logZ_associations"] = associations
        result["observations"].append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest["script_sha256"] != digest(__file__):
        raise ValueError("analysis script differs from the frozen manifest")
    result = dict(manifest=str(args.manifest.resolve()), manifest_sha256=digest(args.manifest),
        scope="Descriptive independent-run diagnostics; no causal attribution, posterior convergence claim, or independent-particle errors.",
        variance_allocation="cov(window_logZ_increment, final_logZ)/var(final_logZ); sums to1, may be negative, includes cross-window covariance; not causal variance contribution.",
        genealogy_warning="Accepted independence moves can replace coordinates while retaining ancestor labels; low ancestry ESS alone does not identify dependence on initial q0 draws.",
        partial_reference_at_freeze=manifest["partial_reference_at_freeze"], groups=[])
    for name, records in manifest["groups"].items():
        if records:
            result["groups"].append(analyze_group(name, records, name == "v6v100_partial"))
            print("STAGE_DIAGNOSIS", name, len(records), flush=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("STAGE_DIAGNOSIS_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
