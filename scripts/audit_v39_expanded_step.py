"""Attribute the saved expanded-prior Newton failure without new inference."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def bins(magnitude):
    return {
        "mag_lt20": magnitude < 20,
        "mag20_22": (magnitude >= 20) & (magnitude < 22),
        "mag22_24": (magnitude >= 22) & (magnitude < 24),
        "mag_ge24": magnitude >= 24,
    }


def audit(root, output):
    comparison = read(root / "comparison.json")
    protocol = read(root / "protocol.json")
    execution = read(root / "execution.json")
    assert digest(root / "protocol.json") == execution["protocol_sha256"]
    assert digest(root / "run_v39_expanded_prior.py") == execution["driver_sha256"]
    matched = Path(protocol["matched_source"])
    generation = read(matched / "mock/manifest.json")
    actual = Path(read(matched / "protocol.json")["actual_root"])
    center = np.asarray(comparison["center"])
    selected = np.asarray(generation["selected_parent_rows"])
    assert len(selected) == 16314 and np.all(np.diff(selected) > 0)
    hashes = {}

    def verify(path, expected=None):
        value = digest(path)
        if expected is not None:
            assert value == expected, path
        hashes[str(path)] = value

    for path in [root / "comparison.json", root / "protocol.json",
                 root / "execution.json", matched / "mock/manifest.json"]:
        verify(path)
    for path, value in execution["sources"].items():
        verify(Path(path), value)
    for name in ["evaluation/measurements.parquet", "evaluation/truth.parquet"]:
        verify(matched / "mock" / name, generation["output_sha256"][name])
    truth = pd.read_parquet(matched / "mock/evaluation/truth.parquet")
    np.testing.assert_array_equal(truth.parent_sample_row, np.arange(20000))
    np.testing.assert_array_equal(np.flatnonzero(~truth.numerical_padding), selected)
    actual_manifest = read(actual / "input/manifest.json")
    verify(actual / "input/measurements.parquet",
           actual_manifest["output_sha256"]["measurements.parquet"])

    # Independently reconstruct derivatives from the nine saved mass nodes.
    normalization = read(root / "normalization.json")
    assembled = read(root / "assembled.json")
    verify(root / "normalization.json", assembled["output_sha256"]["normalization.json"])
    h = normalization["finite_difference_step"]
    mass = {tuple(np.round([x["g1"], x["g2"]], 14)):
            np.log(x["detected_and_selected_mass"]) for x in normalization["points"]}

    def log_mass(i, j):
        return mass[tuple(np.round(center + h * np.array([i, j]), 14))]

    gradient = np.array([(log_mass(1, 0) - log_mass(-1, 0)) / (2*h),
                         (log_mass(0, 1) - log_mass(0, -1)) / (2*h)])
    cross = (log_mass(1, 1) - log_mass(1, -1)
             - log_mass(-1, 1) + log_mass(-1, -1)) / (4*h*h)
    norm_info = np.array([
        [(log_mass(1, 0) - 2*log_mass(0, 0) + log_mass(-1, 0)) / h**2, cross],
        [cross, (log_mass(0, 1) - 2*log_mass(0, 0) + log_mass(0, -1)) / h**2]])
    np.testing.assert_allclose(gradient, comparison["normalization_gradient"], atol=1e-12)
    np.testing.assert_allclose(norm_info, comparison["normalization_hessian"], atol=1e-9)

    result = {}
    for sample in ["mock", "actual"]:
        measured_path = (matched / "mock/evaluation/measurements.parquet" if sample == "mock"
                         else actual / "input/measurements.parquet")
        measured = pd.read_parquet(measured_path)
        keep = selected if sample == "mock" else np.arange(20000)
        parts = []
        for worker in range(4):
            path = root / f"moments/{sample}/worker_{worker}.npz"
            receipt = read(path.with_suffix(".json"))
            assert receipt["status"] == "complete" and receipt["execution"] == execution
            verify(path, receipt["output_sha256"])
            with np.load(path) as data:
                np.testing.assert_array_equal(data["object_rows"],
                                              np.arange(worker*5000, (worker+1)*5000))
                np.testing.assert_array_equal(data["score"], data["ladder_score"][-1])
                np.testing.assert_array_equal(data["information"], data["ladder_information"][-1])
                parts.append({k: data[k] for k in ["ladder_score", "ladder_information"]})
        score, info = [np.concatenate([part[k] for part in parts], axis=1)[:, keep]
                       for k in ["ladder_score", "ladder_information"]]
        assert np.isfinite(score).all() and np.isfinite(info).all()
        np.testing.assert_allclose(info, info.swapaxes(-1, -2), atol=1e-10)
        mag = measured.measured_mag_auto.to_numpy()[keep]
        masks = bins(mag)
        sums, matrices = score.sum(1), info.sum(1)
        eigenvalues, eigenvectors = np.linalg.eigh(matrices[-1])
        direction = eigenvectors[:, 0]
        if direction[0] < 0:
            direction = -direction

        def describe_rows(values, count=12):
            order = np.argsort(np.abs(values))[::-1][:count]
            return [dict(parent_row=int(keep[i]), magnitude=float(mag[i]),
                         value=float(values[i])) for i in order]

        rungs = []
        for index, draws in enumerate(protocol["numerics"]["ladder"]):
            s, matrix = sums[index], matrices[index]
            recorded = comparison["rungs"][index]["mock" if sample == "mock" else "actual_all20k"]
            np.testing.assert_allclose(s, recorded["score_sum"], atol=1e-8, rtol=1e-12)
            np.testing.assert_allclose(matrix, recorded["information_sum"], atol=1e-7, rtol=1e-12)
            values, vectors = np.linalg.eigh(matrix)
            projected = np.einsum("i,nij,j->n", direction, info[index], direction)
            row = dict(draws=draws, score=s.tolist(), information=matrix.tolist(),
                       eigenvalues=values.tolist(), fixed_final_direction=direction.tolist(),
                       projected_information=float(projected.sum()),
                       positive_projected_sum=float(projected[projected > 0].sum()),
                       negative_projected_sum=float(projected[projected < 0].sum()),
                       magnitude_bins={name: dict(n=int(mask.sum()), score=score[index, mask].sum(0).tolist(),
                           projected_information=float(projected[mask].sum())) for name, mask in masks.items()})
            if values[0] > 0:
                step = np.linalg.solve(matrix, s)
                residual = score[index] - np.einsum("nij,j->ni", info[index], step)
                influence = np.linalg.solve(matrix, residual.T).T
                covariance = influence.T @ influence * len(keep)/(len(keep)-1)
                se = np.sqrt(np.diag(covariance))
                np.testing.assert_allclose(center + step, recorded["estimate"], atol=1e-12)
                np.testing.assert_allclose(se, recorded["robust_standard_error"], atol=1e-10)
                row.update(estimate=(center+step).tolist(), standard_error=se.tolist(),
                    score_in_eigenbasis=(vectors.T@s).tolist(),
                    step_in_eigenbasis=((vectors.T@s)/values).tolist(),
                    step_over_stencil=float(np.linalg.norm(step)/h),
                    inverse_information_standard_error=np.sqrt(np.diag(np.linalg.inv(matrix))).tolist(),
                    numerator_only_proposal_diagnostic=(center+np.linalg.solve(
                        matrix-len(keep)*norm_info, s+len(keep)*gradient)).tolist(),
                    largest_g1_influence=describe_rows(influence[:, 0]),
                    g1_variance_fraction_by_magnitude={name: float(np.sum(influence[mask, 0]**2)
                        / np.sum(influence[:, 0]**2)) for name, mask in masks.items()})
                for name, mask in masks.items():
                    row["magnitude_bins"][name]["g1_step_contribution"] = float(
                        np.linalg.solve(matrix, score[index, mask].sum(0))[0])
            rungs.append(row)

        changes = []
        for a, b in [(0, 4), (4, 5), (0, 5)]:
            ds, di = score[b]-score[a], info[b]-info[a]
            delta_info = np.einsum("i,nij,j->n", direction, di, direction)
            change = dict(first=rungs[a]["draws"], second=rungs[b]["draws"],
                score_change=(sums[b]-sums[a]).tolist(),
                information_change=(matrices[b]-matrices[a]).tolist(),
                net_projected_information_change=float(delta_info.sum()),
                absolute_projected_information_change=float(np.abs(delta_info).sum()))
            if min(np.linalg.eigvalsh(matrices[a])) > 0 and min(np.linalg.eigvalsh(matrices[b])) > 0:
                first = np.linalg.solve(matrices[a], sums[a])
                second = np.linalg.solve(matrices[b], sums[b])
                score_term = np.linalg.solve(matrices[b], ds.T).T
                info_term = -np.linalg.solve(matrices[b], np.einsum("nij,j->ni", di, first).T).T
                contribution = score_term + info_term
                np.testing.assert_allclose(contribution.sum(0), second-first, atol=1e-11, rtol=1e-10)
                change.update(step_change=(second-first).tolist(),
                    score_change_term=score_term.sum(0).tolist(),
                    information_change_term=info_term.sum(0).tolist(),
                    largest_g1_changes=describe_rows(contribution[:, 0]),
                    magnitude_bins={name: dict(n=int(mask.sum()),
                        g1_net_change=float(contribution[mask, 0].sum()),
                        g1_absolute_change=float(np.abs(contribution[mask, 0]).sum()))
                        for name, mask in masks.items()})
            changes.append(change)
        result[sample] = dict(n=len(keep), rungs=rungs, changes=changes)

    payload = dict(status="complete", source_hashes=hashes, script_sha256=digest(Path(__file__)),
        center=center.tolist(), normalization_gradient=gradient.tolist(),
        normalization_hessian=norm_info.tolist(), samples=result,
        limitations=["Saved finite-draw moments only; no new likelihood or accuracy oracle.",
            "Magnitude bins are contributions to the full-sample diagnostic, not cut-sample estimates.",
            "Numerator-only proposals omit selection on purpose for attribution, never for inference.",
            "Influence uncertainties are conditional local diagnostics, not calibrated shear errors.",
            "No observation, prior weight, model, seed or production default changed."])
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"n": value["n"], "rungs": [
        {key: row[key] for key in ["draws", "score", "eigenvalues", "positive_projected_sum",
            "negative_projected_sum", "magnitude_bins", "estimate", "standard_error",
            "g1_variance_fraction_by_magnitude", "numerator_only_proposal_diagnostic"] if key in row}
        for row in value["rungs"]], "changes": value["changes"]} for name, value in result.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
