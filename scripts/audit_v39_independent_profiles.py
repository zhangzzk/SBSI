"""Audit saved independent-prior profiles and locate numerical sensitivity."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(root):
    out = root / "audit"
    out.mkdir(exist_ok=True)
    p, result = read(root / "protocol.json"), read(root / "result.json")
    identity = digest(root / "protocol.json")
    assert result["status"] == "complete" and result["protocol_sha256"] == identity
    points = np.asarray(p["points"])
    zero = p["points"].index(p["center"])
    groups = read(root / "cohort.json")["groups"]
    numerators = {s: [] for s in ("prior_a_mock", "fresh_parent_mock")}
    objects = {s: [] for s in numerators}
    norms = np.empty((2, 2, 2, 2, len(points)))
    for worker in range(4):
        folder = root / f"normalization/worker_{worker}"
        rec = read(folder / "manifest.json")
        assert rec["protocol_sha256"] == identity and rec["status"] == "complete"
        assert digest(folder / "masses.npy") == rec["masses_sha256"]
        norms[rec["input"]["arm"], rec["input"]["replicate"]] = np.moveaxis(
            np.log(np.load(folder / "masses.npy")), 0, -1
        )
    for group in groups:
        folder = root / f"groups/group_{group['index']:03d}"
        rec = read(folder / "manifest.json")
        assert rec["protocol_sha256"] == identity and rec["status"] == "complete"
        assert rec["center_candidate_replay_passed"] and rec["response_replay_calls"] == 6
        assert digest(folder / "profile.npz") == rec["profile_sha256"]
        with np.load(folder / "profile.npz") as f:
            values = f["log_numerator"]
        relative = values - values[:, :, :, zero : zero + 1, :]
        for slot, scientific in enumerate(group["scientific"]):
            if not scientific:
                continue
            numerators[group["sample"]].append(relative[..., slot])
            objects[group["sample"]].append(
                dict(
                    group=group["index"],
                    slot=slot,
                    row=group["rows"][slot],
                    magnitude=group["measured_magnitudes"][slot],
                )
            )
    all_reports, object_reports = [], {}
    primary_profiles = {}
    for sample, rows in numerators.items():
        data = np.stack(rows, axis=-1)
        assert data.shape[-1] == p["n_observations"]
        primary_rows = data[:, :, -1].mean(1)
        centered_norm = norms - norms[..., zero, None]
        primary_norm = centered_norm[:, :, -1, -1].mean(1)
        profiles = data.sum(-1) - p["n_observations"] * primary_norm[:, None, None, :]
        primary = profiles[:, :, -1].mean(1)
        primary_profiles[sample] = primary
        rows_report = []
        for j, obj in enumerate(objects[sample]):
            per_arm = []
            for arm in range(2):
                remainder = primary[arm] - primary_rows[arm, :, j] + primary_norm[arm]
                per_arm.append(
                    dict(
                        prior="AB"[arm],
                        max_draw_drift=float(abs(data[arm, :, :, :, j] - primary_rows[arm, :, j]).max()),
                        max_8k_16k_difference=float(abs(data[arm, :, 1, :, j] - data[arm, :, 2, :, j]).max()),
                        max_seed_difference_16k=float(
                            abs(data[arm, 0, 2, :, j] - data[arm, 1, 2, :, j]).max()
                        ),
                        peak_without_this_row=points[remainder.argmax()].tolist(),
                    )
                )
            rows_report.append(
                dict(
                    **obj,
                    arms=per_arm,
                    maximum_centered_prior_difference=float(
                        abs(primary_rows[0, :, j] - primary_rows[1, :, j]).max()
                    ),
                    prior_difference_at_A_peak=float(
                        primary_rows[0, primary[0].argmax(), j] - primary_rows[1, primary[0].argmax(), j]
                    ),
                    prior_difference_at_B_peak=float(
                        primary_rows[0, primary[1].argmax(), j] - primary_rows[1, primary[1].argmax(), j]
                    ),
                )
            )
        object_reports[sample] = rows_report
        for arm in range(2):
            expected = next(r for r in result["reports"] if r["sample"] == sample and r["prior"] == "AB"[arm])
            np.testing.assert_allclose(
                primary[arm], expected["centered_log_likelihood"], rtol=0, atol=1.0e-10
            )
            other_peak = int(primary[1 - arm].argmax())
            report = dict(
                sample=sample,
                prior="AB"[arm],
                peak=points[primary[arm].argmax()].tolist(),
                log_drop_at_other_prior_peak=float(primary[arm].max() - primary[arm, other_peak]),
                grid_peaks_all_draws=points[profiles[arm].argmax(-1)].tolist(),
                max_8k_16k_difference=float(abs(profiles[arm, :, 1] - profiles[arm, :, 2]).max()),
                max_seed_difference_16k=float(abs(profiles[arm, 0, 2] - profiles[arm, 1, 2]).max()),
                numerical_pass=expected["numerical_pass"],
                normalization_pass=expected["normalization_pass"],
            )
            all_reports.append(report)
        np.savez_compressed(
            out / f"{sample}_rows.npz",
            numerator=data,
            primary_rows=primary_rows,
            primary_normalization=primary_norm,
            points=points,
        )
    payload = dict(
        status="complete",
        source_protocol_sha256=identity,
        result_sha256=digest(root / "result.json"),
        n_verified_groups=len(groups),
        reports=all_reports,
        objects=object_reports,
        limitation="Leave-one-out curves are attribution diagnostics, never selection cuts or accepted estimates.",
    )
    (out / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True, sharey=True, constrained_layout=True)
    for row, sample in enumerate(numerators):
        for arm in range(2):
            ax = axes[row, arm]
            profile = primary_profiles[sample][arm]
            mesh = ax.pcolormesh(
                np.unique(points[:, 0]),
                np.unique(points[:, 1]),
                (profile - profile.max()).reshape(7, 7).T,
                vmin=-12,
                vmax=0,
                cmap="viridis",
                shading="nearest",
            )
            ax.scatter(
                *points[profile.argmax()], facecolors="none", edgecolors="white", s=110, linewidths=1.5
            )
            ax.scatter(0.02, 0.0, marker="x", color="#ff9d00", s=65, linewidths=2, label="Injected shear")
            passed = next(
                v["numerical_pass"] for v in all_reports if v["sample"] == sample and v["prior"] == "AB"[arm]
            )
            ax.set_title(
                f"{'A-generated' if row == 0 else 'Fresh-parent'} mocks · prior {'AB'[arm]}\nNumerical gate: {'pass' if passed else 'FAIL'}",
                fontsize=11,
            )
            if row == 1:
                ax.set_xlabel("Shear $g_1$")
            if arm == 0:
                ax.set_ylabel("Shear $g_2$")
    axes[0, 0].legend(loc="lower left", fontsize=9)
    fig.colorbar(mesh, ax=axes, label="Log likelihood − each arm's grid maximum", shrink=0.85)
    fig.suptitle("128 observations per cohort · coarse grid, no accepted shear estimate", fontsize=12)
    fig.savefig(out / "profiles.png", dpi=180)
    fig.savefig(out / "profiles.pdf")
    plt.close(fig)
    print(json.dumps(all_reports, indent=2))
    for sample, rows in object_reports.items():
        print(
            "LARGEST_DRIFT",
            sample,
            sorted(rows, key=lambda r: max(a["max_draw_drift"] for a in r["arms"]), reverse=True)[:5],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root)
