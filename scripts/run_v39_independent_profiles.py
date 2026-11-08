"""Scene-disjoint prior comparison with fixed model mocks and selected likelihood grids."""

import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

from v39_group_response import GroupResponseMemo

ROOT = Path(os.environ["SBSI_INDEPENDENT_ROOT"])
SOURCE = Path(os.environ["SBSI_COHORT_RECOVERY_ROOT"])
sys.path.insert(0, str(SOURCE))
import run_v39_cohort_recovery as cohort  # noqa: E402

domain, study, tail = cohort.domain, cohort.study, cohort.tail
np, pd, torch, save = study.np, study.pd, study.torch, cohort.save


def protocol():
    p = study.read(ROOT / "protocol.json")
    study.verify(__file__, p["script_sha256"])
    for name, sha in p["sources"].items():
        study.verify(name, sha)
    return p


def prepare():
    p = protocol()
    src = study.ROOT
    support = study.read(src / "support/manifest.json")
    for name, sha in support["output_sha256"].items():
        study.verify(src / "support" / name, sha)
    offsets = np.r_[0, np.cumsum([r["n_rows"] for r in support["receipts"]])]
    boundary = int(offsets[10])
    weights = np.load(src / "support/weights.npy")
    assert len(weights) == 20231221 and len(offsets) == 21
    scene_cases = []
    for rec in support["receipts"]:
        source = Path(rec["source_record"]["source_manifest"])
        study.verify(source, rec["source_record"]["source_manifest_sha256"])
        scene_cases.append(study.read(source)["cases"])
    a_cases = sum(scene_cases[:10], [])
    b_cases = sum(scene_cases[10:], [])
    assert len(a_cases) == len(b_cases) == 100 and not set(a_cases) & set(b_cases)
    ranges = [[0, boundary], [boundary, len(weights)]]
    masses = [float(weights[lo:hi].sum()) for lo, hi in ranges]
    np.testing.assert_allclose(sum(masses), 1.0, rtol=0, atol=1.0e-12)
    save(
        ROOT / "priors.json",
        dict(
            ranges=ranges,
            masses=masses,
            cases=[a_cases, b_cases],
            source_support_sha256=study.file_hash(src / "support/manifest.json"),
            definition="Normalize unchanged inverse-inclusion masses separately within each disjoint100-scene realization; same empirical parent source, no new bright extension.",
        ),
    )
    mock = study.MockCatalogue.load(src / "mock")
    for name, sha in study.read(src / "mock/manifest.json")["output_sha256"].items():
        study.verify(src / "mock" / name, sha)
    ids = np.flatnonzero(mock.truth.scene_row.to_numpy() < boundary)[: p["n_observations"]]
    assert len(ids) == p["n_observations"] and (mock.truth.scene_row.iloc[ids] < boundary).all()
    fresh, _ = cohort.parent.observations()
    freshids = np.arange(p["n_observations"])
    assert not fresh.truth.numerical_padding.iloc[freshids].any()
    records = []
    for label, sample, selected in [("prior_a_mock", mock, ids), ("fresh_parent_mock", fresh, freshids)]:
        for group in np.unique(selected // 4):
            rows = np.arange(4 * group, 4 * group + 4)
            w = cohort.window(sample, rows)
            valid = np.isin(rows, selected)
            mag = 30 - 2.5 * np.log10(w.measurements.measured_flux_from_mag_auto.to_numpy())
            assert (mag < 25.8).all() and (w.measurements.measured_flux_radius > 3).all()
            out = ROOT / f"groups/group_{len(records):03d}" / "input"
            w.save(out)
            records.append(
                dict(
                    index=len(records),
                    sample=label,
                    rows=rows.tolist(),
                    scientific=valid.tolist(),
                    measured_magnitudes=mag.tolist(),
                    generating_atoms=w.truth.scene_row.to_numpy(int).tolist()
                    if label == "prior_a_mock"
                    else None,
                    input_sha256={f.name: study.file_hash(f) for f in out.glob("*.parquet")},
                )
            )
    for label in ["prior_a_mock", "fresh_parent_mock"]:
        assert sum(sum(r["scientific"]) for r in records if r["sample"] == label) == p["n_observations"]
    save(
        ROOT / "cohort.json",
        dict(
            protocol_sha256=study.file_hash(ROOT / "protocol.json"),
            groups=records,
            prior_a_selected_rows=ids.tolist(),
            fresh_selected_rows=freshids.tolist(),
            interpretation="First128 selected draws whose generating scene is in A are exact conditional draws from prior A; retain original four-row density groups and exclude companions from scientific sums. Fresh-parent inputs were selected only once.",
        ),
    )
    # Independent weighted-prior draws for each normalizer replicate; all nested budgets share prefixes.
    zero = np.load(src / "support/zero.npy", mmap_mode="r")
    for worker, (arm, rep) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        lo, hi = ranges[arm]
        cdf = np.cumsum(weights[lo:hi])
        cdf /= cdf[-1]
        rng = np.random.default_rng(np.random.SeedSequence([p["normalization_atom_seed"], arm, rep]))
        atoms = lo + np.searchsorted(cdf, rng.random(max(p["normalization_atoms"])), side="right")
        folder = ROOT / f"normalization/worker_{worker}"
        folder.mkdir(parents=True)
        np.save(folder / "atoms.npy", atoms)
        np.save(folder / "zero.npy", zero[atoms])
        save(
            folder / "input.json",
            dict(
                arm=arm,
                replicate=rep,
                output_sha256={f.name: study.file_hash(f) for f in folder.glob("*.npy")},
            ),
        )
    print("INDEPENDENT_PREPARED", dict(ranges=ranges, masses=masses, groups=len(records)), flush=True)


def normalize(worker):
    p = protocol()
    folder = ROOT / f"normalization/worker_{worker}"
    rec = study.read(folder / "input.json")
    for name, sha in rec["output_sha256"].items():
        study.verify(folder / name, sha)
    frame = pd.DataFrame(np.load(folder / "zero.npy"), columns=study.FEATURES)
    support = study.read(study.ROOT / "support/manifest.json")
    flow = study.load_measurement_model(study.V39_LIKE.flow_checkpoints[0], device="cuda")
    detector = study.load_detection_classifier(study.V39_LIKE, device="cuda")
    cache = study.V39CatalogueCache(
        frame, np.ones(len(frame)) / len(frame), support["receipts"][0]["conditions"], detector=detector
    )
    cut = study.output_cut(flow)
    masses = np.zeros((len(p["points"]), 2, 2))
    for pi, point in enumerate(p["points"]):
        view = cache.get(*point)
        seed = p["normalization_flow_seed"] + rec["replicate"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        values = np.zeros((len(frame), 2))
        for start in range(0, len(frame), 8192):
            stop = min(start + 8192, len(frame))
            draw = flow.sample(view.flow.iloc[start:stop], n_samples=64, batch_size=8192, qmc=True)
            passed = np.asarray(cut(draw), bool)
            for di, n in enumerate(p["normalization_draws"]):
                values[start:stop, di] = passed[:, :n].mean(1) * view.detection_probability[start:stop]
        for ni, n in enumerate(p["normalization_atoms"]):
            masses[pi, ni] = values[:n].mean(0)
        cache.discard_views([point])
        save(folder / "progress.json", dict(points=pi + 1, total_points=len(p["points"])))
        print("INDEPENDENT_NORMALIZATION", worker, pi + 1, len(p["points"]), flush=True)
    np.save(folder / "masses.npy", masses)
    save(
        folder / "manifest.json",
        dict(
            status="complete",
            protocol_sha256=study.file_hash(ROOT / "protocol.json"),
            input=rec,
            masses_sha256=study.file_hash(folder / "masses.npy"),
        ),
    )
    del cache, flow, detector
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_group(group, likelihood, *, preflight=False):
    input_root = ROOT / f"groups/group_{group['index']:03d}" / "input"
    sample = study.MockCatalogue.load(input_root)
    raw = sample.measurements[list(likelihood.target_names)].to_numpy(float)
    with GroupResponseMemo(likelihood, raw) as response:
        try:
            _evaluate_group(group, likelihood, response, preflight=preflight)
        except Exception:
            if response.failure is not None:
                np.savez_compressed(ROOT / f"failed_response_replay_{group['index']}.npz", **response.failure)
            raise


def full_block_terms(likelihood, observed, point, ids):
    """Keep the final atom block at the same density-batch shape as full blocks.

    Repeated padding atoms are evaluated but excluded before summing/ranking.
    Masking them would shorten the underlying density batch again.
    """
    assert ids.ndim == 1 and 0 < len(ids) <= 16384
    padded = np.resize(ids, 16384)
    return likelihood.log_importance_weights_tensor(
        observed,
        *point,
        atom_indices=padded,
        proposal_probability=np.ones(len(padded)),
        object_chunk=4,
        atom_chunk=4096,
    )[:, : len(ids)]


def _evaluate_group(group, likelihood, response, *, preflight=False):
    started = time.monotonic()
    p = protocol()
    if preflight:
        p = dict(p, points=p["preflight_points"])
    priors = study.read(ROOT / "priors.json")
    input_root = ROOT / f"groups/group_{group['index']:03d}" / "input"
    out = ROOT / f"{'preflight' if preflight else 'groups'}/group_{group['index']:03d}"
    out.mkdir(parents=True, exist_ok=True)
    for name, sha in group["input_sha256"].items():
        study.verify(input_root / name, sha)
    sample = study.MockCatalogue.load(input_root)
    n = len(likelihood.cache.prior.weights)
    k = p["k"]
    center = p["center"]
    logs = np.empty((4, n))
    best = torch.full((2, 4, k), -torch.inf, device="cuda", dtype=torch.float64)
    bestids = torch.full((2, 4, k), -1, device="cuda", dtype=torch.long)
    for start in range(0, n, 16384):
        ids = np.arange(start, min(start + 16384, n))
        terms = full_block_terms(likelihood, sample.measurements, center, ids)
        assert not torch.isnan(terms).any() and not torch.isposinf(terms).any()
        logs[:, ids] = terms.cpu().numpy()
        for arm, (lo, hi) in enumerate(priors["ranges"]):
            valid = (ids >= lo) & (ids < hi)
            if not valid.any():
                continue
            atomids = torch.as_tensor(ids[valid], device="cuda").expand(4, -1)
            joined = torch.cat([bestids[arm], atomids], dim=1)
            values, positions = torch.topk(torch.cat([best[arm], terms[:, valid]], dim=1), k, dim=1)
            best[arm] = values
            bestids[arm] = joined.gather(1, positions)
        if start // 16384 % 64 == 0:
            save(
                out / "progress.json",
                dict(
                    stage="exact_center",
                    atoms=int(ids[-1]) + 1,
                    total_atoms=n,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            print("INDEPENDENT_CENTER", group["index"], int(ids[-1]) + 1, n, flush=True)
    atoms = bestids.cpu().numpy()
    topvalues = best.cpu().numpy()
    assert (atoms >= 0).all() and np.isfinite(topvalues).all()
    center_seconds = time.monotonic() - started
    np.savez_compressed(out / "center_candidates.npz", atoms=atoms, log_contribution=topvalues)
    exact = np.empty((2, 4))
    distributions = []
    for arm, (lo, hi) in enumerate(priors["ranges"]):
        weights = np.zeros(n)
        weights[lo:hi] = likelihood.cache.prior.weights[lo:hi]
        exact[arm] = np.logaddexp.reduce(logs[:, lo:hi], axis=1)
        arm_distributions = []
        for j in range(4):
            masked = np.full(n, -np.inf)
            masked[lo:hi] = logs[j, lo:hi]
            arm_distributions.append(tail.build_tail_distribution(masked, weights, atoms[arm, j], delta=0.1))
        distributions.append(arm_distributions)
    del logs
    sampled = {}
    for arm in range(2):
        for seed in p["proposal_seeds"]:
            samples = [
                tail.draw_tail(
                    distributions[arm][j],
                    n_draws=max(p["draw_ladder"]),
                    seed=seed,
                    object_id=group["index"] * 4 + j,
                )
                for j in range(4)
            ]
            sampled[arm, seed] = (np.stack([v[0] for v in samples]), np.stack([v[1] for v in samples]))
    del distributions
    values = np.empty((2, len(p["proposal_seeds"]), len(p["draw_ladder"]), len(p["points"]), 4))
    candidate_values = np.empty((2, len(p["points"]), 4))
    zero = p["points"].index(center)
    for pi, point in enumerate(p["points"]):
        # Every candidate and tail set must exactly replay the original response
        # calculation at the centre; the density replay remains at 1e-7.
        response.replay = pi == zero
        for arm in range(2):
            v = (
                likelihood.log_importance_weights_tensor(
                    sample.measurements,
                    *point,
                    atom_indices=atoms[arm],
                    proposal_probability=np.ones_like(atoms[arm], float),
                    object_chunk=4,
                    atom_chunk=4096,
                )
                .cpu()
                .numpy()
            )
            if pi == zero:
                delta = abs(v - topvalues[arm])
                if np.any(delta > 1.0e-7):
                    np.savez_compressed(
                        out / f"failed_center_replay_{arm}.npz",
                        atoms=atoms[arm],
                        reference=topvalues[arm],
                        actual=v,
                    )
                np.testing.assert_allclose(v, topvalues[arm], rtol=0, atol=1.0e-7)
            fixed = np.logaddexp.reduce(v, axis=1)
            candidate_values[arm, pi] = fixed
            for si, seed in enumerate(p["proposal_seeds"]):
                draws, q = sampled[arm, seed]
                v = (
                    likelihood.log_importance_weights_tensor(
                        sample.measurements,
                        *point,
                        atom_indices=draws,
                        proposal_probability=q,
                        object_chunk=4,
                        atom_chunk=4096,
                    )
                    .cpu()
                    .numpy()
                )
                for di, m in enumerate(p["draw_ladder"]):
                    estimate = np.logaddexp(fixed, np.logaddexp.reduce(v[:, :m], axis=1) - np.log(m))
                    values[arm, si, di, pi] = estimate - np.log(priors["masses"][arm])
        save(
            out / "progress.json",
            dict(
                stage="profile",
                points=pi + 1,
                total_points=len(p["points"]),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        print("INDEPENDENT_PROFILE", group["index"], pi + 1, len(p["points"]), flush=True)
    gate = None
    response.replay = False
    if preflight:
        oracle = np.full((2, len(p["points"]), 4), -np.inf)
        oracle[:, zero] = exact - np.log(priors["masses"])[:, None]
        for pi, point in enumerate(p["points"]):
            if pi == zero:
                continue
            for start in range(0, n, 16384):
                ids = np.arange(start, min(start + 16384, n))
                v = full_block_terms(likelihood, sample.measurements, point, ids).cpu().numpy()
                for arm, (lo, hi) in enumerate(priors["ranges"]):
                    active = (ids >= lo) & (ids < hi)
                    if active.any():
                        oracle[arm, pi] = np.logaddexp(
                            oracle[arm, pi],
                            np.logaddexp.reduce(v[:, active], axis=1) - np.log(priors["masses"][arm]),
                        )
            print("INDEPENDENT_PREFLIGHT_ORACLE", group["index"], pi, flush=True)
        relative = values - values[:, :, :, zero : zero + 1, :]
        exact_relative = oracle - oracle[:, zero : zero + 1, :]
        err = relative[:, :, 1:] - exact_relative[:, None, None]
        gate = dict(
            maximum_relative_log_numerator_error=float(abs(err).max()),
            tolerance=p["preflight_log_tolerance"],
            passed=bool(np.all(abs(err) <= p["preflight_log_tolerance"])),
        )
        save(out / "numerical_gate.json", gate)
        np.save(out / "oracle.npy", oracle)
    np.savez_compressed(
        out / "profile.npz",
        log_numerator=values,
        exact_center_log_numerator=exact - np.log(priors["masses"])[:, None],
        log_candidate_numerator=candidate_values - np.log(priors["masses"])[:, None, None],
        candidate_atoms=atoms,
    )
    save(
        out / "manifest.json",
        dict(
            status="complete",
            protocol_sha256=study.file_hash(ROOT / "protocol.json"),
            group=group,
            profile_sha256=study.file_hash(out / "profile.npz"),
            center_candidate_replay_passed=True,
            preflight=preflight,
            numerical_gate=gate,
            response_replay_calls=response.replay_calls,
            response_replay_tolerance=0.0,
            exact_center_seconds=center_seconds,
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    print("INDEPENDENT_GROUP_COMPLETE", group["index"], flush=True)


def load_likelihood():
    likelihood, proposal, _, _ = study.load_inference()
    del proposal
    center = protocol()["center"]
    reference = likelihood.cache.get(*center).detection_probability.copy()
    likelihood.cache.probabilities = None
    likelihood.cache.detector = study.load_detection_classifier(study.V39_LIKE, device="cuda")
    likelihood.cache.discard_views(list(likelihood.cache._views))
    np.testing.assert_allclose(
        likelihood.cache.get(*center).detection_probability, reference, rtol=0, atol=1.0e-7
    )
    likelihood.compile_precision_density()
    return likelihood


def preflight():
    protocol()
    groups = study.read(ROOT / "cohort.json")["groups"]
    chosen = [
        next(g for g in groups if g["sample"] == sample) for sample in ["prior_a_mock", "fresh_parent_mock"]
    ]
    selected = int(os.environ["SLURM_ARRAY_TASK_ID"])
    assert selected in (0, 1)
    likelihood = load_likelihood()
    evaluate_group(chosen[selected], likelihood, preflight=True)
    gate = study.read(ROOT / f"preflight/group_{chosen[selected]['index']:03d}/numerical_gate.json")
    assert gate["passed"], "Profile integration gate failed; do not run broad grids."


def preflight_combine():
    p = protocol()
    groups = study.read(ROOT / "cohort.json")["groups"]
    chosen = [
        next(g for g in groups if g["sample"] == sample) for sample in ["prior_a_mock", "fresh_parent_mock"]
    ]
    assert study.read(ROOT / "replay.json")["status"] == "passed"
    for group in chosen:
        record = study.read(ROOT / f"preflight/group_{group['index']:03d}/manifest.json")
        assert record["status"] == "complete" and record["protocol_sha256"] == study.file_hash(
            ROOT / "protocol.json"
        )
    gates = [study.read(ROOT / f"preflight/group_{g['index']:03d}/numerical_gate.json") for g in chosen]
    save(
        ROOT / "preflight.json",
        dict(
            protocol_sha256=study.file_hash(ROOT / "protocol.json"),
            status="passed" if all(g["passed"] for g in gates) else "failed_numerical_gate",
            groups=[g["index"] for g in chosen],
            gates=gates,
            points=p["preflight_points"],
        ),
    )
    assert all(g["passed"] for g in gates), "Profile integration gate failed; do not run broad grids."


def replay():
    """Bounded numerical probe before repeating any full-centre scans."""
    p = protocol()
    groups = study.read(ROOT / "cohort.json")["groups"]
    likelihood = load_likelihood()
    n = len(likelihood.cache.prior.weights)
    starts = [0, (n // 2 // 16384) * 16384, (n // 16384) * 16384]
    records = []
    for label in ["prior_a_mock", "fresh_parent_mock"]:
        group = next(g for g in groups if g["sample"] == label)
        sample = study.MockCatalogue.load(ROOT / f"groups/group_{group['index']:03d}/input")
        raw = sample.measurements[list(likelihood.target_names)].to_numpy(float)
        original = likelihood._coefficients
        reference = np.full((4, n), np.nan)
        queries = []
        rng = np.random.default_rng(2026092829)
        with GroupResponseMemo(likelihood, raw) as memo:
            for start in starts:
                ids = np.arange(start, min(start + 16384, n))
                base = full_block_terms(likelihood, sample.measurements, p["center"], ids).cpu().numpy()
                reference[:, ids] = base
                padded = np.resize(ids, 16384)
                shuffled = np.stack([rng.permutation(padded) for _ in range(4)])
                queries.append((str(start), shuffled))
            pool = np.concatenate([np.arange(start, min(start + 16384, n)) for start in starts])
            queries.append(
                ("mixed_subsets", np.stack([rng.choice(pool, 16384, replace=False) for _ in range(4)]))
            )
            queries.append(
                ("mixed_duplicates", np.stack([rng.choice(pool, 16384, replace=True) for _ in range(4)]))
            )
            for label_query, shuffled in queries:
                valid = np.ones_like(shuffled, bool)
                memo_coeff = memo(shuffled, raw, valid)
                original_coeff = original(shuffled, raw, valid)
                actual = (
                    likelihood.log_importance_weights_tensor(
                        sample.measurements,
                        *p["center"],
                        atom_indices=shuffled,
                        proposal_probability=np.ones_like(shuffled, float),
                        object_chunk=4,
                        atom_chunk=4096,
                    )
                    .cpu()
                    .numpy()
                )
                expected = np.take_along_axis(reference, shuffled, axis=1)
                finite = np.isfinite(expected) & np.isfinite(actual)
                same_finite = bool(np.array_equal(np.isfinite(expected), np.isfinite(actual)))
                coefficient_error = float(abs(memo_coeff - original_coeff).max())
                error = float(abs(actual[finite] - expected[finite]).max(initial=0))
                passed = coefficient_error == 0 and error <= 1.0e-7 and same_finite
                record = dict(
                    sample=label,
                    query=label_query,
                    coefficient_error=coefficient_error,
                    density_error=error,
                    same_finite=same_finite,
                    passed=passed,
                )
                records.append(record)
                save(ROOT / "replay.json", dict(status="running", records=records))
                if not passed:
                    np.savez_compressed(
                        ROOT / f"replay_{label}_{label_query}.npz",
                        atoms=shuffled,
                        coefficients=memo_coeff,
                        original_coefficients=original_coeff,
                        expected=expected,
                        actual=actual,
                    )
                print("INDEPENDENT_REPLAY", record, flush=True)
    passed = all(r["passed"] for r in records)
    save(ROOT / "replay.json", dict(status="passed" if passed else "failed", records=records))
    assert passed, "Reorder/response replay failed; keep broad profile jobs paused"


def worker(index):
    protocol()
    pre = study.read(ROOT / "preflight.json")
    assert pre["status"] == "passed" and pre["protocol_sha256"] == study.file_hash(ROOT / "protocol.json")
    normalize(index)
    likelihood = load_likelihood()
    for group in study.read(ROOT / "cohort.json")["groups"][index::4]:
        evaluate_group(group, likelihood)
    print("INDEPENDENT_WORKER_COMPLETE", index, flush=True)


def combine():
    p = protocol()
    zero = p["points"].index(p["center"])
    groups = study.read(ROOT / "cohort.json")["groups"]
    sums = {
        name: np.zeros((2, len(p["proposal_seeds"]), len(p["draw_ladder"]), len(p["points"])))
        for name in ["prior_a_mock", "fresh_parent_mock"]
    }
    center_errors = []
    for group in groups:
        out = ROOT / f"groups/group_{group['index']:03d}"
        rec = study.read(out / "manifest.json")
        assert rec["status"] == "complete" and rec["protocol_sha256"] == study.file_hash(
            ROOT / "protocol.json"
        )
        study.verify(out / "profile.npz", rec["profile_sha256"])
        valid = np.asarray(group["scientific"])
        with np.load(out / "profile.npz") as f:
            v = f["log_numerator"]
            sums[group["sample"]] += v[..., valid].sum(-1)
            center_errors.append(
                float(
                    abs(
                        v[:, :, :, zero, :][..., valid]
                        - f["exact_center_log_numerator"][:, None, None, :][..., valid]
                    ).max()
                )
            )
    norms = np.empty((2, 2, 2, 2, len(p["points"])))
    for worker in range(4):
        out = ROOT / f"normalization/worker_{worker}"
        rec = study.read(out / "manifest.json")
        assert rec["status"] == "complete" and rec["protocol_sha256"] == study.file_hash(
            ROOT / "protocol.json"
        )
        study.verify(out / "masses.npy", rec["masses_sha256"])
        mass = np.load(out / "masses.npy")
        assert (mass > 0).all()
        norms[rec["input"]["arm"], rec["input"]["replicate"]] = np.moveaxis(np.log(mass), 0, -1)
    reports = []
    for sample, raw in sums.items():
        for arm in range(2):
            numerator = raw[arm] - raw[arm, ..., zero, None]
            normal = norms[arm] - norms[arm, ..., zero, None]
            normal_primary = normal[:, -1, -1].mean(0)
            profiles = numerator - p["n_observations"] * normal_primary
            primary = profiles[:, -1].mean(0)
            # Include alternate near-modes from every numerator/normalizer
            # variant; a mode missed by the primary curve is not ignorable.
            alternatives = (
                numerator[:, :, None, None, None, :] - p["n_observations"] * normal[None, None, ...]
            )
            alternatives = alternatives.reshape(-1, len(p["points"]))
            relevant = np.any(
                alternatives >= alternatives.max(axis=-1, keepdims=True) - p["relevant_log_drop"], axis=0
            )
            integration_error = float(np.max(abs(profiles - primary)[..., relevant]))
            normalization_error = float(
                p["n_observations"] * np.max(abs(normal - normal_primary)[..., relevant])
            )
            peak = int(primary.argmax())
            point = p["points"][peak]
            boundary = any(
                abs(point[d] - p["center"][d]) >= max(abs(x) for x in p["grid_offsets"]) - 1.0e-12
                for d in range(2)
            )
            reports.append(
                dict(
                    sample=sample,
                    prior="AB"[arm],
                    n_observations=p["n_observations"],
                    centered_log_likelihood=primary.tolist(),
                    profiles=profiles.tolist(),
                    numerical_log_profile_max_difference=integration_error,
                    normalization_log_profile_max_difference=normalization_error,
                    numerical_pass=integration_error <= p["profile_log_tolerance"],
                    normalization_pass=normalization_error <= p["profile_log_tolerance"],
                    grid_peak=point,
                    peak_on_boundary=boundary,
                    status="coarse_grid_only_no_accepted_shear_estimate",
                )
            )
    save(
        ROOT / "result.json",
        dict(
            status="complete",
            protocol_sha256=study.file_hash(ROOT / "protocol.json"),
            points=p["points"],
            reports=reports,
            maximum_center_log_numerator_error=max(center_errors),
            limitation="Small model diagnostic; scene-disjoint finite priors share the empirical source population. Coarse two-dimensional grids require numerical gates and subsequent refinement; no0.3% or actual-image calibration claim.",
        ),
    )
    print("INDEPENDENT_COMPLETE", reports, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "replay", "preflight", "preflight_combine", "worker", "combine"]
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        worker(int(os.environ["SLURM_ARRAY_TASK_ID"])) if args.stage == "worker" else globals()[args.stage]()
    except Exception:
        save(
            ROOT / f"failure_{args.stage}_{os.environ.get('SLURM_ARRAY_TASK_ID', 'cpu')}.json",
            dict(traceback=traceback.format_exc()),
        )
        raise
