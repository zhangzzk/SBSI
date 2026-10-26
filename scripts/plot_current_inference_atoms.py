"""Remake atom maps using a hash-matched runtime and an existing inference receipt.

The runtime is an explicit source snapshot, not this checkout's older sampler.
Run under Slurm: exact centre-node masses require scanning the complete prior.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def plot_row(payload, report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    measured, truth = payload["measured"], payload["truth"]
    kind, heavy_mass = payload["kind"], payload["heavy_mass"]
    true = report["observation_truth"]
    panels = [
        (measured[:, 0], measured[:, 1], "Predicted e₁ − observed e₁",
         "Predicted e₂ − observed e₂", (0, 0), "Proposal coordinates: shape"),
        (measured[:, 2], measured[:, 3], "Predicted radius − observed radius [arcsec]",
         "log₁₀(predicted flux / observed flux)", (0, 0), "Proposal coordinates: size and flux"),
        (truth[:, 0], truth[:, 1], "Intrinsic e₁", "Intrinsic e₂",
         (true["e1"], true["e2"]), "Same atoms: intrinsic shape"),
        (truth[:, 3], truth[:, 2], "Circularized effective radius [arcsec]",
         "True r magnitude", (true["circularized_Re"], true["r"]),
         "Same atoms: true size and magnitude"),
    ]
    norm = LogNorm(vmin=1e-4, vmax=1)
    for ax, (x, y, xlabel, ylabel, cross, title) in zip(axes.flat, panels):
        for code, color, size, alpha, marker in [
            (0, "#c8c8c8", 2, .18, "."),
            (1, "#0072B2", 8, .32, "x"),
            (2, "#009E73", 8, .55, "+"),
        ]:
            mask = kind == code
            ax.scatter(x[mask], y[mask], s=size, c=color, alpha=alpha,
                       marker=marker, linewidths=.5, rasterized=True)
        heavy = heavy_mass > 0
        # Paint the largest masses last across both marker classes, so many
        # nearby negligible atoms cannot cover the dominant atom's colour.
        for rank, index in enumerate(np.flatnonzero(heavy)[np.argsort(heavy_mass[heavy])]):
            ax.scatter([x[index]], [y[index]], c=[heavy_mass[index]], norm=norm,
                       cmap="plasma", marker="o" if payload["reached"][index] else "D",
                       s=55, edgecolors="black", linewidths=.65, zorder=6+rank*.01)
        ax.axvline(cross[0], color="#333333", ls="--", lw=.9)
        ax.axhline(cross[1], color="#333333", ls="--", lw=.9)
        for vals, setter in [(x, ax.set_xlim), (y, ax.set_ylim)]:
            selected = vals[(kind > 0) & np.isfinite(vals)]
            lo, hi = np.percentile(selected, [.5, 99.5])
            hvals = vals[heavy & np.isfinite(vals)]
            ci = cross[0] if setter == ax.set_xlim else cross[1]
            lo, hi = min(lo, hvals.min(), ci), max(hi, hvals.max(), ci)
            pad = max((hi-lo)*.07, .02)
            setter(lo-pad, hi+pad)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(False)
    axes[1, 1].invert_yaxis()
    fig.subplots_adjust(left=.09, right=.87, bottom=.18, top=.89,
                        hspace=.32, wspace=.25)
    cax = fig.add_axes([.90, .26, .017, .49])
    fig.colorbar(ScalarMappable(norm=norm, cmap="plasma"), cax=cax, extend="min",
                 label="Exact centre posterior mass (top 64 atoms)")
    legend = [
        Line2D([], [], marker=".", ls="", color="#aaa", label="Prior background (uniform subsample)"),
        Line2D([], [], marker="x", ls="", color="#0072B2", label="Unique sampled atoms outside exact stratum"),
        Line2D([], [], marker="+", ls="", color="#009E73", label="1,024 atoms summed exactly"),
        Line2D([], [], marker="o", ls="", color="#bf3984", markeredgecolor="black", label="Top-mass atom reached"),
        Line2D([], [], marker="D", ls="", color="#bf3984", markeredgecolor="black", label="Top-mass atom missed"),
        Line2D([], [], ls="--", color="#333", label="Observation (measured / intrinsic truth)"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.48, .035),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Observation {report['row']} · measured r = {report['observed_mag']:.2f}",
                 fontsize=15, y=.985)
    fig.text(.48, .953, "Latest tested proposal · 20.23m weighted prior · V3.6-like models",
             ha="center", fontsize=11)
    fig.text(.48, .925,
             f"16,384 draws → {report['unique_complement']:,} distinct complement atoms; "
             f"{report['wasted_draws']:,} draws hit the exact stratum",
             ha="center", fontsize=10)
    fig.text(.48, .015,
             f"Full-prior posterior mass reached: {report['posterior_reached']:.1%}  |  "
             f"Top 64 contain: {report['top64_mass']:.1%}  |  "
             f"Estimated / exact likelihood: {report['likelihood_ratio']:.3f}",
             ha="center", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"row_{report['row']}.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[142230, 3563, 409188])
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_root.resolve()))

    import pandas as pd
    import torch
    from scripts import run_disk_inference as driver
    from scripts.plot_proposal_atom_map import gather_truth, observation_truth
    from sbsi.catalogue_closure import MockCatalogue
    from sbsi.catalogue_disk_likelihood import DiskCatalogueLikelihood
    from sbsi.catalogue_disk_response import CatalogueDiskResponse
    from sbsi.catalogue_likelihood import CatalogueSelection, OutputCut
    from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
    from sbsi.crowding import FLOW_FEATURES
    from sbsi.disk_inference_store import (
        FrozenDiskCache, ShardedDiskResponse, file_hash, load_subset_manifest,
        subset_weights, write_json,
    )
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import V36_LIKE, load_disk_response

    torch.set_num_threads(4)
    start_time = time.monotonic()

    def log(message):
        print(f"{time.monotonic()-start_time:.1f}s {message}", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    reference_path = args.run / "ten_k_bright20_16631250/result.json"
    reference = json.loads(reference_path.read_text())
    for name, expected in reference["implementation_sha256"].items():
        if file_hash(args.runtime_root / name) != expected:
            raise ValueError(f"runtime differs from tested implementation: {name}")
    prepared_root = args.run / "disk_assembled20m"
    subset_path = args.run / "prior_subset20m_bright20/manifest.json"
    subset = load_subset_manifest(subset_path)
    current = driver.identity(SimpleNamespace(input=args.run / "input", subset=subset_path), subset)
    prepared = json.loads((prepared_root / "manifest.json").read_text())
    if current != reference["preparation_identity"] or prepared["identity"] != current:
        raise ValueError("preparation identity must exactly match the tested run")
    if prepared["status"] != "complete" or prepared["pilot_only"]:
        raise ValueError("complete full-prior preparation required")
    for name, expected in prepared["output_sha256"].items():
        if file_hash(prepared_root / name) != expected:
            raise ValueError(f"prepared array hash differs: {name}")
    log("Prepared identity and arrays verified")
    flow = load_measurement_model(V36_LIKE.flow_checkpoints[0], device="cuda")
    emulator = load_disk_response(V36_LIKE, device="cuda", nthread=4)
    responses = []
    for receipt in prepared["receipts"]:
        source = receipt["subset_receipt"]
        root = Path(source["root"])
        for name in ("pair_indptr.npy", "pair_features.npy", "prior_weight.npy", "flow_zero.parquet"):
            if file_hash(root / name) != source["output_sha256"][name]:
                raise ValueError(f"prior shard hash differs: {root / name}")
        ptr = np.load(root / "pair_indptr.npy", mmap_mode="r")[:receipt["n_rows"]+1]
        features = np.load(root / "pair_features.npy", mmap_mode="r")[:ptr[-1]]
        responses.append(CatalogueDiskResponse(ptr, features, emulator))
    response = ShardedDiskResponse(responses)
    zero = pd.DataFrame(np.load(prepared_root / "zero.npy", mmap_mode="r"), columns=FLOW_FEATURES)
    probability = np.load(prepared_root / "probability.npy", mmap_mode="r")
    cache = FrozenDiskCache(zero, dict(zip(map(tuple, prepared["points"]), probability)),
                            subset["conditions"], weights=subset_weights(subset))
    config = json.loads(driver.CONFIG.read_text())
    cut = OutputCut.from_specs(flow.target_transform.target_names,
                              specs=config["measured_selection"]["bounds"])
    likelihood = DiskCatalogueLikelihood(flow, cache, disk_response=response,
                                          selection=CatalogueSelection(cut))
    proposal_root = Path(reference["proposal_source"])
    if file_hash(proposal_root / "coordinates.npz") != reference["proposal_cache_sha256"]:
        raise ValueError("proposal coordinates differ from tested run")
    coords = ProposalCoordinateTable.load(proposal_root)
    coords = dataclasses.replace(coords, fractional_targets=tuple(reference["proposal_fractional_targets"]))
    if len(coords.values) != subset["n_rows"]:
        raise ValueError("proposal and prior atom counts disagree")
    proposal = DefensiveLocalProposal(coords, cache.prior.weights,
        local_base_weights=cache.get(0., 0.).detection_probability, score_dtype=torch.float64)
    mock = MockCatalogue.load(args.run / "input")
    mock_manifest = json.loads((args.run / "input/image_mock_manifest.json").read_text())
    center = current["center"]
    flow.compile_log_prob(mode=None, dynamic=True)
    n_atoms = subset["n_rows"]
    log(f"Loaded {n_atoms} atoms; CUDA {torch.cuda.get_device_name(0)}")
    reports = []
    for row in args.rows:
        observed = mock.measurements.iloc[[row]].reset_index(drop=True)
        raw = observed[list(coords.target_names)].to_numpy(float)[0]
        candidates = proposal.candidates(observed, n_candidates=1024,
                                          prefilter_candidates=131072, torch_device="cuda")
        draw = proposal.draw_tilted(candidates, observed, n_draws=16384, delta=.1,
                                    temperature=1., seed=8701, object_ids=np.array([row]),
                                    device="cuda")
        exact = candidates.indices[0]
        drawn_all = draw.indices[0]
        inside = np.isin(drawn_all, exact)
        sampled = np.unique(drawn_all[~inside])
        reached = np.union1d(exact, sampled)
        log(f"row={row} candidates={len(exact)} complement_unique={len(sampled)}")
        # This scan supplies actual posterior mass, not normalized mass only
        # among the sampled atoms. Keep all log terms for auditable replotting.
        terms_path = args.output / f"row_{row}_log_terms.npy"
        terms = np.lib.format.open_memmap(terms_path, mode="w+", dtype="float64", shape=(n_atoms,))
        log_total = -np.inf
        for lo in range(0, n_atoms, 65536):
            atoms = np.arange(lo, min(lo+65536, n_atoms))
            block = likelihood.log_importance_weights_tensor(
                observed, *center, atom_indices=atoms,
                proposal_probability=np.ones(len(atoms)), object_chunk=16)
            values = block[0].cpu().numpy()
            if np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError("invalid centre likelihood term")
            terms[lo:lo+len(atoms)] = values
            log_total = np.logaddexp(log_total, torch.logsumexp(block[0], 0).item())
            if lo == 0 or lo // 65536 % 32 == 31 or lo+len(atoms) == n_atoms:
                log(f"row={row} exact_center={lo+len(atoms)}/{n_atoms}")
        terms.flush()
        if not np.isfinite(log_total):
            raise ValueError("nonfinite total likelihood")
        top = np.argpartition(terms, -64)[-64:]
        top = top[np.argsort(terms[top])[::-1]]
        def posterior(ids):
            return np.exp(terms[ids]-log_total)
        top_mass = posterior(top)
        # The sample numerator uses the full 16384 budget, including zero
        # contributions from draws that hit the exact stratum.
        relative_weights = np.zeros(16384)
        relative_weights[~inside] = posterior(drawn_all[~inside]) / draw.probability[0, ~inside]
        likelihood_ratio = posterior(exact).sum() + relative_weights.mean()
        background = np.random.default_rng(20260921).choice(n_atoms, 30000, replace=False)
        background = background[~np.isin(background, np.union1d(reached, top))]
        ids = np.concatenate([background, sampled, exact, top])
        kind = np.repeat([0, 1, 2, 3], [len(background), len(sampled), len(exact), len(top)])
        truth = gather_truth(subset, ids)
        values = coords.values[ids]
        measured = values-raw
        measured[:, 2] *= subset["conditions"]["pixel_size"]
        measured[:, 3] = np.where(values[:, 3] > 0,
            np.log10(np.maximum(values[:, 3], np.finfo(float).tiny)/raw[3]), np.nan)
        heavy_mass = np.zeros(len(ids))
        heavy_mass[-64:] = top_mass
        payload = dict(atom_ids=ids, measured=measured, truth=truth, kind=kind,
                       heavy_mass=heavy_mass, reached=np.isin(ids, reached),
                       exact_atoms=exact, draw_atoms=drawn_all, draw_probability=draw.probability[0])
        np.savez_compressed(args.output / f"row_{row}_plot_data.npz", **payload)
        report = dict(row=row, n_atoms=n_atoms, center=center, seed=8701,
            n_draws=16384, n_candidates=1024, prefilter_candidates=131072,
            candidate_source="location_prefilter", unique_complement=len(sampled),
            wasted_draws=int(inside.sum()), observed=raw.tolist(),
            observed_mag=float(30-2.5*np.log10(raw[3])),
            observation_truth=observation_truth(mock_manifest, mock.truth.iloc[row]),
            posterior_reached=float(posterior(reached).sum()),
            posterior_exact_stratum=float(posterior(exact).sum()),
            posterior_sampled_complement=float(posterior(sampled).sum()),
            top64_mass=float(top_mass.sum()), top64_reached=int(np.isin(top, reached).sum()),
            likelihood_ratio=float(likelihood_ratio), log_numerator=float(log_total),
            complement_ess=float(relative_weights.sum()**2/np.square(relative_weights).sum()),
            proposal_fractional_targets=list(coords.fractional_targets),
            proposal_metadata=coords.metadata,
            nonpositive_plot_flux=int((values[:, 3] <= 0).sum()),
            terms_sha256=file_hash(terms_path),
            plot_data_sha256=file_hash(args.output / f"row_{row}_plot_data.npz"))
        write_json(args.output / f"row_{row}_report.json", report)
        plot_row(payload, report, args.output)
        reports.append(report)
        log(f"row={row} DONE mass_reached={report['posterior_reached']:.6f} ratio={likelihood_ratio:.6f}")
    write_json(args.output / "summary.json", dict(
        status="complete", diagnostic_only=True, rows=reports,
        reference_result=str(reference_path), reference_sha256=file_hash(reference_path),
        runtime_root=str(args.runtime_root), runtime_matches_tested_implementation=True,
        implementation_sha256=reference["implementation_sha256"],
        script_sha256=file_hash(__file__), seconds=time.monotonic()-start_time,
        limitations=["Three previously inspected failure cases, not a representative validation set.",
                     "Only the existing initial shear centre is evaluated.",
                     "Scatter positions are cached zero-shear proposal means, not individual flow draws.",
                     "Axes show central 99% of evaluated points plus every highlighted top-mass atom.",
                     "V3.6 model artifacts; no V3.9 inference integration."]))
    log("COMPLETE")


if __name__ == "__main__":
    main()
