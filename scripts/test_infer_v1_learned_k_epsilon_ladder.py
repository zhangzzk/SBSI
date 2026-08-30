#!/usr/bin/env python
"""Measure the K/epsilon feasibility frontier for learned proposal ESS."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from plot_infer_v1_sampling import _candidate_positions, _sha256
from test_infer_v1_proposal_diversification import _load_pipeline, _sync_cuda
from sbsi.amortized_proposal import ContrastiveRetrievalProposal
from sbsi.sampling_diagnostics import (
    evaluate_exact_proposal_target,
    summarize_defensive_proposal,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-ids", nargs="+", type=int, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, required=True)
    parser.add_argument("--epsilon-values", nargs="+", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atom-chunk", type=int, default=65536)
    return parser.parse_args(argv)


def _percentiles(values):
    return np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90]).tolist()


def main(argv=None):
    args = _parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if len(args.checkpoints) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        raise ValueError("checkpoint labels must be unique and align with checkpoints")
    object_ids = list(dict.fromkeys(map(int, args.object_ids)))
    k_values = sorted(set(map(int, args.k_values)))
    epsilon_values = sorted(set(map(float, args.epsilon_values)))
    if (
        not object_ids
        or min(object_ids) < 0
        or not k_values
        or min(k_values) <= 0
        or not epsilon_values
        or min(epsilon_values) <= 0
        or max(epsilon_values) > 1
        or args.atom_chunk <= 0
    ):
        raise ValueError("object IDs, K, epsilon, and chunk sizes are invalid")

    torch.set_float32_matmul_precision("high")
    reference = Path(args.reference_result).resolve()
    _, identity, likelihood, gaussian, mock, run_metadata = _load_pipeline(
        reference, args.device
    )
    if max(object_ids) >= len(mock.measurements):
        raise ValueError("an object ID lies outside the frozen mock")
    center = tuple(map(float, identity["initial_center"]["center"]))
    maximum_k = max(k_values)
    retrievers = {}
    for label, checkpoint_name in zip(args.labels, args.checkpoints):
        checkpoint = Path(checkpoint_name).resolve()
        retrievers[label] = ContrastiveRetrievalProposal(
            gaussian.coordinates,
            likelihood.cache.prior.weights,
            local_base_weights=likelihood.cache.get(*center).detection_probability,
            likelihood=likelihood,
            checkpoint=checkpoint,
            center=center,
            contrastive_pool=maximum_k,
            retrieval_object_chunk=1,
            rerank_object_chunk=1,
            atom_chunk=int(args.atom_chunk),
        )
    active = next(iter(retrievers.values())).active_indices
    prior = likelihood.cache.prior.weights[active].astype(np.float64, copy=True)
    prior /= prior.sum()
    rows = []
    for number, object_id in enumerate(object_ids, start=1):
        observed = mock.measurements.iloc[[object_id]].reset_index(drop=True)
        supports = {}
        for label, retriever in retrievers.items():
            supports[label] = retriever.candidates(
                observed,
                n_candidates=maximum_k,
                prefilter_candidates=None,
                torch_device=likelihood.flow_model.device,
            ).indices[0]
        _sync_cuda()
        reference_candidates = next(iter(retrievers.values())).candidates(
            observed,
            n_candidates=k_values[0],
            prefilter_candidates=None,
            torch_device=likelihood.flow_model.device,
        )
        comparison = evaluate_exact_proposal_target(
            likelihood,
            observed,
            next(iter(retrievers.values())),
            object_id=int(object_id),
            center=center,
            n_candidates=k_values[0],
            prefilter_candidates=None,
            epsilon=epsilon_values[0],
            candidate_backend="torch",
            atom_chunk=int(args.atom_chunk),
            candidates=reference_candidates,
        )
        _sync_cuda()
        if not np.array_equal(active, comparison.atom_indices):
            raise RuntimeError("active support changed across observations")
        target = comparison.target_probability
        oracle_pool = np.argpartition(target, -maximum_k)[-maximum_k:]
        oracle_rank = oracle_pool[
            np.argsort(-target[oracle_pool], kind="stable")
        ]
        position_prefixes = {
            label: _candidate_positions(active, support)
            for label, support in supports.items()
        }
        position_prefixes["oracle_exact_topk"] = oracle_rank
        for label, positions in position_prefixes.items():
            for k in k_values:
                prefix = positions[:k]
                for epsilon in epsilon_values:
                    metrics = summarize_defensive_proposal(
                        target, prior, prefix, epsilon
                    )
                    rows.append(
                        {
                            "object_id": int(object_id),
                            "method": label,
                            "k": int(k),
                            "epsilon": float(epsilon),
                            **metrics,
                        }
                    )
        print(
            json.dumps(
                {
                    "phase": "k_epsilon_ladder",
                    "completed": number,
                    "total": len(object_ids),
                    "object_id": int(object_id),
                }
            ),
            flush=True,
        )
        del comparison, target, supports, position_prefixes
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    aggregate_rows = []
    for (method, k, epsilon), group in frame.groupby(
        ["method", "k", "epsilon"], sort=False
    ):
        aggregate_rows.append(
            {
                "method": method,
                "k": int(k),
                "epsilon": float(epsilon),
                "capture_percentiles": _percentiles(group["candidate_target_mass"]),
                "ess_fraction_percentiles": _percentiles(
                    group["asymptotic_ess_fraction"]
                ),
                "maximum_ratio_percentiles": _percentiles(
                    group["maximum_target_to_proposal_ratio"]
                ),
                "fraction_ess_at_least_20_percent": float(
                    np.mean(group["asymptotic_ess_fraction"] >= 0.2)
                ),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "per_observation.csv", index=False)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_json(output / "aggregate.json", orient="records", indent=2)
    result = {
        "status": "complete",
        "design": {
            "object_ids": object_ids,
            "center": list(center),
            "k_values": k_values,
            "epsilon_values": epsilon_values,
            "checkpoint_labels": dict(zip(args.labels, map(str, args.checkpoints))),
            "exact_target_used_for_retrieval": False,
            "oracle_is_evaluation_only": True,
        },
        "identity": {
            **run_metadata,
            "reference_result": str(reference),
            "reference_result_sha256": _sha256(reference),
            "diagnostic_sha256": _sha256(Path(__file__)),
        },
        "files": [
            str(output / "per_observation.csv"),
            str(output / "aggregate.json"),
        ],
        "infer_v1_defaults_changed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
