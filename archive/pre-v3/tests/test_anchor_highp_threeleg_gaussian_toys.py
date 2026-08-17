import numpy as np
import pandas as pd

from scripts.analyze_anchor_highp_threeleg_gaussian_toys import (
    build_cell_means,
    validate_decompositions,
    vector_stat,
)
from scripts.prepare_anchor_highp_threeleg_gaussian_toys import (
    choose_templates,
    rank_close_pairs,
)
from scripts.run_anchor_highp_threeleg_gaussian_toy import (
    compose_three_leg_images,
    primary_flux_for_nominal_snr,
    repeat_seed,
)


def test_close_pair_ranking_and_balanced_unique_case_selection():
    pair_rows = []
    candidate_rows = []
    for case in range(700, 730):
        anchor = 1000 + case
        candidate_rows.append({
            "case": case,
            "input_index": anchor,
            "n_close_pairs": 8 if case < 712 else (4 if case < 722 else 2),
        })
        for j, response in enumerate([0.02, -0.3, 0.1, 0.04, 0.03, 0.01, -0.005, 0.002]):
            pair_rows.append({
                "case": case,
                "anchor_index": anchor,
                "secondary_index": 10_000 + case * 10 + j,
                "distance": 0.5 + 0.2 * j,
                "response": response,
                "n_pairs": 8,
            })
    ranked = rank_close_pairs(pd.DataFrame(pair_rows), close_radius=3.0)
    first = ranked.loc[ranked.case == 700].sort_values("close_response_rank")
    assert first.response.tolist()[:3] == [-0.3, 0.1, 0.04]
    candidates = pd.DataFrame(candidate_rows)
    chosen = choose_templates(
        candidates, multiplicities=(2, 4, 8), n_per_multiplicity=3, seed=12
    )
    assert chosen.multiplicity.value_counts().to_dict() == {2: 3, 4: 3, 8: 3}
    assert not chosen.case.duplicated().any()


def test_three_leg_composition_keeps_and_removes_context_exactly():
    image = lambda value: np.full((2, 2), value, dtype=float)
    sources = [
        {"zero": image(10)},
        {
            "zero": image(1), "g1_plus": image(2), "g1_minus": image(0),
            "g2_plus": image(3), "g2_minus": image(-1),
        },
        {
            "zero": image(4), "g1_plus": image(6), "g1_minus": image(2),
            "g2_plus": image(8), "g2_minus": image(0),
        },
    ]
    coherent, contextual, pair_only = compose_three_leg_images(sources)
    np.testing.assert_allclose(coherent["g1_plus"], image(18))
    np.testing.assert_allclose(contextual[0]["g1_plus"], image(16))
    np.testing.assert_allclose(contextual[1]["g1_plus"], image(17))
    np.testing.assert_allclose(pair_only[0]["g1_plus"], image(12))
    np.testing.assert_allclose(pair_only[1]["g1_plus"], image(16))


def test_legacy_snr_scaling_and_matched_seeds_are_deterministic():
    np.testing.assert_allclose(
        primary_flux_for_nominal_snr(20, 10), 20 * 10 * np.sqrt(20)
    )
    assert repeat_seed(4, 10, 17) == repeat_seed(4, 10, 17)
    assert repeat_seed(4, 10, 17) != repeat_seed(4, 11, 17)
    assert repeat_seed(4, 10, 17) != repeat_seed(5, 10, 17)


def synthetic_draws():
    rows = []
    pairs = []
    for template_id in range(3):
        for noise_mode, realizations in (("noiseless", [-1]), ("noisy", [0, 1, 2])):
            for realization in realizations:
                coherent = 0.30 + 0.01 * template_id + 0.001 * max(realization, 0)
                context = 0.28 + 0.01 * template_id
                pair = 0.25 + 0.005 * template_id
                rows.append({
                    "template_id": template_id,
                    "case": 700 + template_id,
                    "input_index": 100 + template_id,
                    "multiplicity": 2,
                    "nominal_snr": 20.0,
                    "realization": realization,
                    "noise_mode": noise_mode,
                    "success": True,
                    "R_trace_coherent": coherent,
                    "R_trace_context_sum": context,
                    "R_trace_pair_sum": pair,
                    "delta_additivity_trace": coherent - context,
                    "delta_context_trace": context - pair,
                    "delta_total_trace": coherent - pair,
                    "decomposition_replay_trace": 0.0,
                })
                for source_rank in (1, 2):
                    pairs.append({
                        "template_id": template_id,
                        "case": 700 + template_id,
                        "input_index": 100 + template_id,
                        "multiplicity": 2,
                        "nominal_snr": 20.0,
                        "realization": realization,
                        "noise_mode": noise_mode,
                        "source_rank": source_rank,
                        "R_trace_context": context / 2,
                        "R_trace_pair_only": pair / 2,
                    })
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def test_analysis_reduces_noise_before_geometry_statistics_and_closes_sums():
    draws, pairs = synthetic_draws()
    replay = validate_decompositions(draws, pairs)
    assert max(replay.values()) < 1e-14
    cells = build_cell_means(draws, min_success_fraction=1.0)
    assert len(cells) == 3
    np.testing.assert_allclose(
        cells.loc[cells.template_id == 0, "R_trace_coherent"], 0.301
    )
    stat = vector_stat(cells.delta_total_trace.to_numpy(float), seed=3, n_boot=500)
    assert stat["n_templates"] == 3
    # Geometry means, not 3 templates x 3 technical repeats.
    np.testing.assert_allclose(stat["mean"], cells.delta_total_trace.mean())
