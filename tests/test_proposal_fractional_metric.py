"""The neighbour metric may compare a coordinate multiplicatively.

A coordinate spanning decades cannot share one additive scale with a bounded
one.  Declaring it fractional compares it through ``asinh(x / scale)``, which
is logarithmic above the additive scale and additive below it.  Leaving the
declaration empty must reproduce the additive metric exactly, so an existing
cache and every completed run keep their meaning.
"""
import numpy as np
import pytest

from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable


NAMES = ("shape", "flux")


def table(fractional=(), n=8192, seed=11):
    """A bounded coordinate beside one shaped like a real flux distribution.

    The robust scale of a catalogue's flux is set by its faint bulk while the
    bright tail runs orders of magnitude above it.  That gap is the whole
    point: it is what makes one additive scale unusable for both ends.
    """

    rng = np.random.default_rng(seed)
    shape = rng.normal(0.0, 0.02, n)
    bright = rng.random(n) < 0.05
    flux = np.where(
        bright,
        10.0 ** rng.normal(4.0, 0.6, n),
        10.0 ** rng.normal(1.7, 0.25, n),
    )
    values = np.column_stack([shape, flux])
    center = np.median(values, axis=0)
    scale = np.diff(np.percentile(values, (25.0, 75.0), axis=0), axis=0)[0] / 1.349
    return ProposalCoordinateTable(
        values,
        NAMES,
        center,
        scale,
        dispersion=np.abs(values) * 0.05 + 1.0e-3,
        fractional_targets=tuple(fractional),
    )


def test_no_declaration_reproduces_the_additive_metric_exactly():
    plain = table()
    assert plain.fractional_targets == ()
    expected = (plain.values - plain.center) / plain.scale
    assert np.array_equal(plain.standardized, expected)
    # the observation query must use the identical map
    probe = plain.values[:5]
    assert np.array_equal(plain.standardize(probe), expected[:5])


def test_declaring_a_target_changes_only_that_column():
    plain = table()
    fractional = table(("flux",))
    assert np.array_equal(plain.standardized[:, 0], fractional.standardized[:, 0])
    assert not np.allclose(plain.standardized[:, 1], fractional.standardized[:, 1])


def test_the_fractional_column_compresses_the_bright_tail():
    """Additively the brightest atom is far out; fractionally it is not."""

    plain = table()
    fractional = table(("flux",))
    brightest = int(np.argmax(plain.values[:, 1]))
    additive_offset = abs(plain.standardized[brightest, 1])
    fractional_offset = abs(fractional.standardized[brightest, 1])
    # the additive metric puts it beyond any plausible shape offset
    assert additive_offset > 50.0
    # and the shape column, being bounded, never reaches that far
    assert np.abs(plain.standardized[:, 0]).max() < 10.0
    assert fractional_offset < additive_offset / 10.0


def test_a_bright_query_stops_ranking_on_flux_alone():
    """The metric decides which atoms the reranker is even offered."""

    coordinates = table()
    rng = np.random.default_rng(3)
    n = len(coordinates.values)
    weights = rng.uniform(0.5, 1.5, n)

    # A bright, strongly sheared observation.  Its flux sits in the decade the
    # additive scale cannot resolve, so an additive search spends the whole
    # budget matching flux and ignores the shape.
    observed = np.array([[0.06, 3.0e5]])

    additive = DefensiveLocalProposal(coordinates, weights)
    multiplicative = DefensiveLocalProposal(table(("flux",)), weights)

    near_additive = additive.candidates(observed, n_candidates=64)
    near_fractional = multiplicative.candidates(observed, n_candidates=64)

    def typical_shape_error(got):
        return float(
            np.median(
                np.abs(coordinates.values[got.indices[0], 0] - observed[0, 0])
            )
        )

    def typical_flux_ratio(got):
        return float(
            np.median(coordinates.values[got.indices[0], 1]) / observed[0, 1]
        )

    # Additively the candidate set is chosen on flux alone, so its shapes are
    # no better than the catalogue at large.
    catalogue_shape_error = float(
        np.median(np.abs(coordinates.values[:, 0] - observed[0, 0]))
    )
    assert typical_shape_error(near_additive) > 0.5 * catalogue_shape_error
    assert typical_shape_error(near_fractional) < typical_shape_error(near_additive)
    # Both still sit in the right part of the flux axis.
    assert 0.1 < typical_flux_ratio(near_fractional) < 10.0


def test_non_positive_values_are_admissible():
    """asinh is defined at zero and below, where a ratio is meaningless."""

    rng = np.random.default_rng(5)
    values = np.column_stack(
        [rng.normal(0.0, 0.02, 512), rng.normal(0.0, 50.0, 512)]
    )
    values[0, 1] = 0.0
    center = np.median(values, axis=0)
    scale = np.diff(np.percentile(values, (25.0, 75.0), axis=0), axis=0)[0] / 1.349
    built = ProposalCoordinateTable(
        values, NAMES, center, scale, fractional_targets=("flux",)
    )
    assert np.isfinite(built.standardized).all()


def test_declaration_survives_save_and_load(tmp_path):
    built = table(("flux",))
    built.save(tmp_path / "proposal")
    loaded = ProposalCoordinateTable.load(tmp_path / "proposal")
    assert loaded.fractional_targets == ("flux",)
    assert np.array_equal(loaded.standardized, built.standardized)

    import json

    manifest = json.loads((tmp_path / "proposal" / "manifest.json").read_text())
    assert manifest["version"] == 5
    assert manifest["fractional_targets"] == ["flux"]


def test_an_undeclared_cache_still_loads_as_additive(tmp_path):
    built = table()
    built.save(tmp_path / "proposal")
    import json

    manifest = json.loads((tmp_path / "proposal" / "manifest.json").read_text())
    assert manifest["version"] == 4
    loaded = ProposalCoordinateTable.load(tmp_path / "proposal")
    assert loaded.fractional_targets == ()
    assert np.array_equal(
        loaded.standardized, (built.values - built.center) / built.scale
    )


def test_re_flooring_carries_the_declaration():
    built = table(("flux",))
    refloored = built.with_dispersion_floor(
        percentile=50.0, fractional_targets=("flux",)
    )
    assert refloored.fractional_targets == ("flux",)
    assert np.array_equal(refloored.standardized, built.standardized)


def test_an_unknown_target_is_refused():
    with pytest.raises(ValueError, match="absent from targets"):
        table(("not_a_coordinate",))


def test_a_degenerate_column_is_refused():
    values = np.column_stack([np.linspace(-0.05, 0.05, 256), np.full(256, 500.0)])
    with pytest.raises(ValueError, match="no usable spread"):
        ProposalCoordinateTable(
            values,
            NAMES,
            np.median(values, axis=0),
            np.array([0.02, 1.0]),
            fractional_targets=("flux",),
        )
