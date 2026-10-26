"""Fixed joint-output summaries, conditional on each observed leg's usability.

The four-dimensional histogram is deliberately fixed before the independent
response is inspected. These coarse diagnostics do not certify a density.
"""

import numpy as np
import torch


COORDINATES = ("e1", "e2", "log_radius_pixels", "MAG_AUTO")
HISTOGRAM_COORDINATES = ("e1", "e2", "radius_pixels", "MAG_AUTO")
BOUNDARIES = (
    (-1., -.6, -.3, 0., .3, .6, 1.),
    (-1., -.6, -.3, 0., .3, .6, 1.),
    (1., 2., 3., 3.75, 5., 7.5, 12.5),
    (18., 20., 22., 23., 24., 25., 25.8, 26.5, 28.),
)
HISTOGRAM_SHAPE = tuple(len(edges) + 1 for edges in BOUNDARIES)


def output_statistics(values, weights, cuts):
    """Sum physical outputs with nonnegative weights; zero-weight rows are ignored.

    Each generated draw must carry its parent's usability divided by draws.
    Inputs may have any leading shape, with four outputs and K cut indicators.
    Radius bins include their upper boundary; other coordinates include their
    lower boundary. Nonpositive photometry is retained in a separate histogram
    bin; log moments are explicitly conditional on positive radius and flux.
    """
    values = torch.as_tensor(values, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64, device=values.device)
    cuts = torch.as_tensor(cuts, device=values.device)
    if (values.shape[-1:] != (4,) or weights.shape != values.shape[:-1]
            or cuts.shape[:-1] != weights.shape or cuts.dtype != torch.bool
            or cuts.shape[-1] < 1):
        raise ValueError("values, weights and boolean cuts must align")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("finite nonnegative weights required")
    use = weights > 0
    values, weights, cuts = values[use], weights[use], cuts[use]
    if not bool(cuts[:, 0].all()):
        raise ValueError("first diagnostic cut must include the full usable population")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("positive-weight measurements must be finite")
    positive = (values[:, 2:] > 0).all(1)
    cut_weights = weights[:, None] * cuts
    moments = {
        "mass": cut_weights.sum(0),
        "outside_disk_mass": weights[values[:, :2].square().sum(-1) >= 1].sum(),
        "invalid_photometry_mass": weights[~positive].sum(),
        "zero_radius_mass": weights[values[:, 2] == 0].sum(),
        "zero_flux_mass": weights[values[:, 3] == 0].sum(),
        "negative_photometry_mass": weights[(values[:, 2:] < 0).any(1)].sum(),
    }
    # No floor or substitute value enters any moment. Undefined log moments
    # have their own retained mass, while the response calculation is separate.
    values, weights, cut_weights = values[positive], weights[positive], cut_weights[positive]
    magnitude = 30 - 2.5 * torch.log10(values[:, 3])
    coordinates = torch.stack((values[:, 0], values[:, 1], values[:, 2].log(), magnitude), -1)
    moments.update({"moment_mass": cut_weights.sum(0),
                    "first": torch.einsum("nk,nd->kd", cut_weights, coordinates),
                    "second": torch.einsum("nk,nd,ne->kde", cut_weights, coordinates, coordinates)})
    physical = torch.stack((values[:, 0], values[:, 1], values[:, 2], magnitude), -1)
    index = torch.zeros(len(values), dtype=torch.int64, device=values.device)
    for axis, boundaries in enumerate(BOUNDARIES):
        edges = values.new_tensor(boundaries)
        index = index * HISTOGRAM_SHAPE[axis] + torch.bucketize(
            physical[:, axis].contiguous(), edges, right=axis != 2)
    moments["histogram"] = torch.bincount(index, weights=weights, minlength=int(np.prod(HISTOGRAM_SHAPE)))
    return {name: value.cpu().numpy() for name, value in moments.items()}


def empty_statistics(ncases, ncuts):
    return {"mass": np.zeros((ncases, ncuts)),
            "moment_mass": np.zeros((ncases, ncuts)),
            "first": np.zeros((ncases, ncuts, 4)),
            "second": np.zeros((ncases, ncuts, 4, 4)),
            "histogram": np.zeros((ncases, int(np.prod(HISTOGRAM_SHAPE)))),
            "outside_disk_mass": np.zeros(ncases),
            "invalid_photometry_mass": np.zeros(ncases),
            "zero_radius_mass": np.zeros(ncases),
            "zero_flux_mass": np.zeros(ncases),
            "negative_photometry_mass": np.zeros(ncases)}


def summarize_statistics(observed, generated, bootstrap_weights):
    """Paired scene errors for moments; histogram TV has no significance claim."""
    weights = np.asarray(bootstrap_weights, dtype=float)
    ncases = len(observed["mass"])
    if (weights.ndim != 2 or weights.shape[1] != ncases
            or not np.isfinite(weights).all() or (weights < 0).any()
            or (weights.sum(1) <= 0).any()):
        raise ValueError("invalid scene-bootstrap weights")

    def integrate(stats, case_weights):
        mass = np.einsum("bc,ck->bk", case_weights, stats["mass"])
        moment_mass = np.einsum("bc,ck->bk", case_weights, stats["moment_mass"])
        if (mass <= 0).any() or (moment_mass <= 0).any():
            raise ValueError("empty predictive cut or no defined log moments in scene bootstrap")
        first = np.einsum("bc,ckd->bkd", case_weights, stats["first"]) / moment_mass[..., None]
        second = np.einsum("bc,ckde->bkde", case_weights, stats["second"]) / moment_mass[..., None, None]
        return {"selection_fraction": mass / mass[:, :1], "mean": first,
                "undefined_log_moment_fraction": 1-moment_mass/mass,
                "covariance": second - first[..., :, None] * first[..., None, :]}

    integrated = {name: integrate(stats, np.concatenate((np.ones((1, ncases)), weights)))
                  for name, stats in (("observed", observed), ("generated", generated))}
    result = {}
    for field in ("selection_fraction", "undefined_log_moment_fraction", "mean", "covariance"):
        left, right = integrated["observed"][field], integrated["generated"][field]
        delta = right - left
        result[field] = {"observed": left[0], "generated": right[0],
                         "generated_minus_observed": delta[0],
                         "paired_case_bootstrap_se": delta[1:].std(0, ddof=1)}
    histograms, invalid_probabilities = {}, {}
    for name, stats in (("observed", observed), ("generated", generated)):
        counts = np.asarray(stats["histogram"]).sum(0)
        invalid = np.asarray(stats["invalid_photometry_mass"]).sum()
        total = stats["mass"][:, 0].sum()
        np.testing.assert_allclose(counts.sum()+invalid, total, rtol=1e-10)
        histograms[name] = counts / total
        invalid_probabilities[name] = float(invalid/total)
    result["joint_histogram"] = {**histograms,
        "invalid_photometry_bin_probability": invalid_probabilities,
        "total_variation": float((np.abs(histograms["generated"] - histograms["observed"]).sum()
                                 + abs(invalid_probabilities["generated"]-invalid_probabilities["observed"]))/2),
        "interpretation": "coarse fixed 4D bins; empirical TV includes sampling noise and is not an acceptance threshold"}
    result["outside_disk_fraction"] = {name: float(stats["outside_disk_mass"].sum()/stats["mass"][:, 0].sum())
                                       for name, stats in (("observed", observed), ("generated", generated))}
    result["photometry_failure_fractions"] = {
        name: {field: float(stats[field].sum()/stats["mass"][:, 0].sum())
               for field in ("invalid_photometry_mass", "zero_radius_mass", "zero_flux_mass", "negative_photometry_mass")}
        for name, stats in (("observed", observed), ("generated", generated))}
    return result


def metadata():
    return {"revision": 2, "moment_coordinates": COORDINATES, "histogram_coordinates": HISTOGRAM_COORDINATES,
            "moment_conditioning": "strictly positive radius and flux; undefined mass reported per cut; no numerical floor",
            "invalid_photometry_histogram_bin": "separate retained mass; included in normalization and total variation",
            "histogram_boundaries": BOUNDARIES, "histogram_shape": HISTOGRAM_SHAPE,
            "histogram_flatten_order": "C", "histogram_right": [True, True, False, True],
            "population": "entire supplied truth parent; no observed-g0 subcohort restriction",
            "usability": "each leg's actual observed usable flag; no observed measured cut on generated draws",
            "limitations": ["fixed coarse histogram and low-order moments do not establish density equivalence",
                            "case errors exclude training uncertainty and do not bound finite-draw bias",
                            "descriptive model diagnostics; no checkpoint selection on independent scenes"]}
