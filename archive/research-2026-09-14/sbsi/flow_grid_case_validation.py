"""Paired case-jackknife uncertainty for held-out forward response residuals.

This is a validation statistic. It does not replace the fixed simulation
covariance in the training objective. Inputs are per-case sufficient statistics
from the same matched population and forward shear convention on both sides.
"""

import numpy as np


def paired_case_response_statistics(normal, measured_cross, model_cross):
    """Fit responses and jackknife the paired model-minus-measurement residual.

    ``normal[c,b] = sum_i gamma_i gamma_i^T`` has shape (cases,cells,2,2).
    Each ``cross[c,b] = sum_i delta_y_i gamma_i^T`` has shape
    (cases,cells,outputs,2). Model ``delta_y`` is the difference of conditional
    physical-output means, using common random numbers between shear legs.
    Simulation ``delta_y`` is the difference of the matched measurements.

    Refit the shear design after deleting each case, on both sides. A covariance
    of measured responses alone misses the covariance shared with the model
    evaluated on the same population. Finite model integration noise must be
    checked with independent draws; this function does not subtract it.

    Zero or singular residual covariance is allowed (e.g. identical predictions
    and measurements); no precision matrix or significance is inferred from it.
    """
    normal = np.asarray(normal, dtype=np.float64)
    measured = np.asarray(measured_cross, dtype=np.float64)
    model = np.asarray(model_cross, dtype=np.float64)
    if (normal.ndim != 4 or normal.shape[-2:] != (2,2)
            or normal.shape[0] < 8 or normal.shape[1] < 1
            or measured.ndim != 4 or measured.shape[:2] != normal.shape[:2]
            or measured.shape[-1] != 2 or measured.shape[-2] < 1
            or model.shape != measured.shape
            or not all(np.isfinite(a).all() for a in (normal, measured, model))):
        raise ValueError("finite aligned case/cell shear sufficient statistics with at least eight cases required")
    if not np.allclose(normal, np.swapaxes(normal,-1,-2), rtol=1e-10, atol=1e-12):
        raise ValueError("shear normal matrices must be symmetric")
    total_normal = normal.sum(axis=0)
    deleted_normal = total_normal[None]-normal
    for design in (total_normal, deleted_normal):
        eigenvalues = np.linalg.eigvalsh(design)
        if np.any(eigenvalues[...,0] <= 0) or np.any(eigenvalues[...,-1]/eigenvalues[...,0] > 1e8):
            raise ValueError("unstable full or case-deleted shear design")
    inverse = np.linalg.inv(total_normal)
    deleted_inverse = np.linalg.inv(deleted_normal)
    measured_total = measured.sum(axis=0)
    model_total = model.sum(axis=0)
    measured_response = measured_total @ inverse
    model_response = model_total @ inverse
    measured_deleted = (measured_total[None]-measured) @ deleted_inverse
    model_deleted = (model_total[None]-model) @ deleted_inverse
    # Subtract sufficient statistics before fitting to avoid cancellation of
    # large fitted matrices when the paired residual is small.
    residual_cross = model-measured
    residual_total = residual_cross.sum(axis=0)
    residual = residual_total @ inverse
    deleted_residual = ((residual_total[None]-residual_cross) @ deleted_inverse).reshape(
        normal.shape[0], normal.shape[1], -1)
    centered = deleted_residual-deleted_residual.mean(axis=0)
    covariance = (normal.shape[0]-1)/normal.shape[0] * np.einsum("cbi,cbj->bij",centered,centered)
    return dict(measured_response=measured_response, model_response=model_response,
                residual=residual, residual_covariance=covariance,
                residual_standard_error=np.sqrt(np.diagonal(covariance,axis1=-2,axis2=-1)).reshape(residual.shape),
                residual_jackknife=deleted_residual,
                measured_jackknife=measured_deleted, model_jackknife=model_deleted)


def case_cross_from_global_contributions(contribution_sums, normal, counts):
    """Recover raw cross statistics from the existing grid contribution sums.

    The grid sampler uses ``delta_e outer (gamma @ H_B^-1)``, where
    ``H_B=mean_B(gamma outer gamma)`` is fixed on the complete cell. A sum of
    these contributions over case c must be multiplied by H_B before a
    case-deleted regression can use its own shear design. Merely deleting and
    averaging these preconditioned contributions would keep the wrong design.

    ``contribution_sums`` has shape (cases,cells,2*outputs); counts are the
    complete matched cell counts, not the per-case counts.
    """
    sums = np.asarray(contribution_sums,dtype=np.float64)
    normal = np.asarray(normal,dtype=np.float64)
    counts = np.asarray(counts,dtype=np.float64)
    if (normal.ndim != 4 or normal.shape[-2:] != (2,2)
            or sums.ndim != 3 or sums.shape[:2] != normal.shape[:2]
            or sums.shape[-1] < 2 or sums.shape[-1] % 2
            or counts.shape != normal.shape[1:2] or np.any(counts <= 0)
            or not all(np.isfinite(a).all() for a in (normal,sums,counts))):
        raise ValueError("aligned finite contribution sums, shear statistics and positive cell counts required")
    h = normal.sum(axis=0)/counts[:,None,None]
    return sums.reshape(*sums.shape[:2],-1,2) @ h[None]


def summarize_paired_grid_predictions(gamma, bin_ids, cases, measured_delta,
                                      contribution_groups, precision):
    """Summarize full-population predictions already produced by a flow audit.

    Each independent group contains one flattened regression contribution per
    matched galaxy, exactly as ``GridResponsePopulation.contributions`` returns.
    This consumes complete populations, not with-replacement training batches.
    ``measured_delta`` is in physical units. The fixed training precision is
    used only to reproduce the normalized objective; paired uncertainty remains
    a separate diagnostic. Substantial arrays must be processed on a worker.
    """
    gamma = np.asarray(gamma,dtype=np.float64)
    bins = np.asarray(bin_ids)
    cases = np.asarray(cases)
    measured = np.asarray(measured_delta,dtype=np.float64)
    precision = np.asarray(precision,dtype=np.float64)
    if (gamma.ndim != 2 or gamma.shape[1] != 2 or not len(gamma)
            or measured.ndim != 2 or len(measured) != len(gamma) or not measured.shape[1]
            or bins.shape != (len(gamma),) or not np.issubdtype(bins.dtype,np.integer)
            or np.any(bins < 0) or cases.shape != bins.shape
            or len(contribution_groups) < 2
            or not np.isfinite(gamma).all() or not np.isfinite(measured).all()):
        raise ValueError("aligned complete matched rows and at least two independent prediction groups required")
    case_labels, case_ids = np.unique(cases,return_inverse=True)
    n_cases, n_cells = len(case_labels), int(bins.max())+1
    components = 2*measured.shape[1]
    if (precision.shape != (n_cells,components,components)
            or not np.isfinite(precision).all()):
        raise ValueError("fixed precision dimensions must match cells and response components")
    counts = np.bincount(bins,minlength=n_cells)
    if np.any(counts == 0):
        raise ValueError("all cells must have matched rows")
    group = case_ids*n_cells+bins
    def aggregate(values):
        return np.bincount(group,weights=values,minlength=n_cases*n_cells).reshape(n_cases,n_cells)
    normal = np.empty((n_cases,n_cells,2,2))
    cross = np.empty((n_cases,n_cells,measured.shape[1],2))
    for j in range(2):
        for k in range(2):
            normal[...,j,k] = aggregate(gamma[:,j]*gamma[:,k])
        for k in range(measured.shape[1]):
            cross[...,k,j] = aggregate(measured[:,k]*gamma[:,j])
    model_cross = []
    for contributions in contribution_groups:
        contributions = np.asarray(contributions,dtype=np.float64)
        if contributions.shape != (len(gamma),components) or not np.isfinite(contributions).all():
            raise ValueError("finite prediction contribution for every matched row required")
        sums = np.stack([aggregate(contributions[:,k]) for k in range(components)],axis=-1)
        model_cross.append(case_cross_from_global_contributions(sums,normal,counts))
    model_cross = np.asarray(model_cross)
    result = paired_case_response_statistics(normal,cross,model_cross.mean(axis=0))
    group_responses = np.stack([paired_case_response_statistics(normal,cross,c)["model_response"]
                                for c in model_cross])
    flat = group_responses.reshape(len(model_cross),n_cells,components)
    centered = flat-flat.mean(axis=0)
    # Estimated integration covariance of the mean over all independent groups.
    integration_covariance = np.einsum("rbi,rbj->bij",centered,centered)/(
        len(model_cross)*(len(model_cross)-1))
    residual = result["residual"].reshape(n_cells,components)
    mean_quadratic = .5*np.einsum("bi,bij,bj->b",residual,precision,residual)/components
    integration_term = .5*np.einsum("bij,bji->b",precision,integration_covariance)/components
    # Algebraically the mean of all distinct-group cross scores. Do not clamp
    # a negative estimate or treat it as proof of an accurate integrated mean.
    cross_score = mean_quadratic-integration_term
    result.update(cases=case_labels,counts=counts,group_model_responses=group_responses,
                  integration_covariance_of_mean=integration_covariance,
                  mean_quadratic_per_cell=mean_quadratic,
                  integration_quadratic_per_cell=integration_term,
                  cross_score_per_cell=cross_score,normalized_cross_score=float(cross_score.mean()))
    return result
