"""Joint paired-case uncertainty for two physical response coordinate frames.

This is an evaluation statistic, never a replacement for fixed training V_B.
Shape uses the sky shear design and radius/flux the fixed intrinsic design.
"""
import numpy as np
from .flow_grid_case_validation import paired_case_response_statistics
from .flow_grid_accumulation import summarize_validation_means


def _case_covariance(deleted):
    centered=deleted-deleted.mean(axis=0)
    return (len(deleted)-1)/len(deleted)*np.einsum('cbi,cbj->bij',centered,centered)


def paired_physical_case_statistics(normal,measured_cross,model_cross_groups,precision=None):
    """Refit both WLS designs after deleting each case, retaining correlations.

    normal and measured_cross have shape (cases,cells,blocks,2,2), with
    blocks=(shape,scalars). The last axes of cross are (output,shear).
    Model cross groups prepend an independent latent-group axis (at least4).
    Scalar means/cross statistics must already be in physical units.

    The paired residual covariance includes finite integration noise. Separate
    MC covariance is estimated from independent groups; neither is subtracted
    from the other or treated as an exact confidence guarantee.
    """
    normal,measured,groups=(np.asarray(x,dtype=np.float64) for x in
                           (normal,measured_cross,model_cross_groups))
    if (normal.ndim!=5 or normal.shape[2:]!=(2,2,2) or measured.shape!=normal.shape
            or groups.ndim!=6 or groups.shape[1:]!=normal.shape or len(groups)<4
            or not all(np.isfinite(x).all() for x in (normal,measured,groups))):
        raise ValueError('finite aligned two-design statistics and at least four independent groups required')
    blocks=[paired_case_response_statistics(normal[:,:,k],measured[:,:,k],groups.mean(0)[:,:,k])
            for k in range(2)]
    nc,nb=normal.shape[:2]
    joined=lambda key:np.concatenate([b[key].reshape(nc,nb,4) for b in blocks],axis=-1)
    residual_deleted=joined('residual_jackknife')
    model_deleted=joined('model_jackknife');measured_deleted=joined('measured_jackknife')
    full=lambda key:np.concatenate([b[key].reshape(nb,4) for b in blocks],axis=-1)
    group_responses=[]
    for group in groups:
        group_responses.append(np.concatenate([
            paired_case_response_statistics(normal[:,:,k],measured[:,:,k],group[:,:,k])['model_response'].reshape(nb,4)
            for k in range(2)],axis=-1))
    group_responses=np.stack(group_responses)
    deviations=group_responses-group_responses.mean(0)
    mc=np.einsum('gbi,gbj->bij',deviations,deviations)/(len(groups)*(len(groups)-1))
    covariance=_case_covariance(residual_deleted)
    result=dict(measured_response=full('measured_response'),model_response=full('model_response'),
                residual=full('residual'),residual_jackknife=residual_deleted,
                measured_jackknife=measured_deleted,model_jackknife=model_deleted,
                residual_covariance=covariance,residual_standard_error=np.sqrt(np.diagonal(covariance,axis1=-2,axis2=-1)),
                measurement_covariance=_case_covariance(measured_deleted),
                group_model_responses=group_responses,integration_covariance_of_mean=mc)
    if precision is not None:
        result['objective']=summarize_validation_means(group_responses,result['measured_response'],precision)
    return result


def fixed_cell_average_statistics(statistics,weights):
    """Fixed cell/marginal averages, retaining case and MC cross-cell covariance.

    Weights sum to one and are chosen without fitting model residuals. This is
    not the pooled unit-pair WLS; pooled WLS requires summing raw sufficient
    statistics over cells BEFORE fitting each of the two shear designs.
    """
    weights=np.asarray(weights,dtype=np.float64)
    residual=np.asarray(statistics['residual'])
    if (weights.shape!=(len(residual),) or not np.isfinite(weights).all()
            or np.any(weights<0) or not np.isclose(weights.sum(),1.,rtol=0,atol=1e-12)):
        raise ValueError('nonnegative fixed cell weights summing to one required')
    deleted=np.einsum('b,cbi->ci',weights,statistics['residual_jackknife'])
    groups=np.einsum('b,gbi->gi',weights,statistics['group_model_responses'])
    d=groups-groups.mean(0)
    return dict(residual=weights@residual,
                measured_response=weights@statistics['measured_response'],
                model_response=weights@statistics['model_response'],
                residual_covariance=_case_covariance(deleted[:,None,:])[0],
                integration_covariance_of_mean=d.T@d/(len(groups)*(len(groups)-1)))


def physical_case_sufficient_statistics(gamma,scalar_gamma,bins,cases,delta,*,n_cells,case_labels):
    """Accumulate raw two-frame case statistics; chunk outputs add exactly in real arithmetic.

    Explicit full-population case labels and cell count prevent chunk-dependent
    reindexing. The caller must preserve these labels across both measurements
    and every model group. Inputs are physical per-row shear differences, not
    already preconditioned WLS contributions.
    """
    gamma,scalar_gamma,delta=(np.asarray(x,dtype=np.float64) for x in (gamma,scalar_gamma,delta))
    bins,cases,labels=map(np.asarray,(bins,cases,case_labels))
    if (gamma.ndim!=2 or gamma.shape[1]!=2 or scalar_gamma.shape!=gamma.shape
            or delta.shape!=(len(gamma),4) or bins.shape!=(len(gamma),) or cases.shape!=bins.shape
            or labels.ndim!=1 or len(labels)<8 or len(np.unique(labels))!=len(labels)
            or n_cells<1 or not np.issubdtype(bins.dtype,np.integer)
            or np.any(bins<0) or np.any(bins>=n_cells)
            or not all(np.isfinite(x).all() for x in (gamma,scalar_gamma,delta))):
        raise ValueError('aligned finite physical rows, explicit unique cases and valid cells required')
    lookup={label:index for index,label in enumerate(labels.tolist())}
    try:case_ids=np.array([lookup[c] for c in cases.tolist()],dtype=np.int64)
    except KeyError as exc:raise ValueError('row case absent from declared case labels') from exc
    ids=case_ids*n_cells+bins
    shape=(len(labels),n_cells)
    def aggregate(values):return np.bincount(ids,weights=values,minlength=len(labels)*n_cells).reshape(shape)
    normal=np.zeros((*shape,2,2,2));cross=np.zeros_like(normal)
    for block,g in enumerate((gamma,scalar_gamma)):
        for j in range(2):
            for k in range(2):
                normal[:,:,block,j,k]=aggregate(g[:,j]*g[:,k])
                cross[:,:,block,j,k]=aggregate(delta[:,block*2+j]*g[:,k])
    return dict(normal=normal,cross=cross,counts=np.bincount(ids,minlength=len(labels)*n_cells).reshape(shape),case_labels=labels.copy())
