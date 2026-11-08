"""Physical shape/radius/flux response targets and per-draw transformations.

Radius is the catalogue's measured flux radius in pixels; flux uses its
zeropoint-30 instrumental units. No clipping, contraction or fitted scaling.
"""
import numpy as np
import torch
from .flow_grid_response import fit_grid_response


_RESPONSE_GROUP_INDICES = {
    'shape': np.arange(4, dtype=np.int64),
    'radius': np.arange(4, 6, dtype=np.int64),
    'flux': np.arange(6, 8, dtype=np.int64),
}


def _equilibrated_precision(covariance):
    """Invert response covariances after an algebraically exact equilibration."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if (covariance.ndim != 3 or covariance.shape[1] != covariance.shape[2]
            or not np.isfinite(covariance).all()):
        raise ValueError('finite square response covariance required')
    sd = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    if np.any(sd <= 0) or not np.isfinite(sd).all():
        raise ValueError('positive response variances required')
    correlation = covariance/sd[:, :, None]/sd[:, None, :]
    eigenvalues = np.linalg.eigvalsh(correlation)
    if (not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0)
            or np.any(eigenvalues[:, -1]/eigenvalues[:, 0] > 1e10)):
        raise ValueError('singular or ill-conditioned joint response covariance')
    precision = np.linalg.inv(correlation)/sd[:, :, None]/sd[:, None, :]
    return precision, eigenvalues


def marginal_physical_response_statistics(statistics, groups=('shape', 'radius'),
                                          component_weights=None):
    """Select a marginal response covariance and recompute its precision.

    The four-component objective retains shape only; the six-component
    objective keeps the full shape--radius block while dropping flux.
    In particular, this function never slices the
    corresponding block from the eight-component precision, which would
    condition on flux rather than marginalize it. ``component_weights`` are
    nonnegative diagonal weights; :class:`PhysicalResponsePopulation` applies
    their symmetric square-root sandwich to the marginal precision.
    """
    groups = tuple(groups)
    if groups not in (('shape',), ('shape', 'radius'), ('shape', 'radius', 'flux')):
        raise ValueError('response groups must start with shape, optionally followed by radius and flux')
    indices = np.concatenate([_RESPONSE_GROUP_INDICES[group] for group in groups])
    target = np.asarray(statistics['target'], dtype=np.float64)
    covariance = np.asarray(statistics['covariance'], dtype=np.float64)
    jackknife = np.asarray(statistics['jackknife'], dtype=np.float64)
    n_cells = len(target)
    if (target.shape != (n_cells, 8) or covariance.shape != (n_cells, 8, 8)
            or jackknife.ndim != 3 or jackknife.shape[1:] != (n_cells, 8)
            or not all(np.isfinite(value).all() for value in (target, covariance, jackknife))):
        raise ValueError('finite aligned eight-component response statistics required')
    marginal_covariance = covariance[:, indices][:, :, indices]
    precision, eigenvalues = _equilibrated_precision(marginal_covariance)
    if component_weights is None:
        component_weights = np.ones(len(indices), dtype=np.float64)
    component_weights = np.asarray(component_weights, dtype=np.float64)
    if (component_weights.shape != (len(indices),) or not np.isfinite(component_weights).all()
            or np.any(component_weights < 0) or not np.any(component_weights > 0)):
        raise ValueError('one finite nonnegative weight per retained response component required')
    result = dict(statistics)
    result.update(target=target[:, indices], covariance=marginal_covariance,
                  precision=precision, jackknife=jackknife[:, :, indices],
                  correlation_eigenvalues=eigenvalues,
                  response_component_indices=indices,
                  response_groups=np.asarray(groups),
                  component_weights=component_weights)
    return result


def physical_outputs(raw, zero_point=30.):
    """Convert (...,e1,e2,mag,ln radius) before any sample averaging."""
    if raw.shape[-1] != 4:
        raise ValueError('expected e1,e2,mag,ln radius')
    if isinstance(raw, torch.Tensor):
        raw=raw.double()
        result=torch.stack((raw[...,0],raw[...,1],torch.exp(raw[...,3]),
                            torch.exp((zero_point-raw[...,2])*(.4*np.log(10)))),dim=-1)
        finite=bool(torch.isfinite(result).all())
    else:
        raw=np.asarray(raw,dtype=np.float64)
        with np.errstate(over='ignore',invalid='ignore'):
            result=np.stack((raw[...,0],raw[...,1],np.exp(raw[...,3]),
                             np.exp((zero_point-raw[...,2])*(.4*np.log(10)))),axis=-1)
        finite=np.isfinite(result).all()
    if not finite:
        raise ValueError('nonfinite physical radius/flux; draws must not be clipped or discarded')
    return result


def baseline_frame_gamma(gamma, ellipticity):
    """Fixed baseline spin-2 frame; exactly circular inputs use the sky axes."""
    gamma=np.asarray(gamma,dtype=float);e=np.asarray(ellipticity,dtype=float)
    if e.shape != gamma.shape or e.ndim!=2 or e.shape[1]!=2 or not np.isfinite(e).all():
        raise ValueError('finite aligned baseline ellipticity and shear required')
    norm=np.linalg.norm(e,axis=1);u=np.zeros_like(e);u[:,0]=1
    np.divide(e,norm[:,None],out=u,where=norm[:,None]>0)
    return np.stack((np.sum(u*gamma,axis=1),-u[:,1]*gamma[:,0]+u[:,0]*gamma[:,1]),axis=1)


def fit_physical_response(delta, gamma, bins, cases, n_cells, scalar_gamma=None):
    """Joint eight-component target/case covariance, with separate fixed designs.

    Shape uses sky axes. Scalars optionally use a fixed baseline frame. Matrix
    inversion is equilibrated algebraically for units, without a covariance
    floor or modification of the quadratic form.
    """
    delta=np.asarray(delta,float)
    if delta.ndim!=2 or delta.shape[1]!=4:
        raise ValueError('four physical output differences required')
    a=fit_grid_response(delta[:,:2],gamma,bins,cases,n_cells)
    # Scalar units differ strongly; normalize only to stabilize this computation.
    scale=np.std(delta[:,2:],axis=0)
    if np.any(scale<=0) or not np.isfinite(scale).all():raise ValueError('degenerate scalar data')
    b=fit_grid_response(delta[:,2:]/scale,gamma if scalar_gamma is None else scalar_gamma,bins,cases,n_cells)
    units=np.repeat(scale,2)
    target=np.concatenate((a['target'],b['target']*units),axis=1)
    leave=np.concatenate((a['jackknife'],b['jackknife']*units),axis=2)
    z=leave-leave.mean(0);n=len(z)
    cov=(n-1)/n*np.einsum('cbi,cbj->bij',z,z)
    precision,eig=_equilibrated_precision(cov)
    return dict(target=target,covariance=cov,precision=precision,jackknife=leave,
                shape_design=a['design'],scalar_design=b['design'],counts=a['counts'],
                correlation_eigenvalues=eig)
