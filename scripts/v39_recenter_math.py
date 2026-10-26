"""Small reductions shared by the frozen V3.9 recentering diagnostic."""
import numpy as np


def summarize(center, score, information):
    score, information = np.asarray(score), np.asarray(information)
    if len(score) < 2 or score.shape != (len(score), 2) or information.shape != (len(score), 2, 2):
        raise ValueError('aligned two-component moments required')
    if not np.isfinite(score).all() or not np.isfinite(information).all():
        raise ValueError('nonfinite moments')
    hessian = information.sum(0)
    hessian = (hessian+hessian.T)/2
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= 0:
        return dict(status='non_positive_information', eigenvalues=eigenvalues.tolist()), None
    step = np.linalg.solve(hessian, score.sum(0))
    influence = np.linalg.solve(hessian/len(score),
        (score-np.einsum('nij,j->ni', information, step)).T).T
    se = np.sqrt(np.diag(np.cov(influence, rowvar=False, ddof=1))/len(score))
    result = dict(status='estimated', center=list(center), step=step.tolist(),
        estimate=(np.asarray(center)+step).tolist(), robust_standard_error=se.tolist(),
        eigenvalues=eigenvalues.tolist(),
        quadratic_gain=float(.5*step@hessian@step),
        small_step=bool(np.all(np.abs(step) <= .1*se)))
    return result, influence


def paired(first, second, first_influence, second_influence):
    if first_influence is None or second_influence is None:
        return dict(status='invalid_information')
    delta = second_influence-first_influence
    return dict(status='estimated',
        difference=(np.asarray(second['estimate'])-first['estimate']).tolist(),
        standard_error=np.sqrt(np.diag(np.cov(delta, rowvar=False, ddof=1))/len(delta)).tolist())


def next_center(result):
    """No truth-based starts, Hessian repair, clipping, or empirical offsets."""
    if result['status'] != 'estimated':
        raise ValueError('cannot recentre with non-positive information')
    point, step = np.asarray(result['estimate']), np.asarray(result['step'])
    if not np.isfinite(point).all() or np.linalg.norm(point) >= 1 or np.linalg.norm(step) > .05:
        raise ValueError('Newton step exceeds predeclared diagnostic guard; inspect a profile')
    return point.tolist()
