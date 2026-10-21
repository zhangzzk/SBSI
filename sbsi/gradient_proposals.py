"""Experimental deterministic drift proposals; MH always uses the full target."""
import numpy as np


def bounded_langevin_mean(values, gradient, cholesky, *, max_drift=1.):
    """Gaussian mean with drift bounded in proposal-standard-deviation units.

    ``cholesky`` includes the frozen step scale. It must be identical at the
    two endpoints of one move. Gradients may approximate the target gradient;
    correctness comes from evaluating both actual proposal densities in MH.
    """
    x, grad, factor = map(lambda a: np.asarray(a, float), (values, gradient, cholesky))
    if (x.ndim != 2 or grad.shape != x.shape or factor.shape != (len(x), x.shape[1], x.shape[1])
            or not all(np.isfinite(v).all() for v in (x, grad, factor))
            or not np.isfinite(max_drift) or max_drift <= 0
            or np.any(np.diagonal(factor, axis1=1, axis2=2) <= 0)
            or np.any(np.triu(factor, 1) != 0)):
        raise ValueError('finite aligned states/gradients and positive lower Cholesky factors required')
    drift = .5*np.einsum('nji,nj->ni', factor, grad)
    norm = np.linalg.norm(drift, axis=1)
    drift *= np.minimum(1., max_drift/np.maximum(norm, np.finfo(float).tiny))[:, None]
    return x+np.einsum('nij,nj->ni', factor, drift)


def gaussian_log_hastings(old, proposed, forward_mean, reverse_mean, cholesky):
    """log q(old|proposed)-log q(proposed|old), same frozen covariance."""
    forward = np.linalg.solve(cholesky, (proposed-forward_mean)[..., None])[..., 0]
    reverse = np.linalg.solve(cholesky, (old-reverse_mean)[..., None])[..., 0]
    return .5*(np.square(forward).sum(1)-np.square(reverse).sum(1))


class ReplicaTransport:
    """RNG-free cold/hot lineage diagnostics; auxiliary levels are not chains."""
    def __init__(self, levels, slots, dimension):
        if levels < 2 or slots < 1 or dimension < 1:
            raise ValueError('multiple levels and positive slots/dimension required')
        self.levels, self.slots = levels, slots
        self.identity = np.arange(levels*slots)
        self.stage = np.zeros(levels*slots, dtype=int)
        self.origin = np.zeros((levels*slots, dimension))
        self.count = np.zeros(slots, dtype=int)
        self.displacement = np.zeros((slots, dimension))

    def update(self, permutation, values):
        """Values are in post-swap row order. Count cold -> hot -> cold trips."""
        self.identity = self.identity[permutation]
        cold = self.identity[:self.slots]
        hot = self.identity[-self.slots:]
        completed = self.stage[cold] == 2
        slots = cold[completed] % self.slots
        self.count[slots] += 1
        self.displacement[slots] += np.abs(values[:self.slots][completed]-self.origin[cold[completed]])
        starting = (self.stage[cold] == 0) | completed
        self.origin[cold[starting]] = values[:self.slots][starting]
        self.stage[cold] = 1
        self.stage[hot[self.stage[hot] == 1]] = 2

    def report(self):
        return dict(round_trips_per_ladder=self.count.tolist(),
                    mean_absolute_endpoint_change=(self.displacement/np.maximum(self.count[:, None], 1)).tolist(),
                    endpoint_change_sum=self.displacement.tolist(),
                    interpretation='A round trip alone does not certify decorrelation or discovery of a new mode.')
