"""Joint physical cell responses using the exact physical draw map."""
import numpy as np
import torch
from .flow_training import complete_flow_context_means
from .flow_grid_accumulation import summarize_validation_means


class _PhysicalView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        if not callable(getattr(model, 'forward_physical', None)):
            raise ValueError('explicit physical flow required')
        self.model = model
        self.target_dim, self.context_dim = model.target_dim, model.context_dim

    def forward(self, z, context):
        return self.model.forward_physical(z, context)


def physical_context_means(model, contexts, draws, seed, return_samples=False):
    return complete_flow_context_means(_PhysicalView(model), contexts, draws, seed,
                                       return_samples=return_samples)


class PhysicalResponsePopulation:
    """Sky-frame shape and baseline-frame scalar WLS, with four, six or eight entries.

    The six-entry form contains four shape and two radius responses. Flux still
    passes through the physical flow and diagnostics, but is absent from this
    response objective. Component weights form a symmetric square-root
    sandwich around the unweighted inverse covariance.
    """
    def __init__(self, context, pairs, bins, statistics, precision=None):
        self.context = context
        device = context.device
        bins = np.asarray(bins)
        order = np.argsort(bins, kind='stable')
        self.n_cells = len(statistics['target'])
        counts = np.bincount(bins, minlength=self.n_cells)
        if not np.array_equal(counts, statistics['counts']) or np.any(counts <= 0):
            raise ValueError('cell membership does not match fitted response counts')
        self.offsets = np.r_[0, counts.cumsum()]
        self.bin_ids = torch.as_tensor(bins[order], device=device, dtype=torch.long)
        self.pairs = torch.as_tensor(np.asarray(pairs)[order], device=device, dtype=torch.long)
        self.design = torch.as_tensor(np.stack((statistics['shape_design'],
            statistics['scalar_design']), axis=1)[order], device=device, dtype=torch.float64)
        self.target = torch.as_tensor(statistics['target'], device=device, dtype=torch.float64)
        components = self.target.shape[1] if self.target.ndim == 2 else -1
        if components not in (4, 6, 8):
            raise ValueError('physical response objective requires four, six or eight components')
        component_indices = np.asarray(statistics.get(
            'response_component_indices', np.arange(components)), dtype=np.int64)
        if not np.array_equal(component_indices, np.arange(components)):
            raise ValueError('physical response components must retain shape, radius, then flux order')
        weights = np.asarray(statistics.get('component_weights', np.ones(components)), dtype=np.float64)
        if (weights.shape != (components,) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.any(weights > 0)):
            raise ValueError('one finite nonnegative weight per response component required')
        base_precision = torch.as_tensor(statistics['precision'] if precision is None else precision,
                                         device=device, dtype=torch.float64)
        root_weights = torch.as_tensor(np.sqrt(weights), device=device, dtype=torch.float64)
        self.base_precision = base_precision
        self.component_weights = root_weights.square()
        self.precision = (root_weights[None, :, None]*base_precision
                          *root_weights[None, None, :])
        self.response_components = components
        if (self.target.shape != (self.n_cells,components)
                or self.precision.shape != (self.n_cells,components,components)
                or self.design.shape != (len(self.pairs),2,2)
                or not bool(torch.isfinite(self.target).all() & torch.isfinite(self.precision).all()
                            & torch.isfinite(self.design).all())):
            raise ValueError('finite aligned physical-response WLS statistics required')
        if not bool(torch.allclose(self.precision, self.precision.transpose(1, 2), rtol=1e-10,
                                   atol=1e-12)):
            raise ValueError('weighted physical-response precision must be symmetric')

    def contributions(self, model, indices, draws, seed=None):
        pair = self.pairs[indices]
        means = physical_context_means(model,[self.context[pair[:,0]],self.context[pair[:,1]]],draws,seed)
        return self.from_difference(means[1]-means[0],indices)

    def from_difference(self, delta, indices):
        return self.all_from_difference(delta, indices)[:, :self.response_components]

    def all_from_difference(self, delta, indices):
        """Return all eight physical responses, including diagnostic-only flux."""
        design = self.design[indices]
        return torch.cat(((delta[:,:2,None]*design[:,0,None,:]).reshape(-1,4),
                          (delta[:,2:,None]*design[:,1,None,:]).reshape(-1,4)),dim=1)


def evaluate_physical_groups(model,population,draws=64,groups=4,batch_size=1024,seed=970000000001):
    """Independent latent groups; finite MC diagnostics, no case-CI claim."""
    if groups < 4 or groups > 100 or len(population.pairs)>=1000000000:
        raise ValueError('four or more disjoint groups required')
    model.eval()
    counts=torch.as_tensor(np.diff(population.offsets),device=population.context.device)
    all_means=[];all_physical_means=[];lo=np.full(4,np.inf);hi=np.full(4,-np.inf)
    nonpositive=np.zeros(2,dtype=np.int64);boundary=material=0;draw_count=0
    with torch.no_grad():
        for group in range(groups):
            total=torch.zeros_like(population.target)
            total_physical=torch.zeros((population.n_cells,8),device=population.context.device,
                                       dtype=torch.float64)
            for start in range(0,len(population.pairs),batch_size):
                ids=torch.arange(start,min(start+batch_size,len(population.pairs)),device=population.context.device)
                pair=population.pairs[ids]
                means,samples=physical_context_means(model,[population.context[pair[:,0]],population.context[pair[:,1]]],draws,seed+group*1000000000+start,True)
                for sample in samples:
                    if not bool(torch.isfinite(sample).all()):raise ValueError('nonfinite physical draw')
                    lo=np.minimum(lo,sample.amin((0,1)).cpu().numpy());hi=np.maximum(hi,sample.amax((0,1)).cpu().numpy())
                    nonpositive+=(sample[...,2:]<=0).sum((0,1)).cpu().numpy()
                    r2=sample[...,:2].square().sum(-1)
                    boundary+=int((r2>=1).sum());material+=int((r2>1+16*np.finfo(float).eps).sum())
                    draw_count+=sample.shape[0]*sample.shape[1]
                contribution=population.from_difference(means[1]-means[0],ids)
                all_contribution=population.all_from_difference(means[1]-means[0],ids)
                if not bool(torch.isfinite(contribution).all()):raise ValueError('nonfinite physical response')
                total.index_add_(0,population.bin_ids[ids],contribution)
                total_physical.index_add_(0,population.bin_ids[ids],all_contribution)
            all_means.append((total/counts[:,None]).cpu().numpy())
            all_physical_means.append((total_physical/counts[:,None]).cpu().numpy())
    result=summarize_validation_means(np.stack(all_means),population.target.cpu().numpy(),population.precision.cpu().numpy())
    result.update(minima=lo,maxima=hi,nonpositive_scalar_draws=nonpositive,
                  literal_disk_boundary=boundary,material_disk_exterior=material,draw_count=draw_count,
                  all_physical_group_means=np.stack(all_physical_means),
                  seed=seed,draws=draws,groups=groups)
    return result
