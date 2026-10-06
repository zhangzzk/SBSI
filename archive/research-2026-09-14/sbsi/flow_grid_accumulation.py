"""Large effective NLL/response batches with exact chunked response gradients."""

import math

import numpy as np
import torch

from .flow_grid_full_cells import _validate as validate_replay
from .flow_grid_full_cells import epoch_full_cells
from .flow_grid_response import independent_grid_score


def response_plan(population, cells_per_step, pairs_per_cell, seed, pairs_per_group=None):
    if cells_per_step < 1 or pairs_per_cell < 1 or seed < 0:
        raise ValueError("positive response batch sizes and nonnegative seed required")
    rng = np.random.default_rng(seed)
    cells = rng.permutation(population.n_cells)[:cells_per_step]
    device = population.context.device
    counts = np.full(len(cells), pairs_per_cell, dtype=np.int64)
    if pairs_per_group is not None:
        if pairs_per_group < len(cells):
            raise ValueError("response budget must give each selected cell at least one pair")
        counts[:] = pairs_per_group // len(cells)
        counts[:pairs_per_group % len(cells)] += 1
    local = torch.arange(len(cells),device=device).repeat_interleave(torch.as_tensor(counts,device=device))
    groups = [torch.as_tensor(np.concatenate([
        rng.integers(population.offsets[b],population.offsets[b+1],int(n))
        for b,n in zip(cells,counts)]),device=device) for _ in range(2)]
    return cells,local,groups


def sampled_response_backward(model, population, cells_per_step, pairs_per_cell,
                              draws, *, chunk_size=256, seed=7301, weight=1., pairs_per_group=None):
    """Accumulate the exact sampled cross-score gradient, retaining NLL gradients.

    Draw independent galaxy samples and latent streams for groups A/C. First
    compute group means without a graph, then replay identical samples in
    chunks. Model parameters must remain fixed throughout both passes.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("finite nonnegative response weight required")
    cells,local,groups = response_plan(population,cells_per_step,pairs_per_cell,seed,pairs_per_group)
    validate_replay(model,population,cells,draws,chunk_size,seed)
    ids = torch.as_tensor(cells,device=population.context.device)
    components = population.target.shape[1]
    def chunks(stream):
        for block,start in enumerate(range(0,len(local),chunk_size)):
            stop = min(start+chunk_size,len(local))
            yield groups[stream][start:stop],local[start:stop],seed+100000+2*block+stream
    divisor = torch.bincount(local,minlength=len(cells)).to(torch.float64)
    means = []
    with torch.no_grad():
        for stream in range(2):
            total = torch.zeros((len(cells),components),device=population.context.device,dtype=torch.float64)
            for rows,bins,latent_seed in chunks(stream):
                values = population.contributions(model,rows,draws,seed=latent_seed)
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("nonfinite sampled response contribution")
                total.index_add_(0,bins,values)
            means.append(total/divisor[:,None])
        target,precision = population.target[ids],population.precision[ids]
        score = independent_grid_score(*means,target,precision).mean()/components
        if not bool(torch.isfinite(score)):
            raise ValueError("nonfinite sampled response score")
        coefficients = [
            .5*torch.einsum('bij,bj->bi',precision,means[1]-target)/(components*len(cells)),
            .5*torch.einsum('bji,bj->bi',precision,means[0]-target)/(components*len(cells)),
        ]
    if weight:
        with torch.enable_grad():
            for stream in range(2):
                for rows,bins,latent_seed in chunks(stream):
                    values = population.contributions(model,rows,draws,seed=latent_seed)
                    surrogate = (values*coefficients[stream][bins]/divisor[bins,None]).sum()*weight
                    if not bool(torch.isfinite(surrogate)):
                        raise ValueError("nonfinite response replay")
                    surrogate.backward()
    return dict(loss=float(score),mean_a=means[0],mean_c=means[1],cells=cells.tolist(),
                pairs_per_group=len(local))


def accumulated_epoch(model, values, context, population, optimizer, *, weight,
                      mode, nll_batch_size, nll_chunk_size, cells_per_step,
                      pairs_per_cell, draws, response_chunk_size, seed,
                      max_grad_norm=5., max_nll_rows=None, response_pairs_per_epoch=None):
    """Train one complete NLL epoch; report sampled versus full pair coverage."""
    if mode == 'full' and (max_nll_rows is not None or response_pairs_per_epoch is not None):
        raise ValueError('epoch caps require sampled mode')
    if mode == 'full':
        result = epoch_full_cells(model,values,context,population,optimizer,weight=weight,
                                  cells_per_step=cells_per_step,draws=draws,
                                  response_chunk_size=response_chunk_size,nll_chunk_size=nll_chunk_size,
                                  seed=seed,max_grad_norm=max_grad_norm)
        result['response_pair_coverage'] = 'every matched pair once per latent group, plus replay'
        return result
    if (mode != 'sampled' or min(nll_batch_size,nll_chunk_size) < 1
            or len(values) != len(context) or not len(values)
            or not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("valid sampled/full mode, aligned NLL rows and positive batch/clipping options required")
    # Validate before any optimizer update or accumulated NLL gradient.
    validate_replay(model,population,[0],draws,response_chunk_size,seed)
    if cells_per_step < 1 or pairs_per_cell < 1 or not math.isfinite(weight) or weight < 0:
        raise ValueError("valid response batch options required")
    nll_limit = len(values) if max_nll_rows is None else min(max_nll_rows,len(values))
    if nll_limit < 1:
        raise ValueError('NLL row cap must be positive')
    steps = math.ceil(nll_limit/nll_batch_size)
    if response_pairs_per_epoch is not None and response_pairs_per_epoch < steps*min(cells_per_step,population.n_cells):
        raise ValueError('response epoch budget must populate every selected cell at every update')
    model.train()
    generator = torch.Generator(device=context.device)
    generator.manual_seed(int(seed))
    order = torch.randperm(len(values),device=context.device,generator=generator)[:nll_limit]
    nll_total = score_total = 0.
    nll_seen = pairs_seen = 0
    norms = []
    for step,start in enumerate(range(0,nll_limit,nll_batch_size)):
        rows = order[start:start+nll_batch_size]
        optimizer.zero_grad(set_to_none=True)
        nll = 0.
        for offset in range(0,len(rows),nll_chunk_size):
            indices = rows[offset:offset+nll_chunk_size]
            contribution = -model.log_prob(values[indices],context[indices]).sum()/len(rows)
            if not bool(torch.isfinite(contribution)):
                raise ValueError("nonfinite accumulated NLL")
            contribution.backward()
            nll += float(contribution.detach())
        step_pairs = None if response_pairs_per_epoch is None else (
            response_pairs_per_epoch//steps + int(step < response_pairs_per_epoch%steps))
        response = sampled_response_backward(model,population,cells_per_step,pairs_per_cell,draws,
                                             chunk_size=response_chunk_size,seed=seed+step*1000003,
                                             weight=weight,pairs_per_group=step_pairs)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        if not bool(torch.isfinite(norm)):
            raise ValueError("nonfinite combined gradient")
        optimizer.step()
        norms.append(float(norm))
        nll_total += nll*len(rows)
        score_total += response['loss']*len(rows)
        nll_seen += len(rows)
        pairs_seen += response['pairs_per_group']
    if nll_seen != nll_limit:
        raise RuntimeError("incomplete NLL epoch")
    return dict(nll=nll_total/nll_seen,grid=score_total/nll_seen,nll_rows=nll_seen,
                response_pairs_per_group=pairs_seen,optimizer_steps=len(norms),gradient_norms=norms,
                response_pair_coverage='with-replacement occurrences; no claim of complete pair coverage')


def summarize_validation_means(means, target, precision):
    """All distinct-group cross score plus an MC jackknife diagnostic.

    Jackknife SE describes finite integration repetitions, not model seeds or
    simulation cases; near a degenerate quadratic it is not an exact confidence
    bound. Retain group means for independent re-evaluation of selected epochs.
    """
    means,target,precision = (np.asarray(a,dtype=np.float64) for a in (means,target,precision))
    if (means.ndim != 3 or means.shape[0] < 4 or means.shape[2] not in (4,6,8)
            or target.shape != means.shape[1:] or precision.shape != (means.shape[1],means.shape[2],means.shape[2])
            or not all(np.isfinite(a).all() for a in (means,target,precision))):
        raise ValueError("at least four finite aligned validation group means required")
    def score(group_means):
        residual = group_means-target
        center = residual.mean(0)
        deviations = residual-center
        mc_cov = np.einsum('rbi,rbj->bij',deviations,deviations)/(
            len(residual)*(len(residual)-1))
        quadratic = .5*np.einsum('bi,bij,bj->b',center,precision,center)/means.shape[2]
        mc_term = .5*np.einsum('bij,bji->b',precision,mc_cov)/means.shape[2]
        return quadratic-mc_term,quadratic,mc_term,mc_cov
    per_cell,quadratic,mc_term,mc_cov = score(means)
    deleted = np.array([score(np.delete(means,k,axis=0))[0].mean() for k in range(len(means))])
    error = math.sqrt((len(means)-1)/len(means)*np.square(deleted-deleted.mean()).sum())
    return dict(loss=float(per_cell.mean()),per_cell=per_cell,group_means=means,
                mean_quadratic=float(quadratic.mean()),integration_quadratic=float(mc_term.mean()),
                integration_covariance_of_mean=mc_cov,mc_jackknife_se=error,
                mc_deleted_scores=deleted)


class _TailObservedPopulation:
    """Observe the actual validation draws without changing their distribution."""

    def __init__(self,population,shape_means):
        self.population = population
        self.shape_means = torch.as_tensor(shape_means,device=population.context.device,dtype=torch.float64)
        if self.shape_means.shape != (2,) or not bool(torch.isfinite(self.shape_means).all()):
            raise ValueError('two finite physical shape offsets required')
        self.thresholds = np.array([1.,2.,10.,100.])
        self.counts = np.zeros(4,dtype=np.int64)
        self.draw_count = 0
        self.maximum = 0.
        # This band describes finite-precision outputs. It neither repairs a
        # draw nor proves that each event in the band was caused by rounding.
        self.squared_radius_band = 16*np.finfo(np.float64).eps
        self.support_labels = np.array([
            'squared_radius_ge_1', 'squared_radius_ge_1_within_roundoff_band',
            'squared_radius_above_roundoff_band', 'interior_within_roundoff_band'])
        self.support_counts = np.zeros(4,dtype=np.int64)
        self.maximum_squared = 0.

    def __getattr__(self,name):
        return getattr(self.population,name)

    def observe_shapes(self, physical):
        squared = physical.double().square().sum(-1)
        amplitude = squared.sqrt()
        if not bool(torch.isfinite(amplitude).all()):
            raise ValueError('nonfinite physical shape sample in validation')
        self.maximum = max(self.maximum,float(amplitude.max()))
        self.maximum_squared = max(self.maximum_squared,float(squared.max()))
        self.draw_count += amplitude.numel()
        for j,threshold in enumerate(self.thresholds):
            self.counts[j] += int((amplitude>threshold).sum())
        outside = squared >= 1
        masks = [outside, outside & (squared <= 1+self.squared_radius_band),
                 squared > 1+self.squared_radius_band,
                 (squared < 1) & (squared >= 1-self.squared_radius_band)]
        self.support_counts += torch.stack([mask.sum() for mask in masks]).cpu().numpy()

    def contributions(self,model,indices,draws,seed=None):
        from .flow_training import complete_flow_context_means
        pairs = self.pairs[indices]
        means,samples = complete_flow_context_means(
            model,[self.context[pairs[:,0]],self.context[pairs[:,1]]],draws,
            seed=seed,return_samples=True)
        for values in samples:
            physical = values[:,:,:2].double()*self.scales.double()+self.shape_means
            self.observe_shapes(physical)
        delta = ((means[1][:,:2]-means[0][:,:2])*self.scales).double()
        return (delta[:,:,None]*self.design[indices,None,:]).reshape(-1,4)


def evaluate_validation_groups(model,population,draws,groups,batch_size,seed,shape_means=None):
    if groups < 4 or groups % 2 or batch_size < 1:
        raise ValueError("even validation group count >=4 and positive batch size required")
    means = []
    stride = 2*math.ceil(len(population.pairs)/batch_size)+2
    observer = _TailObservedPopulation(population,shape_means) if shape_means is not None else None
    for repeat in range(groups//2):
        if observer is None:
            result = population.evaluate(model,draws,batch_size,seed+repeat*stride)
        else:
            from .flow_grid_response import GridResponsePopulation
            result = GridResponsePopulation.evaluate(observer,model,draws,batch_size,seed+repeat*stride)
        means.extend([result['mean_a'],result['mean_c']])
    result = summarize_validation_means(means,population.target.cpu().numpy(),population.precision.cpu().numpy())
    if observer is not None:
        result.update(tail_thresholds=observer.thresholds,tail_exceedance_counts=observer.counts,
                      tail_draw_count=observer.draw_count,max_shape_amplitude=observer.maximum,
                      tail_exceedance_fractions=observer.counts/observer.draw_count,
                      shape_support_labels=observer.support_labels,shape_support_counts=observer.support_counts,
                      shape_support_fractions=observer.support_counts/observer.draw_count,
                      shape_squared_radius_band=observer.squared_radius_band,
                      max_shape_squared_radius=observer.maximum_squared)
    return result
