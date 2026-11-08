"""Conditional squared-error score, independent latent banks on the SAME row."""
import numpy as np
import torch
from sbsi.flow_guard_response import torch_guard_weights


def cross_score(first, second, observed):
    if first.shape != second.shape or first.shape != observed.shape:
        raise ValueError('Both predictions and observed targets must share row/cut axes')
    return ((first - observed) * (second - observed)).mean()


class Population:
    def __init__(self, context, observed, target, cfg, helpers):
        self.context, self.observed = context, observed
        self.c = helpers
        self.cuts = cfg['regularization_cuts']
        self.settings = cfg['probability_loss']
        self.means = torch.as_tensor(target.means, device=context.device, dtype=torch.float64)
        self.scales = torch.as_tensor(target.scales, device=context.device, dtype=torch.float64)

    def targets(self, rows):
        physical = self.observed[rows] * self.scales + self.means
        return torch_guard_weights(physical, self.cuts,
                                  radius_softness=self.settings['radius_softness'],
                                  magnitude_softness=self.settings['magnitude_softness'])

    def probability(self, model, rows, draws, seed):
        ctx = self.context[rows]
        bank = self.c.portable.latent_bank(len(rows), draws, 4, seed)
        z = torch.from_numpy(bank).to(ctx.device).reshape(-1, 4)
        expanded = ctx[:, None, :].expand(-1, draws, -1).reshape(len(z), -1)
        photo = self.c.photo_module.photo(model, z, expanded).reshape(len(rows), draws, 2)
        physical = torch.cat([torch.zeros_like(photo), photo], -1)
        masks = torch_guard_weights(physical, self.cuts,
                                    radius_softness=self.settings['radius_softness'],
                                    magnitude_softness=self.settings['magnitude_softness'])
        return masks.mean(1)

    def chunk_score(self, model, rows, draws, seed):
        first = self.probability(model, rows, draws, seed)
        second = self.probability(model, rows, draws, seed + 100000000)
        return cross_score(first, second, self.targets(rows))

    def backward(self, model, *, seed, weight, rows=None, draws=None, chunk=None):
        cfg = self.settings
        draws = draws or cfg['draws_per_bank']
        chunk = chunk or cfg['chunk']
        if rows is None:
            rows = torch.as_tensor(np.random.default_rng(seed).integers(
                len(self.context), size=cfg['rows']), device=self.context.device)
        total = 0.0
        for block, lo in enumerate(range(0, len(rows), chunk)):
            part = rows[lo:lo + chunk]
            # Separate row sampling from both latent-bank random streams.
            loss = self.chunk_score(model, part, draws, seed + 10000000 + block)
            assert bool(torch.isfinite(loss))
            fraction = len(part) / len(rows)
            total += float(loss.detach()) * fraction
            if weight:
                (weight * fraction * loss).backward()
        return {'loss': total, 'rows': len(rows), 'draws_per_bank': draws,
                'same_rows_both_banks': True, 'weight': weight}


def gradient_check(model, population):
    """Compare chunk accumulation to the same graph summed before backward."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    rows = torch.as_tensor(np.random.default_rng(202611071301).integers(
        len(population.context), size=130), device=population.context.device)
    seed, chunk, draws = 202611071302, 64, 4

    def gradients():
        return [p.grad.detach().double().clone() if p.grad is not None
                else torch.zeros_like(p, dtype=torch.float64) for p in parameters]

    model.zero_grad(set_to_none=True)
    result = population.backward(model, rows=rows, seed=seed, weight=1.,
                                 draws=draws, chunk=chunk)
    expected = gradients()
    model.zero_grad(set_to_none=True)
    direct = sum(population.chunk_score(model, rows[lo:lo + chunk], draws, seed + 10000000 + block)
                 * len(rows[lo:lo + chunk]) / len(rows)
                 for block, lo in enumerate(range(0, len(rows), chunk)))
    direct.backward()
    actual = gradients()
    norm = sum(float(g.square().sum()) for g in actual)
    error = sum(float((a - b).square().sum()) for a, b in zip(actual, expected))
    relative = (error / max(norm, 1e-30)) ** .5
    assert norm > 0 and relative < 3e-5, (norm, relative)
    assert abs(float(direct.detach()) - result['loss']) < 1e-10
    assert all(p.grad is None for p in model.base.base.shape.parameters())
    model.zero_grad(set_to_none=True)
    return {'relative_gradient_error': relative, 'loss': result['loss'],
            'same_rows_both_banks': True, 'uneven_final_chunk': True,
            'conditional_shape_gradients_absent': True}
