"""Differentiable summary kernel for diagnostic scores and proposal drifts.

This path never replaces SummaryLikelihood in MH acceptance. Its preprocessing
arithmetic can differ from the frozen numpy path. Parameter tensors are not
modified. At fixed scene, response b is independent of shear; latent proposal
gradients hold b fixed at the separately recomputed value of each endpoint.
"""
import numpy as np
import torch

from .disk_precision import LOG_GAP_COLUMN, physical_log_prob_with_gap, transport_with_log_gap


def tensor_log_kernel(flow, detector, raw_context, physical, gap, coefficient, shear):
    """Rowwise kernel with torch context shear and lossless radial metadata."""
    e = raw_context[:, :2]
    g = shear.expand_as(e) if shear.ndim == 1 else shear
    real = 1+(g*e).sum(-1)
    imag = g[:, 0]*e[:, 1]-g[:, 1]*e[:, 0]
    numerator = e+g
    denominator = real.square()+imag.square()
    sheared = torch.stack(((numerator[:, 0]*real+numerator[:, 1]*imag)/denominator,
                          (numerator[:, 1]*real-numerator[:, 0]*imag)/denominator), -1)
    context = torch.cat((sheared, raw_context[:, 2:]), -1)
    network_dtype = next(flow.model.parameters()).dtype
    standard = flow.condition_preprocessor.transform_tensor(context.to(network_dtype))
    shape, new_gap, ld = transport_with_log_gap(physical[:, :2], gap,
                                               coefficient[:, None]*g, inverse=True)
    density = physical_log_prob_with_gap(flow.model, torch.cat((shape, physical[:, 2:]), -1),
                                         new_gap, standard)+ld
    density += torch.as_tensor(flow.target_transform.scales, device=context.device, dtype=torch.float64).log().sum()
    members = getattr(detector, 'members', (detector,))
    logits = [member.logits_from_raw_tensor(context.to(next(member.model.parameters()).dtype))
              for member in members]
    logq = torch.stack([torch.nn.functional.logsigmoid(v) for v in logits])
    return density+torch.logsumexp(logq, dim=0)-np.log(len(members))


class SummaryGradients:
    def __init__(self, flow, detector, *, batch_size=4096):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('positive fixed gradient batch size required')
        self.flow, self.detector, self.batch_size = flow, detector, batch_size

    def evaluate(self, values, observed, shear, *, wrt='context', hessian=False):
        """Return rowwise density/gradient, optionally the full shear Hessian.

        With hessian=True return (density, gradient, Hessian). Scene coordinates,
        observed auxiliaries and response b stay fixed while differentiating
        shear. No latent support map, finite stencil or emulator derivative is
        involved. This remains a diagnostic path, separate from MH acceptance.
        """
        if hessian and wrt != 'shear':
            raise ValueError('Hessian evaluation is defined for shear only')
        if wrt not in ('context', 'shear'):
            raise ValueError('differentiate context or shear')
        values = np.asarray(values, float)
        n = len(values)
        if values.shape != (n, 9) or not np.isfinite(values).all() or n < 1:
            raise ValueError('nonempty finite nine-column summaries required')
        raw = observed[list(self.flow.target_transform.target_names)].to_numpy(float)
        gaps = observed[LOG_GAP_COLUMN].to_numpy(float)
        if len(raw) == 1:
            raw, gaps = np.repeat(raw, n, axis=0), np.repeat(gaps, n)
        if len(raw) != n or not np.isfinite(raw).all() or not np.isfinite(gaps).all():
            raise ValueError('finite observations aligned with summaries required')
        result, gradient = np.empty(n), np.empty((n, 8 if wrt == 'context' else 2))
        curvature = np.empty((n, 2, 2)) if hessian else None
        for start in range(0, n, self.batch_size):
            stop = min(n, start+self.batch_size)
            rows = np.minimum(np.arange(start, start+self.batch_size), stop-1)
            tensor = lambda a: torch.as_tensor(a, device=self.flow.device, dtype=torch.float64)
            context = tensor(values[rows, :8]).clone().requires_grad_(wrt == 'context')
            g = tensor(np.broadcast_to(shear, (len(rows), 2)).copy()).requires_grad_(wrt == 'shear')
            density = tensor_log_kernel(self.flow, self.detector, context, tensor(raw[rows]), tensor(gaps[rows]),
                                        tensor(values[rows, 8]), g)
            grad, = torch.autograd.grad(density.sum(), context if wrt == 'context' else g,
                                        create_graph=hessian)
            if hessian:
                second = [torch.autograd.grad(grad[:, j].sum(), g, retain_graph=j == 0)[0]
                          for j in range(2)]
                curvature[start:stop] = torch.stack(second, dim=1).detach().cpu().numpy()[:stop-start]
            result[start:stop] = density.detach().cpu().numpy()[:stop-start]
            gradient[start:stop] = grad.detach().cpu().numpy()[:stop-start]
        if not np.isfinite(result).all() or not np.isfinite(gradient).all():
            raise FloatingPointError('nonfinite diagnostic density or gradient')
        if hessian:
            if not np.isfinite(curvature).all():
                raise FloatingPointError('nonfinite diagnostic shear Hessian')
            return result, gradient, curvature
        return result, gradient
