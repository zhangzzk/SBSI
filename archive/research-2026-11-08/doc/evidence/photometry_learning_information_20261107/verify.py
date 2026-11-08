"""Small analytical checks of the probability estimator, not model fitting."""
import itertools
import json
from pathlib import Path
import torch

H = Path(__file__).resolve().parent


def main():
    ns = {'__file__': str(H / 'probability.py'), '__name__': 'probability_fixture'}
    exec(compile((H / 'probability.py').read_bytes(), str(H / 'probability.py'), 'exec'), ns)
    score = ns['cross_score']
    q = torch.tensor([[.2, .75], [.8, .35]], dtype=torch.float64)
    y = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
    # Explicitly marginalize independent zero-mean integration errors.
    noise = torch.tensor([[.06, .03], [.04, .08]], dtype=torch.float64)
    expectation = torch.stack([score(q + a * noise, q + b * noise, y)
                               for a, b in itertools.product([-1., 1.], repeat=2)]).mean()
    truth = (q - y).square().mean()
    torch.testing.assert_close(expectation, truth, rtol=0, atol=1e-15)
    reused = torch.stack([score(q + a * noise, q + a * noise, y) for a in [-1., 1.]]).mean()
    torch.testing.assert_close(reused - truth, noise.square().mean(), rtol=0, atol=1e-15)
    # Aggregation before squaring misses opposite conditional errors.
    errors = torch.tensor([[.2], [-.2]], dtype=torch.float64)
    assert errors.mean().square() == 0
    torch.testing.assert_close(score(errors, errors, torch.zeros_like(errors)),
                               torch.tensor(.04, dtype=torch.float64))
    theta = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    directions = torch.tensor([[.7, -.2], [.3, .6]], dtype=torch.float64)
    loss = torch.stack([score(q + theta * directions + a * noise,
                             q + theta * directions + b * noise, y)
                        for a, b in itertools.product([-1., 1.], repeat=2)]).mean()
    gradient, = torch.autograd.grad(loss, theta)
    exact = 2 * ((q + theta * directions - y) * directions).mean()
    torch.testing.assert_close(gradient, exact, rtol=0, atol=1e-15)
    try:
        score(q, q[:1], y)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched row axes accepted')
    result = dict(independent_banks_unbiased=True, reused_bank_variance_bias_detected=True,
                  conditional_cancellation_detected=True, expected_gradient_exact=True,
                  row_shape_mismatch_rejected=True)
    (H / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ANALYTICAL_CHECKS', result, flush=True)


if __name__ == '__main__':
    main()
