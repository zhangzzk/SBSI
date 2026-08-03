"""Arithmetic precision for the flow's forward passes, as one switchable region.

The §5B score pass is 10 affine coupling layers of 256x256 matmuls evaluated once per
`(galaxy, node)` -- at `G = 2765` and 2M rows that is `2.2e10` network evaluations, about
`60 PFLOP`, per leg.  It is pure matmul, which is exactly what Ampere's tensor cores exist
for, and PyTorch leaves TF32 OFF by default for matmul.  So the default configuration runs
the most tensor-core-shaped workload in the repo in the one precision that cannot use them.

This module exists to make that a choice with a name attached rather than an accident, and
to make the choice REVERSIBLE.  `torch.backends.cuda.matmul.allow_tf32` is process-global:
flipping it in one place silently changes every later matmul, including any that a
subsequent "fp32" control was meant to measure against.  `precision_region` restores what
it found.

ON WHETHER THIS IS SAFE.  It is not safe by fiat and must not be treated that way.  The
useful facts are that the log-likelihoods are already stored to fp16 (`out_dtype=float16`
in `PosteriorShapeEstimator.log_likelihood`), that everything downstream is a softmax over
the node bank -- invariant to a per-row constant -- and that TF32 carries a 10-bit mantissa,
the same as the fp16 those values already pass through.  So TF32 has good reason to be
below the existing noise floor.  bf16 (8-bit mantissa) does not have that reason.  Either
way the acceptance test is empirical and is `scripts/bench_score_speed.py`: run the mode
end-to-end and compare `ghat`, not `max |dlogL|`, against the run's statistical error.
"""

import contextlib

import torch

MODES = ("fp32", "tf32", "bf16", "fp16")


@contextlib.contextmanager
def precision_region(mode, device="cuda"):
    """Run the enclosed flow evaluations in `mode`, then restore the previous settings.

    `fp32` is the PyTorch default and a genuine no-op, so it stays usable as a control.
    `tf32` flips the two backend flags; `bf16`/`fp16` additionally open an autocast region,
    which reaches the `nn.Linear` calls inside the coupling layers without any model or
    call-site changes.  On CPU everything but `fp32` is a no-op with a warning, so a laptop
    smoke test does not silently measure something else.
    """
    if mode not in MODES:
        raise ValueError(f"precision must be one of {MODES}, got {mode!r}")
    cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    if mode != "fp32" and not cuda:
        print(f"[precision] {mode} requested without CUDA; running fp32", flush=True)
        mode = "fp32"
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    torch.backends.cudnn.allow_tf32 = mode == "tf32"
    try:
        if mode in ("bf16", "fp16"):
            dt = torch.bfloat16 if mode == "bf16" else torch.float16
            with torch.autocast("cuda", dtype=dt):
                yield mode
        else:
            yield mode
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved[0]
        torch.backends.cudnn.allow_tf32 = saved[1]
