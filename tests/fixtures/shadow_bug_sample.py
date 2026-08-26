"""FIXTURE, not production code.  Reproduces the `pre` shadowing verbatim in shape.

This is the pattern that killed SLURM jobs 15844986/87/88 21 s in: `pre` is bound before
the loop, read on every iteration, and rebound INSIDE the loop to an unrelated object.
Iteration 1 still sees the preprocessor; iteration 2 gets a tensor and raises
AttributeError.  The sweep must fire on this file.  Do not "fix" it.
"""


def draw_loop(bundle, draws, ev):
    pre = bundle.condition_preprocessor
    out = []
    for m in range(draws):
        ctx = pre.add_missing_indicators(ev[m])
        out.append(ctx)
        pre = ev[:, :m + 1]          # <-- the bug: unrelated rebind of a live name
    return out
