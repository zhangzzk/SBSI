"""FIXTURE, not production code.  Patterns the sweep must NEVER report.

Every rebind here is a self-refine: the new value is DERIVED from the old one, so no
earlier meaning is lost and the loop cannot be surprised on its second iteration.  A sweep
that only ever fires is useless -- this file is the other half of the evidence.

What is deliberately NOT here: a fixed-point update like `shear = candidate`.  That one
DOES get reported, on purpose, and lives in shadow_reviewed_sample.py -- see its docstring.
"""


def self_refine(batches, device):
    total = 0
    for batch in batches:
        batch = batch.to(device)          # loop target, self-refine -- ruff PLW2901's job
        total += len(batch)
    return total


def accumulate(chunks):
    reservoir = None
    for chunk in chunks:
        reservoir = chunk if reservoir is None else reservoir + chunk   # self-refine
    return reservoir


def narrow_in_place(rows, valid):
    keep = rows
    for _ in range(3):
        n = len(keep)
        keep = keep[valid[:n]]            # derived from itself
    return keep
