"""FIXTURE, not production code.  Patterns the sweep reports ON PURPOSE.

The sweep cannot tell a deliberate fixed-point update from an accidental clobber -- both
are "rebound to something not derived from the old value, after being read above in the
same loop".  That is not a defect to be tuned away: telling them apart requires knowing
what the loop MEANS, which is a human judgement.  So the sweep reports them and
`scripts/loop_shadowing_allowlist.txt` records the judgement with a reason.

This fixture pins that behaviour, so nobody later "fixes" the sweep into silence here and
loses the real bug along with it.  `sbsi/inference.py::infer_shear::shear` is the live
instance of exactly this shape.
"""


def fixed_point(start, steps, tol=1e-9):
    shear = start
    for _ in range(steps):
        candidate = 0.5 * shear + 0.5
        if abs(candidate - shear) < tol:  # reads the PREVIOUS iterate, on purpose
            shear = candidate
            break
        shear = candidate  # deliberate loop-carried state
    return shear
