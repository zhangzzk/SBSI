import numpy as np

from scripts.refine_guard_control_variate_flow import acceptable_step


def test_density_gate_rejects_response_improvement_that_spends_excess_density():
    assert acceptable_step(.1, .05, -4.760, -4.7605)
    assert not acceptable_step(.1, .01, -4.75, -4.7605)
    assert not acceptable_step(.1, .2, -4.77, -4.7605)
    assert not acceptable_step(.1, .1, -4.77, -4.7605)
    assert not acceptable_step(.1, np.nan, -4.77, -4.7605)
