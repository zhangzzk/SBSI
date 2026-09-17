import pandas as pd

from scripts.prepare_full_domain_flow import truth_parent_mask


def test_truth_magnitude_parent_cut_is_strict_and_finite():
    frame = pd.DataFrame({"r_input_p": [25.999, 26.0, 26.001, float("nan")]})
    assert truth_parent_mask(frame, 26.0).tolist() == [True, False, False, False]
