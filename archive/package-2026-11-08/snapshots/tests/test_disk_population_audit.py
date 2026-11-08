import numpy as np
import pandas as pd
import pytest

from scripts.audit_disk_population import match_truth, validation_rows


def test_reserved_panel_is_reproducible_unique_and_excludes_development():
    mag = np.repeat([19., 21., 23., 25.], 100)
    excluded = np.arange(50)
    panel = validation_rows(mag, excluded)
    assert panel == validation_rows(mag, excluded)
    rows = np.concatenate(list(panel.values()))
    assert len(rows) == len(np.unique(rows)) == 64
    assert not np.isin(rows, excluded).any()
    for key, lo, hi in [('-inf:20', -np.inf, 20), ('20:22', 20, 22),
                        ('22:24', 22, 24), ('24:25.8', 24, 25.8)]:
        assert ((mag[panel[key]] >= lo) & (mag[panel[key]] < hi)).all()


def test_truth_matching_preserves_identity_and_rejects_missing_or_duplicate():
    frozen = pd.DataFrame(dict(source_case=[40, 40], source_input_index=[7, 2]))
    source = pd.DataFrame(dict(index_input=[2, 7], r_input=[27., 25.], Re_input=[1., 2.]))
    result = match_truth(frozen, source)
    assert result.r_input.tolist() == [25., 27.]
    with pytest.raises(ValueError, match='unmatched'):
        match_truth(frozen, source.iloc[:1])
    with pytest.raises(ValueError, match='duplicate'):
        match_truth(frozen, pd.concat([source, source.iloc[:1]]))
