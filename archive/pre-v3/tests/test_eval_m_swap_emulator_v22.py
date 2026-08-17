import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pytest

from scripts.eval_m_swap_emulator_v22 import read_dump_case_range


def test_read_dump_case_range_filters_before_returning(tmp_path):
    path = tmp_path / "dump.feather"
    pf.write_feather(
        pa.table(
            {
                "case": [40, 40, 41, 42, 43],
                "input_index": [0, 1, 2, 3, 4],
                "R_flow": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        ),
        path,
    )

    result = read_dump_case_range(
        str(path), ["case", "input_index", "R_flow"], min_case=41, max_case=43
    )

    np.testing.assert_array_equal(result["case"].to_numpy(), [41, 42])
    np.testing.assert_array_equal(result["input_index"].to_numpy(), [2, 3])
    np.testing.assert_allclose(result["R_flow"].to_numpy(), [0.3, 0.4])


def test_read_dump_case_range_rejects_empty_range(tmp_path):
    path = tmp_path / "dump.feather"
    pf.write_feather(pa.table({"case": [40], "input_index": [0]}), path)

    with pytest.raises(ValueError, match="no dump rows"):
        read_dump_case_range(
            str(path), ["case", "input_index"], min_case=50, max_case=60
        )
