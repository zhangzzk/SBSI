import json

from scripts.continue_minimal_selected_shape_flow import write_json


def test_continuation_json_is_atomic_and_strict(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"response_used": False, "score": 0.5})
    assert json.loads(path.read_text()) == {"response_used": False, "score": 0.5}
