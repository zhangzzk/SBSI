import json

from scripts.refine_minimal_selected_shape_flow import write_json


def test_refinement_json_is_strict(tmp_path):
    path = tmp_path / "decision.json"
    write_json(path, {"response_used": False, "ready": True})
    assert json.loads(path.read_text())["ready"] is True
