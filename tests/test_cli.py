import json

import pytest

from sbsi import cli


def test_cli_exposes_only_model_identity_commands():
    help_text = cli.build_parser().format_help()
    assert "show-model" in help_text
    assert "validate-model" in help_text
    assert "flow" not in help_text


def test_show_model_reports_the_bundled_model(capsys):
    assert cli.main(["show-model", "v3.9-like"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "V3.9-like"
    assert len(payload["flow_checkpoints"]) == 1
    assert len(payload["detection_classifiers"]) == 1
    assert payload["emulator_model"].endswith("v3_9_like/blending/model.json")


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError, match="V3.9-like"):
        cli.main(["show-model", "V3.3-like"])
