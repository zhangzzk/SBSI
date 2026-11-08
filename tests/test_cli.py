import json

import pytest

from sbsi import cli


def test_cli_exposes_only_model_identity_commands():
    help_text = cli.build_parser().format_help()
    assert "show-model" in help_text
    assert "validate-model" in help_text
    assert "flow" not in help_text


def test_show_model_reports_v35_ensemble(capsys):
    assert cli.main(["show-model", "v3.5-like"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "V3.5-like"
    assert len(payload["flow_checkpoints"]) == 1
    assert len(payload["detection_classifiers"]) == 3
    assert payload["emulator_model"].endswith("trial9_base.json")


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError, match="V3.5-like"):
        cli.main(["show-model", "V3.3-like"])
