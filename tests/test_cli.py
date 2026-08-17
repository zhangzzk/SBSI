import json
from dataclasses import replace

import pytest

from sbs_shear import cli
from sbs_shear import flow
from sbs_shear.flow import FlowTrainingConfig, FlowTrial


def _training_yaml(tmp_path):
    catalogue = tmp_path / "train.feather"
    response = tmp_path / "response.npz"
    coupling = tmp_path / "coupling.npz"
    for path in (catalogue, response, coupling):
        path.touch()
    return catalogue, response, coupling


def test_flow_train_cli_loads_explicit_yaml(monkeypatch, tmp_path, capsys):
    catalogue, response, coupling = _training_yaml(tmp_path)
    output = tmp_path / "flow.pt"
    config_path = tmp_path / "flow.yaml"
    config_path.write_text(
        f"""training:
  catalogue: {catalogue}
  output: {output}
  response_target: {response}
  coupling_target: {coupling}
  seed: 507
  learning_rate: 0.0005
""",
        encoding="utf-8",
    )
    seen = {}

    def fake_train(config):
        seen["config"] = config
        return config.swa_output

    monkeypatch.setattr(cli, "train_flow", fake_train)
    assert cli.main(["flow", "--config", str(config_path), "--mode", "train"]) == 0
    assert seen["config"].catalogue == catalogue
    assert seen["config"].seed == 507
    assert seen["config"].learning_rate == 0.0005
    assert capsys.readouterr().out.strip() == str(seen["config"].swa_output)


def test_flow_tune_cli_writes_ranked_manifest(monkeypatch, tmp_path):
    catalogue, response, coupling = _training_yaml(tmp_path)
    validation = tmp_path / "validation.feather"
    validation.touch()
    results = tmp_path / "tuning.json"
    output_a = tmp_path / "trial_a.pt"
    output_b = tmp_path / "trial_b.pt"
    config_path = tmp_path / "flow.yaml"
    config_path.write_text(
        f"""training:
  catalogue: {catalogue}
  output: {tmp_path / 'base.pt'}
  response_target: {response}
  coupling_target: {coupling}
tuning:
  validation_catalogue: {validation}
  scorer: project.validation:score_flow
  results: {results}
  candidates:
    - output: {output_a}
      learning_rate: 0.0005
    - output: {output_b}
      learning_rate: 0.0007
""",
        encoding="utf-8",
    )
    scorer = object()
    monkeypatch.setattr(cli, "_load_callable", lambda reference: scorer)

    def fake_tune(base, candidates, *, validation_catalogue, scorer):
        assert validation_catalogue == validation
        assert scorer is globals_scorer
        configs = [replace(base, **candidate) for candidate in candidates]
        return [FlowTrial(config=configs[1], score=0.1), FlowTrial(config=configs[0], score=0.2)]

    globals_scorer = scorer
    monkeypatch.setattr(cli, "tune_flow", fake_tune)
    assert cli.main(["flow", "--config", str(config_path), "--mode", "tune"]) == 0

    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["best_checkpoint"] == str(output_b.with_name("trial_b_swaavg.pt"))
    assert [trial["score"] for trial in payload["trials"]] == [0.1, 0.2]
    assert payload["trials"][0]["training"]["output"] == str(output_b)


def test_cli_exposes_one_config_driven_flow_command():
    parser = cli.build_parser()
    assert "flow" in parser.format_help()
    assert "train-flow" not in parser.format_help()


def test_tuning_preflights_all_outputs_before_training(monkeypatch, tmp_path):
    catalogue, response, coupling = _training_yaml(tmp_path)
    occupied = tmp_path / "occupied.pt"
    occupied.touch()
    base = FlowTrainingConfig(
        catalogue=catalogue,
        output=tmp_path / "base.pt",
        response_target=response,
        coupling_target=coupling,
    )
    calls = []
    monkeypatch.setattr(flow, "train_flow", lambda config: calls.append(config))

    with pytest.raises(FileExistsError, match="occupied.pt"):
        flow.tune_flow(
            base,
            [{"output": tmp_path / "free.pt"}, {"output": occupied}],
            validation_catalogue=tmp_path / "validation.feather",
            scorer=lambda checkpoint, catalogue: 0.0,
        )
    assert calls == []
