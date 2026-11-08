import json

from _script_loader import load_script_module


SCREEN = load_script_module("summarize_flow_coupling_ladder.py")


def _write_audit(path, model, *, nll, shape, photo):
    payload = {
        "model": str(model),
        "n_rows": 10,
        "validation_cases": [1, 2],
        "nll_balanced": nll,
        "shape_response_loss": shape,
        "magnitude_derivative_loss": 0.25 * photo,
        "log_size_derivative_loss": 0.75 * photo,
        "photometric_derivative_loss": photo,
    }
    path.write_text(json.dumps(payload))


def test_coupling_screen_selects_lowest_photometric_loss_inside_hard_gates(tmp_path):
    baseline_model = tmp_path / "baseline.pt"
    baseline_model.write_bytes(b"baseline")
    baseline_audit = tmp_path / "baseline.json"
    _write_audit(baseline_audit, baseline_model, nll=0.0, shape=1.0, photo=1.0)

    root = tmp_path / "ladder"
    root.mkdir()
    for index, (tag, _) in enumerate(SCREEN.LADDER):
        model = root / f"{tag}.pt"
        model.write_bytes(tag.encode())
        if tag == "t0p5":
            values = (0.01, 1.01, 0.8)
        elif tag == "t1":
            values = (0.04, 1.04, 0.5)
        else:
            values = (0.06 + index / 100, 1.06, 0.2)
        audit = root / f"heldout_derivative_losses_E_r450_{tag}_s501.json"
        _write_audit(audit, model, nll=values[0], shape=values[1], photo=values[2])

    base_likelihood = tmp_path / "likelihood.json"
    base_likelihood.write_text(json.dumps({
        "schema_version": 1,
        "release": "base",
        "description": "base",
        "measurement_model": {"path": None, "sha256": "old"},
    }))
    output = tmp_path / "summary.json"
    likelihood_output = tmp_path / "selected_likelihood.json"
    SCREEN.summarize(
        root,
        baseline_audit,
        [baseline_audit],
        base_likelihood,
        output,
        likelihood_output,
    )

    summary = json.loads(output.read_text())
    likelihood = json.loads(likelihood_output.read_text())
    assert summary["selected"]["tag"] == "t1"
    assert summary["selected"]["eligible"] is True
    assert likelihood["release"] == "v3.3-flow-retrain-screen"
    assert likelihood["measurement_model"]["sha256"] == summary["selected"]["model_sha256"]
