import torch

from scripts.train_scene_classifier import initial_model


def test_matched_initial_predictions_and_learnable_geometry():
    baseline, _ = initial_model(8)
    geometry, _ = initial_model(13)
    raw = torch.arange(52, dtype=torch.float32).reshape(4, 13)/100
    torch.testing.assert_close(baseline(raw[:, :8]), geometry(raw), atol=1e-7, rtol=1e-6)
    geometry(raw).sum().backward()
    assert torch.isfinite(geometry.net[0].weight.grad).all()
    assert torch.count_nonzero(geometry.net[0].weight.grad[:, 8:]) > 0
