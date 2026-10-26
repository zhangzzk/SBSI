import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/train_minimal_rblend_flow_condition.py"
SPEC = importlib.util.spec_from_file_location("minimal_rblend_flow_condition", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_function_preserving_dead_flag_replacement():
    config = {
        "flow_type": "physical_disk_affine", "target_dim": 4, "context_dim": 16,
        "hidden_dim": 8, "n_layers": 1, "n_flows": 2, "scale_limit": 1.0,
        "activation": "silu", "mean_hidden": 0, "disk_map": "radial_tanh",
        "disk_shape_means": [0.0, 0.0], "disk_shape_scales": [0.3, 0.3],
        "disk_coordinate_means": [0.0, 0.0], "disk_coordinate_scales": [0.5, 0.5],
        "physical_map": "softplus", "physical_means": [3.0, 100.0],
        "physical_scales": [2.0, 80.0], "physical_units": [2.0, 80.0],
        "physical_coordinate_means": [1.0, 0.0], "physical_coordinate_scales": [1.0, 2.0],
    }
    torch.manual_seed(11)
    old = MODULE.build_flow(config)
    with torch.no_grad():
        for value in old.parameters():
            value.normal_(0, 0.1)
    parent = {"model_config": config, "state_dict": old.state_dict()}
    values = torch.tensor([[0.1, -0.2, 0.5, 1.0], [-0.2, 0.1, -0.3, 0.4]],
                          dtype=torch.float64)
    context8 = torch.randn(2, 8)
    expected = old.log_prob(values, torch.cat((context8, torch.zeros(2, 8)), 1))
    for dim in (8, 9):
        model, mapped_config, layers = MODULE.map_parent_model(parent, dim)
        compact = context8 if dim == 8 else torch.cat((context8, torch.randn(2, 1)), 1)
        torch.testing.assert_close(model.log_prob(values, compact), expected, rtol=0, atol=0)
        assert mapped_config["context_dim"] == dim
        assert layers == config["n_flows"]
