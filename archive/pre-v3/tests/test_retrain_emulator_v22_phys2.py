import copy

import pytest

from scripts import retrain_emulator_v22_phys2 as trainer


def valid_config():
    return {
        "training": {
            "model_tag": trainer.PHYS2_TAG,
            "features": list(trainer.PHYS2_FEATURES),
            "regression_cuts": copy.deepcopy(trainer.EXPECTED_CUTS),
        }
    }


def test_assert_phys2_config_accepts_exact_ablation():
    trainer.assert_phys2_config(valid_config())


@pytest.mark.parametrize("drift", ["tag", "features", "cuts"])
def test_assert_phys2_config_rejects_scientific_drift(drift):
    cfg = valid_config()
    if drift == "tag":
        cfg["training"]["model_tag"] = "wrong"
    elif drift == "features":
        cfg["training"]["features"].append("sersic_n_input_p")
    else:
        cfg["training"]["regression_cuts"][1][1] = 26.0
    with pytest.raises(SystemExit):
        trainer.assert_phys2_config(cfg)
