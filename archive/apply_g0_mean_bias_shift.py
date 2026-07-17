"""Embed a g=0-derived global additive mean shift into a flow checkpoint.

This is a direct checkpoint edit, not an inference-time sidecar.  It estimates
the remaining zero-shear measured-flow residual on g=0 fit cases using the
actual saved-flow sampler, then adds that residual to the final mean-head bias.

Because the edit is a constant output bias in the mean head, it changes the
additive first moment but not the local response derivative carried by the mean
head.  It uses g=0 rows only.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    TargetStandardizer,
    build_flow,
    load_measurement_model,
    save_measurement_model,
)
from scripts.diagnose_additive_origin import attach_lookups, flow_mean, load_g0  # noqa: E402
from scripts.response_ratio_diagnostic import _shape_target_indices  # noqa: E402
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402


def _final_bias_parameter(mean_net):
    import torch.nn as nn

    if isinstance(mean_net, nn.Linear):
        return mean_net.bias
    for module in reversed(list(mean_net.modules())):
        if isinstance(module, nn.Linear):
            return module.bias
    raise TypeError("could not find a Linear bias in mean_net")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-model", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_v1.pt")
    ap.add_argument("--output", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_bias_v1.pt")
    ap.add_argument("--g0-catalogue", default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.0_train_c0-39.feather")
    ap.add_argument("--blend-lookup", default="results/blend_lookup_extnbrho_c0-39.feather")
    ap.add_argument("--crowd-flux-lookup", default="results/crowd_flux_c0-39.feather")
    ap.add_argument("--ood-lookup", default="results/ood_split_c0-39.feather")
    ap.add_argument("--nn-lookup", default="results/nn_dist_const_c0-39.feather")
    ap.add_argument("--fit-max-case", type=int, default=19)
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    ap.add_argument("--model-rows", type=int, default=1_500_000)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.input_model, device=device)
    print(f"device={device} input={args.input_model}")

    g0 = attach_lookups(load_g0(args), args, overwrite=False)
    g0 = g0[g0["case"] <= args.fit_max_case].reset_index(drop=True)
    g0 = flow_mean(bundle, g0, args)
    residual = np.array([
        np.nanmean(g0["measured_c1"].to_numpy(float) - g0["flow_c1"].to_numpy(float)),
        np.nanmean(g0["measured_c2"].to_numpy(float) - g0["flow_c2"].to_numpy(float)),
    ], dtype=np.float32)
    print(f"g0 fit residual to embed: c1={residual[0]:+.6f} c2={residual[1]:+.6f}")

    try:
        checkpoint = torch.load(args.input_model, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.input_model, map_location=device)
    condition_preprocessor = TabularPreprocessor.from_state(checkpoint["condition_preprocessor"])
    target_transform = TargetStandardizer.from_state(checkpoint["target_transform"])
    model_config = dict(checkpoint["model_config"])
    model = build_flow(model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    shape_idx = _shape_target_indices(target_transform.target_names)
    bias = _final_bias_parameter(model.mean_net)
    with torch.no_grad():
        for j, raw_delta in zip(shape_idx, residual):
            bias[j] += float(raw_delta) / float(target_transform.scales[j])

    metadata = dict(checkpoint.get("metadata", {}))
    metadata["g0_global_mean_bias_shift"] = {
        "input_model": args.input_model,
        "g0_catalogue": args.g0_catalogue,
        "fit_max_case": int(args.fit_max_case),
        "residual_raw_c1_c2": residual,
        "model_rows": int(args.model_rows),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_measurement_model(args.output, model, condition_preprocessor, target_transform, model_config, metadata=metadata)
    print(f"wrote {args.output}")
    print("APPLY_G0_MEAN_BIAS_SHIFT_DONE")


if __name__ == "__main__":
    main()
