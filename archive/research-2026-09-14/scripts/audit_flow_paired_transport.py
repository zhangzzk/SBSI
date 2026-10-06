#!/usr/bin/env python
"""Audit the flow-induced coupling on matched, held-out simulation pairs.

Observed zero-leg measurements initialize the latent. These are conditional
transport diagnostics, never independent forward-model closure estimates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.flow_transport import paired_flow_transport
from sbsi.measurement_model import load_measurement_model
from sbsi.training import build_shifted_context
from scripts.plot_flow_theta_validation import load_validation, digest
from scripts.audit_flow_derivative_losses import RESCALE_KWARGS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g0-catalogue", type=Path, required=True)
    p.add_argument("--sheared-catalogue", type=Path, required=True)
    p.add_argument("--response-target", type=Path, required=True)
    p.add_argument("--model", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("*.json")) or any(args.output_dir.glob("*.parquet")):
        raise FileExistsError("refusing to overwrite an existing audit")
    frame, table, gamma, g, cases, counts = load_validation(args)
    columns = ["measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto"]
    y0 = np.column_stack([frame[columns].to_numpy(float), np.log(frame["measured_flux_radius"])])
    y1 = np.column_stack([frame[[c+"_sheared" for c in columns]].to_numpy(float),
                          np.log(frame["measured_flux_radius_sheared"])])
    actual_delta = (y1-y0)/g[:,None]
    for j in range(4):
        table[f"measured_response_{j}"] = actual_delta[:,j]
    report = {"counts": counts, "cases": cases.tolist(), "models": [],
              "definition": "Observed-zero-leg initialized latent transport; not forward closure"}
    for mi, path in enumerate(args.model):
        bundle = load_measurement_model(str(path), device=args.device)
        minimum = float(bundle.metadata.get("primary_re_min", .5))
        if (frame["Re_input_p"] <= minimum).any():
            raise ValueError("audit population extends below model's training-size floor")
        names = bundle.metadata["condition_features"]
        predictions, identity_errors = [], []
        with torch.no_grad():
            for start in range(0, len(frame), args.batch_size):
                stop = min(start+args.batch_size, len(frame))
                batch = frame.iloc[start:stop]
                c0 = build_shifted_context(batch, (1.,0.), 0., bundle.condition_preprocessor,
                                           names, RESCALE_KWARGS)
                directions = (gamma[start:stop,0]/g[start:stop], gamma[start:stop,1]/g[start:stop])
                c1 = build_shifted_context(batch, directions, .05, bundle.condition_preprocessor,
                                           names, RESCALE_KWARGS)
                x = bundle.target_transform.transform_array(y0[start:stop])
                x, c0, c1 = [torch.as_tensor(v, dtype=torch.float32, device=args.device)
                             for v in (x,c0,c1)]
                moved = paired_flow_transport(bundle.model, x, c0, c1).cpu().numpy()
                same = paired_flow_transport(bundle.model, x, c0, c0).cpu().numpy()
                identity_errors.append(np.max(np.abs(same-x.cpu().numpy()), axis=1))
                predictions.append(bundle.target_transform.inverse_transform_array(moved))
        predicted = np.concatenate(predictions)
        identity_error = np.concatenate(identity_errors)
        if not np.isfinite(predicted).all():
            raise ValueError("nonfinite transported measurements")
        response = (predicted-y0)/g[:,None]
        for j in range(4):
            table[f"model{mi}_response_{j}"] = response[:,j]
        error = response-actual_delta
        summary = {"path": str(path), "sha256": digest(path),
                   "response_rmse": np.sqrt(np.mean(error**2,axis=0)).tolist(),
                   "response_mean_error": error.mean(axis=0).tolist(),
                   "identity_max_standardized_error": float(identity_error.max()),
                   "identity_error_quantiles": np.quantile(identity_error,[.5,.99,.999,1]).tolist(),
                   "response_error_abs_quantiles": np.quantile(np.abs(error),[.5,.9,.99,.999,1],axis=0).tolist()}
        report["models"].append(summary)
        print(json.dumps(summary),flush=True)
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["zero_response_rmse"] = np.sqrt(np.mean(actual_delta**2,axis=0)).tolist()
    table.to_parquet(args.output_dir/"per_galaxy.parquet",index=False)
    (args.output_dir/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
