"""Measure latent-only and combined cell/latent gradient noise at frozen weights."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.flow_grid_full_cells import full_cell_response
from sbsi.flow_grid_response import GridResponsePopulation
from sbsi.measurement_model import build_flow


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.replicates < 2 or args.cells < 1:
        parser.error("at least two replicates and one cell required")
    if not torch.cuda.is_available():
        raise RuntimeError("full-population gradient audit requires scheduled CUDA")
    manifest = json.loads(Path(args.manifest).read_text())
    for path, expected in manifest["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source changed: {path}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cached = load(Path(manifest["prepared"])/"train.pt")
    context = cached["context"].cuda()
    result = dict(arguments=vars(args), models={},
                  manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  definition="Frozen weights; complete selected cells; normalized independent cross-quadratic gradient; no optimizer steps",
                  comparison="fixed cells vary latents only; random cells vary both cells and latents; these variances are not directly subtracted")
    for name, entry in manifest["models"].items():
        path = Path(entry["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"checkpoint changed: {path}")
        checkpoint = load(path)
        model = build_flow(checkpoint["model_config"]).cuda()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        population = GridResponsePopulation(context, cached, checkpoint["target_transform"]["scales"][:2])
        trials = {}
        for selection in ("random_cells", "fixed_cells"):
            gradients, records = [], []
            torch.cuda.reset_peak_memory_stats()
            for replicate in range(args.replicates):
                seed = 14001+replicate*100003
                cell_seed = seed if selection == "random_cells" else 14001
                cells = np.random.default_rng(cell_seed).permutation(population.n_cells)[:args.cells]
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                estimate = full_cell_response(model, population, cells, args.draws,
                                              chunk_size=args.chunk_size, seed=seed+100000,
                                              backward=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                                      for p in model.parameters()]).detach().cpu().double()
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError("nonfinite full-cell response gradient")
                gradients.append(gradient)
                records.append(dict(cells=cells.tolist(), loss=estimate["loss"], seconds=elapsed,
                                    pairs_per_group=estimate["pairs_per_group"],
                                    mean_a=estimate["mean_a"].cpu().tolist(),
                                    mean_c=estimate["mean_c"].cpu().tolist()))
            gradients = torch.stack(gradients)
            mean = gradients.mean(0)
            noise = float((gradients-mean).square().sum()/(args.replicates-1))
            signal = float(mean.square().sum())
            summary = dict(records=records, mean_gradient_norm=signal**.5,
                           gradient_noise_rms=noise**.5,
                           naive_noise_to_mean_ratio=(noise/signal)**.5 if signal > 0 else None,
                           estimated_true_gradient_norm_sq=signal-noise/args.replicates,
                           loss_mean=float(np.mean([r["loss"] for r in records])),
                           loss_std=float(np.std([r["loss"] for r in records], ddof=1)),
                           mean_seconds=float(np.mean([r["seconds"] for r in records])),
                           peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
            trials[selection] = summary
            result["models"][name] = dict(checkpoint_sha256=entry["sha256"], trials=trials)
            np.savez_compressed(output/f"{name}_{selection}_gradients.npz",
                                gradients=gradients.float().numpy())
            temporary = output/"results.tmp.json"
            temporary.write_text(json.dumps(result, indent=2)+"\n")
            temporary.replace(output/"results.json")
            print(json.dumps(dict(model=name, selection=selection,
                                  **{k:v for k,v in summary.items() if k != "records"})), flush=True)
        del model, population, checkpoint, gradients, gradient
    print("FULL_CELL_GRADIENT_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
