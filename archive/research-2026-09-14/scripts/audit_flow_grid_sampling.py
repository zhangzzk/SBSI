"""Locate Monte Carlo instability in full-population grid-response validation.

Read immutable checkpoint copies and prepared held-out pairs. No training or
selection changes. Expensive evaluation belongs on a scheduled GPU worker.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.flow_grid_response import GridResponsePopulation
from sbsi.flow_training import complete_flow_context_means
from sbsi.measurement_model import build_flow


def summarize_groups(contributions, cell_ids, counts, target, precision):
    """Decompose the normalized cross-score and each row's MC disagreement.

    The row contribution is a response regression contribution, not a
    per-galaxy identifiable 2x2 response matrix. Squared row influence is a
    ranking diagnostic and does not add up to the loss (cross terms exist).
    """
    a, c = (np.asarray(v, float) for v in contributions)
    cells = np.asarray(cell_ids, int)
    counts = np.asarray(counts, int)
    n_cells = len(counts)
    if (a.shape != c.shape or a.shape != (len(cells), 4)
            or not np.isfinite(a).all() or not np.isfinite(c).all()
            or not np.array_equal(np.bincount(cells, minlength=n_cells), counts)
            or np.any(counts <= 0)):
        raise ValueError("need finite aligned full-population contributions and counts")
    means = []
    for values in (a, c):
        sums = np.stack([np.bincount(cells, weights=values[:, d], minlength=n_cells)
                         for d in range(4)], axis=1)
        means.append(sums/counts[:, None])
    x, y = means[0]-target, means[1]-target
    factor = .5/(4*n_cells)
    quadratic = lambda u, v: factor*np.einsum("bi,bij,bj->b", u, precision, v)
    cross = quadratic(x, y)
    average = quadratic((x+y)/2, (x+y)/2)
    disagreement = quadratic(x-y, x-y)
    np.testing.assert_allclose(average, cross+disagreement/4, rtol=1e-8, atol=1e-10)
    row_difference = (a-c)/counts[cells, None]
    influence = factor*np.einsum("bi,bij,bj->b", row_difference, precision[cells], row_difference)
    # This nonnegative ranking diagnoses concentration without modifying data.
    influence = np.maximum(influence, 0.)
    order = np.argsort(influence)[::-1]
    total_influence = float(influence.sum())
    concentration = {str(k): (float(influence[order[:k]].sum()/total_influence)
                              if total_influence else 0.) for k in (1, 10, 100, 1000)}
    return dict(mean_a=means[0], mean_c=means[1], cross=cross, average=average,
                group_a=quadratic(x,x), group_c=quadratic(y,y),
                disagreement=disagreement, row_influence=influence,
                top_rows=order[:1000], concentration=concentration)


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def evaluate(model, population, shape_means, draws, batch_size, seed):
    """Two independent CRN streams; keep full row contributions for diagnosis."""
    n = len(population.pairs)
    offset = torch.as_tensor(shape_means, device=population.context.device)
    groups, extrema, outside, totals = [], [], [], []
    for stream in range(2):
        contribution = np.empty((n,4), float)
        row_maximum = np.empty(n, float)
        outside_count, total = 0, 0
        for index, start in enumerate(range(0,n,batch_size)):
            stop = min(start+batch_size,n)
            pair = population.pairs[start:stop]
            means, samples = complete_flow_context_means(
                model,[population.context[pair[:,0]],population.context[pair[:,1]]],
                draws,seed=seed+2*index+stream,return_samples=True)
            delta = ((means[1][:,:2]-means[0][:,:2])*population.scales).double()
            values = (delta[:,:,None]*population.design[start:stop,None,:]).reshape(-1,4)
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"nonfinite contribution at rows {start}:{stop}, stream {stream}")
            contribution[start:stop] = values.cpu().numpy()
            maximum = torch.zeros(stop-start,device=population.context.device)
            for sample in samples:
                physical = sample[:,:,:2]*population.scales+offset
                amplitude = physical.square().sum(-1).sqrt()
                if not bool(torch.isfinite(amplitude).all()):
                    raise ValueError("nonfinite shape draw")
                maximum = torch.maximum(maximum,amplitude.max(1).values)
                outside_count += int((amplitude>1).sum())
                total += amplitude.numel()
            row_maximum[start:stop] = maximum.cpu().numpy()
        groups.append(contribution)
        extrema.append(row_maximum)
        outside.append(outside_count)
        totals.append(total)
    return groups, extrema, outside, totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--draws",type=int,nargs="+",default=[16,64])
    parser.add_argument("--batch-size",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=7301)
    args=parser.parse_args(argv)
    if args.batch_size<1 or any(d<2 or d%2 for d in args.draws):
        raise ValueError("positive batch size and even antithetic draw counts required")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=manifest["audit_source_sha256"]:
        raise ValueError("audit source changed since submission")
    for path,expected in manifest.get("source_hashes",{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"source changed: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("full population sampling audit requires a scheduled GPU")
    prepared=Path(manifest["prepared"])
    payload=load(prepared/"validation.pt")
    with np.load(prepared/"validation_targets.npz") as data:
        covariance=data["covariance"]
    context=payload["context"].cuda()
    results={"manifest":str(args.manifest),"models":{},"seed":args.seed,
             "population":"all held-out matched four-output-usable pairs", "pairs":len(payload["pairs"])}
    for name,entry in manifest["models"].items():
        path=Path(entry["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("checkpoint hash changed")
        checkpoint=load(path)
        transform=checkpoint["target_transform"]
        pop=GridResponsePopulation(context,payload,transform["scales"][:2])
        model=build_flow(checkpoint["model_config"]).cuda()
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        model.eval()
        measured=payload["values"][:,:2].numpy()*np.asarray(transform["scales"][:2])+np.asarray(transform["means"][:2])
        measured_norm=np.linalg.norm(measured,axis=1)
        model_results={"epoch":checkpoint.get("metadata",{}).get("epoch"),
                       "measured_leg_rows_outside_unit_disk":int((measured_norm>1).sum()),
                       "measured_shape_amplitude_max":float(measured_norm.max()),"draw_counts":{}}
        for draws in args.draws:
            groups,extrema,outside,totals=evaluate(model,pop,transform["means"][:2],draws,args.batch_size,args.seed)
            detail=summarize_groups(groups,payload["bin_ids"],payload["counts"],payload["target"],payload["precision"])
            top=detail["top_rows"]
            np.savez(args.output_dir/f"{name}_d{draws}.npz",
                     **{k:v for k,v in detail.items() if k not in ("concentration","row_influence")},
                     top_pair_keys=payload["pair_keys"][top],top_cells=payload["bin_ids"][top],
                     top_contribution_a=groups[0][top],top_contribution_c=groups[1][top],
                     top_row_influence=detail["row_influence"][top],
                     top_max_shape_a=extrema[0][top],top_max_shape_c=extrema[1][top],covariance=covariance)
            summary={k:float(detail[k].sum()) for k in ["cross","average","group_a","group_c","disagreement"]}
            summary.update(row_disagreement_concentration=detail["concentration"],
                           max_draw_shape_amplitude=[float(x.max()) for x in extrema],
                           per_galaxy_max_shape_quantiles=[np.quantile(x,[.5,.99,.999,.9999,1]).tolist() for x in extrema],
                           draws_outside_unit_disk=outside,total_shape_draws=totals)
            model_results["draw_counts"][str(draws)]=summary
            results["models"][name]=model_results
            (args.output_dir/"results.json").write_text(json.dumps(results,indent=2)+"\n")
            print(json.dumps({"model":name,"draws":draws,**summary}),flush=True)
        del model,pop,checkpoint
    print("FULL_GRID_SAMPLING_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":
    main()
