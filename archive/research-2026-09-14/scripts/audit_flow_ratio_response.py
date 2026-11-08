"""Cut-free balanced density-ratio BCE on a fixed matched validation sample.

The population specification supplies only case split and true-domain cuts;
no grid target values enter the score. This is not full ConstGold closure.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.flow_ratio import paired_ratio_losses
from sbsi.measurement_model import load_measurement_model
from sbsi.training import build_shifted_context
from scripts.audit_flow_derivative_losses import RESCALE_KWARGS
from scripts.plot_flow_theta_validation import load_validation, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g0-catalogue', type=Path, required=True)
    p.add_argument('--sheared-catalogue', type=Path, required=True)
    p.add_argument('--population-spec', type=Path, dest='response_target', required=True)
    p.add_argument('--model', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sample-rows', type=int, default=200000)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError('audit output must be empty')
    frame, table, gamma, shear, cases, counts = load_validation(args)
    take = np.sort(np.random.default_rng(8301).choice(
        len(frame), min(args.sample_rows, len(frame)), replace=False))
    frame, table = frame.iloc[take].reset_index(drop=True), table.iloc[take].reset_index(drop=True)
    gamma, shear = gamma[take], shear[take]
    cols = ['measured_ngmix_g1', 'measured_ngmix_g2', 'measured_mag_auto']
    y0 = np.column_stack((frame[cols], np.log(frame['measured_flux_radius'])))
    y1 = np.column_stack((frame[[c+'_sheared' for c in cols]], np.log(frame['measured_flux_radius_sheared'])))
    case_ids, ci = np.unique(frame['case'], return_inverse=True)
    count = np.bincount(ci)
    rng = np.random.default_rng(20260906)
    boot = rng.multinomial(len(case_ids), np.ones(len(case_ids))/len(case_ids), size=2000)
    baseline = None
    models = []
    for index, path in enumerate(args.model):
        bundle = load_measurement_model(str(path), device=args.device)
        if bundle.target_transform.target_names != cols+['measured_log_flux_radius']:
            raise ValueError('ratio audit requires the four standard measurement outputs')
        names = bundle.metadata['condition_features']
        scores, densities = [], []
        with torch.no_grad():
            for start in range(0, len(frame), args.batch_size):
                end = min(start+args.batch_size, len(frame))
                batch = frame.iloc[start:end]
                directions = (gamma[start:end,0]/shear[start:end], gamma[start:end,1]/shear[start:end])
                c0 = build_shifted_context(batch, (1.,0.), 0., bundle.condition_preprocessor, names, RESCALE_KWARGS)
                c1 = build_shifted_context(batch, directions, .05, bundle.condition_preprocessor, names, RESCALE_KWARGS)
                x0, x1 = [bundle.target_transform.transform_array(y[start:end]) for y in (y0,y1)]
                x0,c0,x1,c1 = [torch.as_tensor(v, device=args.device, dtype=torch.float32) for v in (x0,c0,x1,c1)]
                scores.append(paired_ratio_losses(bundle.model,(x0,c0,x1,c1)).cpu().numpy())
                # Physical output coordinates, to compare different model transforms.
                jacobian = float(np.log(bundle.target_transform.scales).sum())
                nll = -.5*(bundle.model.log_prob(x0,c0)+bundle.model.log_prob(x1,c1))+jacobian
                densities.append(nll.cpu().numpy())
        scores, densities = np.concatenate(scores).astype(float), np.concatenate(densities).astype(float)
        if not np.isfinite(scores).all() or not np.isfinite(densities).all():
            raise ValueError('nonfinite ratio or NLL audit')
        if baseline is None:
            baseline = scores.copy()
        sums = np.bincount(ci, weights=scores, minlength=len(case_ids))
        diff = np.bincount(ci, weights=scores-baseline, minlength=len(case_ids))
        replicas, differences = (boot@sums)/(boot@count), (boot@diff)/(boot@count)
        table[f'model{index}_bce'] = scores
        table[f'model{index}_matched_nll'] = densities
        models.append({'path':str(path),'sha256':digest(path),'bce':float(scores.mean()),
                       'case_se':float(replicas.std(ddof=1)),
                       'minus_first_model':float((scores-baseline).mean()),
                       'difference_case_se':float(differences.std(ddof=1)),
                       'difference_ci95':np.quantile(differences,[.025,.975]).tolist(),
                       'matched_physical_output_nll':float(densities.mean())})
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    table.to_parquet(args.output_dir/'per_pair_scores.parquet', index=False)
    summary={'counts':counts,'rows':len(frame),'cases':case_ids.tolist(),'models':models,
             'sampling_seed':8301,'bootstrap_seed':20260906,'bootstrap_replicates':2000,
             'logit':'log q(x|S_0.05 U)-log q(x|S_0 U), no division by shear',
             'null_bce':float(np.log(2)),
             'definition':'Balanced BCE over matched both-measured pairs, lower better; full four-output ratio; no model sampling.',
             'limitation':'Both-measured finite-forward diagnostic, not per-leg likelihood or full ConstGold m.'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
