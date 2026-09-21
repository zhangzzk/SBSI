"""Compare held-out distribution-secants without selection cuts or fitted offsets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.flow_distribution_response import (
    GaussianFourierFeatures, draw_feature_secant, feature_secant,
    independent_response_score,
)
from sbsi.measurement_model import ConditionalMeanFlow, ConditionalMeanFlowRA, load_measurement_model
from sbsi.training import build_shifted_context
from scripts.audit_flow_derivative_losses import RESCALE_KWARGS
from scripts.plot_flow_theta_validation import digest, load_validation


OUTPUT_NAMES = ['measured_ngmix_g1', 'measured_ngmix_g2',
                'measured_mag_auto', 'measured_log_flux_radius']


def observed_coordinates(frame, indices, suffix=''):
    """Physical measured coordinates, including log of the raw flux radius."""
    columns = []
    for index in indices:
        if index == 3:
            columns.append(np.log(frame['measured_flux_radius'+suffix].to_numpy(float)))
        else:
            columns.append(frame[OUTPUT_NAMES[index]+suffix].to_numpy(float))
    return np.column_stack(columns)


def coordinate_affine(model_transform, reference_transform, indices):
    """Convert model-standardized selected outputs to reference coordinates."""
    scale = np.asarray(reference_transform.scales, float)[indices]
    offset = np.asarray(reference_transform.means, float)[indices]
    return (np.asarray(model_transform.scales, float)[indices]/scale,
            (np.asarray(model_transform.means, float)[indices]-offset)/scale)


def common_samples(model, c0, c1, draws, seed):
    """One independent group, antithetic within group, identical latent per leg."""
    if isinstance(model, ConditionalMeanFlowRA):
        raise TypeError('RA model is outside this audit')
    generator = torch.Generator(device=c0.device).manual_seed(seed)
    half = torch.randn(len(c0), draws//2, model.target_dim, generator=generator,
                       device=c0.device, dtype=c0.dtype)
    z = torch.cat((half, -half), dim=1).reshape(-1, model.target_dim)
    result = []
    for context in [c0, c1]:
        expanded = context[:, None, :].expand(len(c0), draws, model.context_dim).reshape(-1, model.context_dim)
        if isinstance(model, ConditionalMeanFlow):
            y = model.flow.forward(z, model._flow_ctx(expanded))[0] + model._mu(expanded)
        else:
            y = model.forward(z, expanded)[0]
        result.append(y.reshape(len(c0), draws, model.target_dim))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g0-catalogue', type=Path, required=True)
    p.add_argument('--sheared-catalogue', type=Path, required=True)
    p.add_argument('--response-target', type=Path, required=True)
    p.add_argument('--reference-model', type=Path, required=True)
    p.add_argument('--model', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sample-rows', type=int, default=200000)
    p.add_argument('--draws-per-group', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--feature-seed', type=int, default=501)
    p.add_argument('--target-index', type=int, choices=range(4), action='append',
                   help='output coordinate to include; repeat for a joint audit (default: 3, log radius)')
    args = p.parse_args()
    indices = args.target_index if args.target_index is not None else [3]
    if len(indices) != len(set(indices)):
        p.error('target indices must be unique')
    if args.draws_per_group < 2 or args.draws_per_group % 2:
        raise ValueError('each independent group requires even draws >=2')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir/'summary.json').exists():
        raise FileExistsError('audit already complete')
    frame, table, gamma, shear, cases, counts = load_validation(args)
    take = np.sort(np.random.default_rng(8301).choice(len(frame), min(args.sample_rows,len(frame)), replace=False))
    frame, table = frame.iloc[take].reset_index(drop=True), table.iloc[take].reset_index(drop=True)
    gamma, shear = gamma[take], shear[take]
    reference = load_measurement_model(str(args.reference_model), device='cpu')
    reference_transform = reference.target_transform
    if reference_transform.target_names != OUTPUT_NAMES:
        raise ValueError('unexpected reference output coordinates')
    ref_mean = np.asarray(reference_transform.means, float)[indices]
    ref_scale = np.asarray(reference_transform.scales, float)[indices]
    del reference
    phi = GaussianFourierFeatures(len(indices), seed=args.feature_seed).to(device='cuda', dtype=torch.float32)
    np.save(args.output_dir/'frequencies.npy', phi.frequencies.cpu().numpy())
    report = {'counts':counts,'rows':len(frame),'cases':cases.tolist(),'reference_model':str(args.reference_model),
              'reference_sha256':digest(args.reference_model),
              'reference_transform':{'indices':indices, 'names':[OUTPUT_NAMES[j] for j in indices],
                                     'means':ref_mean.tolist(),'scales':ref_scale.tolist()},
              'feature_map':{'coordinate':'standardized log measured radius' if indices == [3] else 'joint standardized measured outputs',
                             'indices':indices,'bandwidths':[.25,.5,1.,2.],
                             'frequencies_per_band':32,'seed':args.feature_seed,'features':256},
              'draws_per_independent_group':args.draws_per_group,'sampling_seed':8301,'base_seed':97301,'models':[],
              'definition':'E[M_A.M_B-(M_A+M_B).T], an unbiased Monte Carlo estimator of conditional feature-response squared error up to a data-only constant. Lower is better; negative is allowed. Matched both-measured population; finite forward shear interval; not ConstGold m.'}
    if indices == [3]:
        report['reference_log_radius_transform'] = {'mean':float(ref_mean[0]),'scale':float(ref_scale[0])}
    results = table[['case','input_index']].copy()
    case_ids = np.sort(results.case.unique())
    ci = np.searchsorted(case_ids, results.case.to_numpy())
    boot = np.random.default_rng(20260906).multinomial(len(case_ids),np.full(len(case_ids),1/len(case_ids)),size=2000)
    count = np.bincount(ci,minlength=len(case_ids))
    baseline = None
    for mi,path in enumerate(args.model):
        bundle = load_measurement_model(str(path),device='cuda')
        model = bundle.model.eval()
        if bundle.target_transform.target_names != OUTPUT_NAMES:
            raise ValueError('unexpected measurement output coordinates')
        names = bundle.metadata['condition_features']
        scale, offset = [torch.as_tensor(v, device='cuda', dtype=torch.float32)
                         for v in coordinate_affine(bundle.target_transform, reference_transform, indices)]
        scores=[]; naive=[]; energies=[]
        with torch.no_grad():
            for start in range(0,len(frame),args.batch_size):
                stop = min(start+args.batch_size,len(frame));batch=frame.iloc[start:stop]
                directions=(gamma[start:stop,0]/shear[start:stop],gamma[start:stop,1]/shear[start:stop])
                c0=build_shifted_context(batch,(1.,0.),0.,bundle.condition_preprocessor,names,RESCALE_KWARGS)
                # The stored shear norms are checked by load_validation. The context
                # builder uses the same fixed 0.05 finite interval as the pair audits.
                c1=build_shifted_context(batch,directions,.05,bundle.condition_preprocessor,names,RESCALE_KWARGS)
                c0,c1=[torch.as_tensor(v,device='cuda',dtype=torch.float32) for v in [c0,c1]]
                h=torch.as_tensor(shear[start:stop],device='cuda',dtype=torch.float32)
                y0=(observed_coordinates(batch,indices)-ref_mean)/ref_scale
                y1=(observed_coordinates(batch,indices,'_sheared')-ref_mean)/ref_scale
                y0,y1=[torch.as_tensor(v,device='cuda',dtype=torch.float32) for v in [y0,y1]]
                target=feature_secant(phi(y0),phi(y1),h)
                estimates=[]
                for group in [0,1]:
                    draws=common_samples(model,c0,c1,args.draws_per_group,97301+2*(start//args.batch_size)+group)
                    u0,u1=[v[:,:,indices]*scale+offset for v in draws]
                    estimates.append(draw_feature_secant(phi,u0,u1,h))
                a,b=estimates
                scores.append(independent_response_score(a,b,target).cpu().numpy())
                naive.append(independent_response_score(a,a,target).cpu().numpy())
                energies.append(target.square().sum(-1).cpu().numpy())
        score=np.concatenate(scores).astype(float);assert np.isfinite(score).all()
        results[f'model{mi}_score']=score
        if baseline is None: baseline=score.copy()
        sums=np.bincount(ci,weights=score,minlength=len(case_ids))
        reps=(boot@sums)/(boot@count)
        diff=score-baseline
        ds=np.bincount(ci,weights=diff,minlength=len(case_ids))
        dr=(boot@ds)/(boot@count)
        entry={'path':str(path),'sha256':digest(path),'score':float(score.mean()),'case_se':float(reps.std(ddof=1)),
               'minus_first_model':float(diff.mean()),'difference_case_se':float(dr.std(ddof=1)),
               'difference_ci95':np.quantile(dr,[.025,.975]).tolist(),
               'naive_single_group_score':float(np.concatenate(naive).mean()),'target_squared_mean':float(np.concatenate(energies).mean())}
        report['models'].append(entry);print(json.dumps(entry),flush=True)
        del bundle,model
        torch.cuda.empty_cache()
    results.to_parquet(args.output_dir/'per_pair_scores.parquet',index=False)
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
