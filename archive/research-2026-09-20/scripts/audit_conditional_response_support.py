"""Check physical shape support of fixed and output-conditioned blend shifts.

These are conditional-flow draws on two fixed truth-parent scenes. Counts do
not establish population calibration or a classifier-weighted escape bound.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.flow_guard_response import physical_context_draws
from sbsi.measurement_model import load_measurement_model
from sbsi.output_conditioned_response import OutputConditionedResponse, predict_scene_draw_responses
from scripts.evaluate_constgold_fixed_g0_response import shear_frames
from scripts.evaluate_constgold_truth_parent import CUTS, load_candidate_pairs, output_cuts
from scripts.train_fixed_g0_flow import sha256, write_json


@torch.no_grad()
def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('conditional-flow support audit requires Slurm')
    data = Path('/project/ls-gruen/users/zekang.zhang')
    cache = data/'sbsi_caches'
    root = cache/'output_conditioned_response_20260917_v1'
    output = root/'conditional_support.json'
    if output.exists():
        raise FileExistsError(output)
    parent = cache/'constgold_truth_parent_20260917_v1/parent_v2'
    flow = cache/'full_domain_true_mag26_guard_v1/flow/guard/selected.pt'
    model = load_measurement_model(flow, device=torch.device('cuda'))
    model.model.eval()
    response = OutputConditionedResponse(data/'blendemu_runs/output_conditioned_response_20260917_v1/models_cpu/output_conditioned',
                                         nthread=8, device='cuda')
    lookup = pd.read_feather(cache/'constgold_truth_parent_20260917_v1/rblend/cases40_49.feather')
    reports = []
    for case in (40, 41):
        path = parent/f'case{case:03d}.npz'
        with np.load(path) as arrays:
            ids, context = arrays['input_index'], arrays['context'].astype(float)
        old = lookup.loc[lookup['case'] == case].set_index('input_index').reindex(ids)['R_blend'].to_numpy()
        if not np.isfinite(old).all():
            raise ValueError('unmatched baseline responses')
        pairs, _ = load_candidate_pairs(root/'pairs', case, path, ids, old)
        pair_ids = pairs['index_input_p'].to_numpy()
        frames = list(shear_frames(context, .02))
        contexts = [model.context_tensor(frame) for frame in frames]
        sums = np.zeros((2, 3, 3, 4))  # leg, cut, recipe, mass/bad/newbad/bad e1
        for block, start in enumerate(range(0, len(ids), 1024)):
            stop = min(len(ids), start+1024)
            generated = physical_context_draws(model.model, [x[start:stop] for x in contexts], 64,
                                                7301+10000019*case+1000003*block)
            lo = np.searchsorted(pair_ids, ids[start], side='left')
            hi = np.searchsorted(pair_ids, ids[stop-1], side='right')
            for li, tensor in enumerate(generated):
                values = tensor.cpu().numpy()
                fitted, _ = predict_scene_draw_responses(response, pairs.iloc[lo:hi], ids[start:stop], values)
                delta = frames[li].iloc[start:stop, :2].to_numpy()-context[start:stop, :2]
                shapes = (values[..., :2], values[..., :2]+old[start:stop, None, None]*delta[:, None, :],
                          values[..., :2]+fitted[..., None]*delta[:, None, :])
                base_bad = np.square(shapes[0]).sum(-1) >= 1
                keep = output_cuts(values)
                for ri, e in enumerate(shapes):
                    bad = np.square(e).sum(-1) >= 1
                    for ki in range(3):
                        mask = keep[..., ki]
                        selected_bad = bad & mask
                        sums[li, ki, ri] += [mask.sum(), selected_bad.sum(), (selected_bad & ~base_bad).sum(),
                                             e[..., 0][selected_bad].sum()]
            if block == 0 or (block+1) % 25 == 0:
                print(f'CONDITIONAL_SUPPORT case={case} rows={stop}/{len(ids)}', flush=True)
        reports.append({'case': case, 'sums': sums, 'invalid_fraction': sums[..., 1]/sums[..., 0],
                        'new_invalid_fraction': sums[..., 2]/sums[..., 0]})
    write_json(output, {'cases': reports, 'cuts': CUTS, 'legs': [+.02, -.02],
                        'recipes': ['flow_only', 'original_scalar', 'output_conditioned'],
                        'draws': 64, 'flow_sha256': sha256(flow), 'candidate': response.metadata,
                        'script_sha256': sha256(__file__),
                        'limitations': ['two reused scenes; conditional-flow draws without classifier weighting',
                                        'sampling diagnostic, not an analytic bound or calibration acceptance',
                                        'no shape clipping or renormalization applied']})
    print(f'CONDITIONAL_SUPPORT_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
