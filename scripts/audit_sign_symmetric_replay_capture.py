"""Test which captured summary change explains a failed invariant replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic_root', type=Path)
    parser.add_argument('label')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU audit requires the scheduler')
    root = args.diagnostic_root.resolve()
    sys.path[:0] = [str(root/'source_snapshot'), str(root/'source_snapshot/scripts')]
    import numpy as np
    import pandas as pd
    import torch
    from signsym_candidate import CANDIDATE, IDENTITY
    from sbsi.galsbi_summary import SummaryBatch, SummaryLikelihood
    from sbsi.measurement_model import load_measurement_model
    from sbsi.models import load_detection_classifier
    from sbsi.v39_inference import FEATURES, V39CatalogueCache

    torch.set_num_threads(1)
    CANDIDATE.validate()
    flow = load_measurement_model(CANDIDATE.flow_checkpoints[0], device='cuda')
    detector = load_detection_classifier(CANDIDATE, device='cuda')
    likelihood = SummaryLikelihood(flow, detector)
    folder = root/args.label
    capture = folder/'failure_state.npz'
    context = json.loads((folder/'failure_context.json').read_text())
    row = context['worst_row']
    with np.load(capture, allow_pickle=False) as z:
        old, new = z['cached_summary'].copy(), z['recomputed_summary'].copy()
        old[:, 8] = new[:, 8] = 0.  # b is unused at the frozen zero sampling shear.
        observed = pd.DataFrame(z['observations'], columns=z['observation_columns'])
        auxiliary = z['auxiliary'].copy()
        recorded_old, recorded_new = z['cached_log_kernel'].copy(), z['recomputed_log_kernel'].copy()
        valid = z['valid'].copy()
    def evaluate(values):
        return likelihood.log_kernel(SummaryBatch(values, auxiliary), observed, 0., 0.)
    baseline, recomputed = evaluate(old), evaluate(new)
    report = dict(model_identity=IDENTITY, context=context,
        capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        feature_names=list(FEATURES), zero_shear_b_set_to_zero=True,
        baseline_replay_max=float(np.max(np.abs(baseline[valid]-recorded_old[valid]))),
        recomputed_replay_max=float(np.max(np.abs(recomputed[valid]-recorded_new[valid]))),
        worst_delta=float(recomputed[row]-baseline[row]),
        rows_above_tolerance=int(np.sum(np.abs(recomputed[valid]-baseline[valid]) > 1e-6)),
        float32_changes=[], interventions=[], production_eligible=False)
    for i, col in np.argwhere((old[:, :8].astype('f4') != new[:, :8].astype('f4')) & valid[:, None]):
        report['float32_changes'].append(dict(row=int(i), feature=FEATURES[col],
            old64=float(old[i, col]), new64=float(new[i, col]),
            old32=float(np.float32(old[i, col])), new32=float(np.float32(new[i, col]))))
    for col in np.flatnonzero(old[row, :8] != new[row, :8]):
        forward, reverse = old.copy(), new.copy()
        forward[row, col], reverse[row, col] = new[row, col], old[row, col]
        report['interventions'].append(dict(feature=FEATURES[col],
            forward_delta=float(evaluate(forward)[row]-baseline[row]),
            reverse_residual=float(evaluate(reverse)[row]-baseline[row])))
    probabilities = []
    for values in (old, new):
        frame = pd.DataFrame(values[:, :8], columns=FEATURES)
        probabilities.append(V39CatalogueCache(frame, np.ones(len(frame)), {}, detector=detector)
                             .get(0., 0.).detection_probability)
    report['worst_classifier_log_delta'] = float(np.log(probabilities[1][row])-np.log(probabilities[0][row]))
    (folder/'capture_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
