"""Independent saved-array phase-estimator audit; CPU, no fits/model forwards."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    summary = json.loads(args.summary.read_text())
    coefficient_path = Path(summary['coefficient_path'])
    assert digest(coefficient_path) == summary['coefficient_sha256']
    frozen = json.loads(coefficient_path.read_text())
    coefficients = np.asarray(frozen['coefficients'])
    pilots = set(frozen['metadata']['pilot_seeds'])
    inputs, evidence, groups, all_seeds = {}, [], {}, set()
    native_hashes, source_prior_hashes = [], []
    for group, group_receipt in summary['groups'].items():
        report_path = Path(group_receipt['report'])
        assert digest(report_path) == group_receipt['report_sha256']
        report = json.loads(report_path.read_text())
        assert not pilots.intersection(report['seeds'])
        assert not all_seeds.intersection(report['seeds'])
        all_seeds.update(report['seeds'])
        arrays, logz, counts = [], [], []
        for path, expected_hash in report['sources'].items():
            path = Path(path)
            assert digest(path) == expected_hash
            inputs[str(path)] = expected_hash
            d = json.loads(path.read_text())
            assert d['stride'] == 1 and d['numerical_checks_pass'] is True
            for original, value in d['source_hashes'].items():
                assert digest(original) == value
            protocol = json.loads((Path(d['run'])/'protocol.json').read_text())
            assert d['retained_particles'] == protocol['particles']
            recorded = {Path(k).name:v for k,v in protocol['implementation_sha256'].items()}
            native_hashes.append({k:recorded[k] for k in ('galsbi_particle_target.py',
                'galsbi_target_coordinates.py', 'galsbi_summary.py', 'summary_gradients.py')})
            target_path = next(Path(k) for k in protocol['implementation_sha256']
                               if Path(k).name == 'galsbi_particle_target.py')
            # These historical snapshots contain the prior sources even when
            # their protocols did not declare the hashes. Record that distinction.
            source_prior_hashes.append({name:digest(target_path.parent/name)
                for name in ('galsbi_prior.py', 'galsbi_scene_prior.py')})
            npz = path.with_suffix('.npz')
            assert digest(npz) == d['arrays_sha256']
            with np.load(npz, allow_pickle=False) as a:
                raw, control, delta = a['raw_scores'].copy(), a['phase_control'].copy(), a['delta_log_kernel'].copy()
                np.testing.assert_array_equal(a['sampled_particle_indices'], np.arange(protocol['particles']))
                np.testing.assert_allclose(raw, a['saved_raw_scores'], atol=1e-6, rtol=0)
                corrected = raw-coefficients[:, None, :]*control[:, :, None]
                if 'corrected_scores' in a:
                    np.testing.assert_array_equal(corrected, a['corrected_scores'])
            arrays.append((np.concatenate((control[..., None], raw, corrected), axis=2), delta))
            counts.append(protocol['particles'])
            logz.append(d['log_normalizer'])
        logz = np.asarray(logz)
        groups[group] = {}
        for oi, original_row in enumerate(report['observations']):
            row = {}
            for use64, key in ((False, 'raw_float32_sampled'), (True, 'precision_reweighted_float64')):
                # Direct unnormalized sum over every particle. No production
                # aggregation helper or per-run normalized moments are reused.
                masses, numerators = [], []
                for ri, (values, delta) in enumerate(arrays):
                    w = np.exp(logz[ri, oi]-logz[:, oi].max())/counts[ri]
                    weights = np.full(counts[ri], w)
                    if use64:
                        weights *= np.exp(delta[oi])
                    masses.append(weights.sum())
                    numerators.append(weights@values[oi])
                masses, numerators = np.asarray(masses), np.asarray(numerators)
                mean = numerators.sum(0)/masses.sum()
                influence = (numerators-masses[:, None]*mean)/masses.mean()
                covariance = influence.T@influence/(len(masses)*(len(masses)-1))
                reference = original_row[key]
                mean_error = float(np.max(np.abs(mean-reference['mean'])))
                covariance_error = float(np.max(np.abs(covariance-reference['covariance'])))
                assert mean_error < 1e-10 and covariance_error < 1e-10
                row[key] = dict(mean=mean.tolist(), mcse=np.sqrt(np.diag(covariance)).tolist(),
                               mean_error=mean_error, covariance_error=covariance_error)
            groups[group][str(original_row['observation'])] = row
        evidence.append(dict(group=group, runs=len(arrays), original_particles=counts,
            all_particles_retained=True, seeds=report['seeds']))
    assert all(h == native_hashes[0] for h in native_hashes)
    assert all(h == source_prior_hashes[0] for h in source_prior_hashes)
    model_root = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
    specs = [('flow', 'v36_f8_20260924_v1/joint/joint_probability_refinement_full/selected.pt',
              'condition_preprocessor'),
             ('classifier', 'v36_classifier_f8_inputs_20260925_v1/C8_20260913/selected.pt', 'preprocessor')]
    models = {}
    for name, relative, preprocessor in specs:
        path = model_root/relative
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        p, config = checkpoint[preprocessor], checkpoint['model_config']
        log_features = list(p.get('log_features', []))
        assert not {'e1_input_p', 'e2_input_p'}.intersection(log_features)
        assert config.get('activation', 'silu') in ('silu', 'tanh', 'gelu')
        models[name] = dict(path=str(path), sha256=digest(path), activation=config.get('activation', 'silu'),
            log_features=log_features, feature_names=p['feature_names'],
            add_missing_indicators=p['add_missing_indicators'], flow_type=config.get('flow_type'),
            dropout=config.get('dropout', 0.))
    gap = {}
    for key in ('raw_float32_sampled', 'precision_reweighted_float64'):
        a, b = groups['v7']['0'][key], groups['v6v100']['0'][key]
        diff = a['mean'][3]-b['mean'][3]
        se = float(np.hypot(a['mcse'][3], b['mcse'][3]))
        gap[key] = dict(corrected_g1_difference=diff, combined_reported_mcse=se,
                        descriptive_mcse_ratio=diff/se)
    result = dict(status='passed', summary=str(args.summary), summary_sha256=digest(args.summary),
        coefficient_sha256=digest(coefficient_path), coefficients=coefficients.tolist(),
        pilot_seeds=sorted(pilots), populations=evidence, groups=groups, observation0_gap=gap,
        common_recorded_target_source_hashes=native_hashes[0],
        common_snapshot_prior_source_hashes=source_prior_hashes[0], model_phase_smoothness=models,
        diagnostic_json_sha256=inputs, audit_script_sha256=digest(__file__),
        limitations=['No fit, GPU, new inference, or heldout outcomes.',
            'Checkpoint metadata inspection only; no new model forwards.',
            'Derivative/replay evidence remains the preserved v2 experiment.',
            'Snapshot prior hashes are independently inspected, not retroactively declared by old protocols.',
            'MCSE ratio is descriptive with eight populations per group, not a calibrated Gaussian significance.'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('PHASE_AUDIT_PASS', json.dumps(dict(populations=evidence, observation0_gap=gap, models=models)), flush=True)


if __name__ == '__main__':
    main()
