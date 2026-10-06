#!/usr/bin/env python
"""Freeze an isolated F8 training checkout and its pre-fit experiment contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path, old, new):
    value = path.read_text()
    if value.count(old) != 1:
        raise ValueError(f'expected one exact patch site in {path}: {old[:80]}')
    path.write_text(value.replace(old, new))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    if a.checkout.exists() or a.root.exists():
        raise FileExistsError('experiment paths must be new')
    a.checkout.mkdir(parents=True)
    a.root.mkdir(parents=True)
    sources = {}
    for folder in ('sbsi', 'scripts', 'plots', 'tests'):
        for source in sorted((a.source / folder).rglob('*.py')):
            relative = source.relative_to(a.source)
            target = a.checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources[str(relative)] = digest(source)
    for name in ('AGENTS.md', 'pyproject.toml'):
        shutil.copy2(a.source / name, a.checkout / name)
        sources[name] = digest(a.source / name)
    # Preserve the full parent/classifier contract. Only training FLOW_FEATURES
    # is reduced. The evaluation scenes still contain all eleven inputs.
    path = a.checkout / 'sbsi/fixed_g0_domain.py'
    replace_once(path, 'FLOW_FEATURES = (', 'FULL_FLOW_FEATURES = (')
    replace_once(path, 'PARENT_FEATURES = FLOW_FEATURES[:PARENT_CONTEXT_WIDTH]',
                 'PARENT_FEATURES = FULL_FLOW_FEATURES[:PARENT_CONTEXT_WIDTH]\n'
                 'FLOW_FEATURES = (*FULL_FLOW_FEATURES[:5], *CLOSE_CROWDING_FEATURES)')
    replace_once(a.checkout / 'scripts/joint_response_axes.py',
                 'from sbsi.fixed_g0_domain import FLOW_FEATURES',
                 'from sbsi.fixed_g0_domain import FULL_FLOW_FEATURES as FLOW_FEATURES')
    path = a.checkout / 'scripts/evaluate_smooth_classifier_joint.py'
    replace_once(path,
        'from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS, PARENT_CONTEXT_WIDTH, PARENT_FEATURES',
        'from sbsi.fixed_g0_domain import FLOW_FEATURES, FULL_FLOW_FEATURES, FLOW_TARGETS, PARENT_CONTEXT_WIDTH, PARENT_FEATURES')
    replace_once(path, 'bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES)',
                 'bundle.condition_preprocessor.feature_names not in (list(FLOW_FEATURES), list(FULL_FLOW_FEATURES))')
    replace_once(path, 'context_names=np.array(FLOW_FEATURES)', 'context_names=np.array(FULL_FLOW_FEATURES)')
    # Install scripts and plots into a task-local venv as well as sbsi. This
    # avoids PYTHONPATH overrides and changes to the shared py31 installation.
    replace_once(a.checkout / 'pyproject.toml', 'packages = ["sbsi"]',
                 'packages = ["sbsi", "scripts", "plots"]')
    final = {str(p.relative_to(a.checkout)): digest(p) for p in a.checkout.rglob('*') if p.is_file()}
    modified = {k: dict(before=sources[k], after=v) for k, v in final.items() if sources[k] != v}
    experiment = dict(name='F8 close-crowding feature removal', checkout=str(a.checkout),
        source_checkout=str(a.source), source_hashes=sources, checkout_hashes=final, modifications=modified,
        raw_features=['e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p',
            'circularized_Re_input_p', 'nbr_close_050', 'nbr_close_100', 'nbr_close_200'],
        removed_features=['nbr_flux_near', 'nbr_flux_far', 'nbr_flux_max'],
        training=dict(seed=611, split_seed=501, cases=list(range(200)),
            recipe='original NLL/guard, harmonized control-variate and common-descent chain',
            initialization='fresh training; no weight slicing or distillation',
            warm_start=False, constant_shear_used_for_training_or_checkpoint_selection=False),
        fixed_components=dict(flow_control_sha256='1433b7217fb77f41f42cbacd97aed317d8db32d74ffe19a17927c56f194ae107',
            classifier_sha256='13cd6a81ecff1ed98596ec770bcfaf99505c139b6dbfe76cef164df9028a52d0',
            disk_emulator_sha256='194c1d11b67d6fd52039060270663e8e1cd610b478b9e9a66e34271701761fe7'),
        development_evaluation=dict(cases=list(range(40, 120)), h=.02, axis=1, draws=64,
            latent_seed=7301, parent='complete generated target role; no truth-property cuts',
            radius_min_pixels=3., magnitude_max=25.8, per_leg_selection=True,
            density_degradation_limit_nats_per_row=.001,
            paired_bias_change_equivalence_margin_pp=.05,
            bootstrap_seed=20260924, bootstrap_replicates=10000,
            tests='own-normalized selected response, selected mass, joint predictive distributions and exact regional contributions'),
        limitations=['Initial one-training-seed development screen, not independent acceptance.',
            'Existing control checkpoint; reduced context changes parameter initialization layout.',
            'Both axes, draw/training repeats, full-role transfer and new-scene acceptance required before promotion.',
            'No automatic promotion even when development gates pass.'],
        maximum_concurrent_sbsi_gpus=2, production_accepted=False)
    (a.root / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    print(json.dumps(dict(checkout=str(a.checkout), files=len(final), changed=list(modified)), indent=2))


if __name__ == '__main__':
    main()
