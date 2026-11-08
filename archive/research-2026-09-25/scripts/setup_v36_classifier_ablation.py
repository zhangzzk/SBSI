"""Freeze matched C17/C12 training and cached-draw evaluation before fitting."""
import hashlib
import json
from pathlib import Path
import shutil

CACHE = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
ROOT = CACHE / 'v36_classifier_occam_20260925_v1'
SOURCE = CACHE.parent / 'sbsi_flow_f8_20260924'
CHECKOUT = CACHE.parent / 'sbsi_classifier_occam_20260925'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'patch must match exactly once: {old}')
    return text.replace(old, new)


def main():
    if ROOT.exists() or CHECKOUT.exists():
        raise FileExistsError('new experiment paths required')
    ROOT.mkdir()
    CHECKOUT.mkdir()
    for folder in ('sbsi', 'scripts', 'plots', 'tests'):
        for source in (SOURCE / folder).rglob('*.py'):
            target = CHECKOUT / source.relative_to(SOURCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ('pyproject.toml', 'AGENTS.md'):
        shutil.copy2(SOURCE / name, CHECKOUT / name)
    (CHECKOUT / 'configs').mkdir()
    for source in Path('configs').glob('*.json'):
        shutil.copy2(source, CHECKOUT / 'configs' / source.name)
    path = CHECKOUT / 'scripts/train_smooth_classifier.py'
    code = path.read_text()
    code = replace(code, 'SEED = 20260913',
        'SEED = 20260913\nFULL_FEATURES = FEATURES\nFEATURE_INDICES = list(range(17))')
    code = replace(code, "stored[f'x{i}'][:n, :dimensions]", "stored[f'x{i}'][:n, FEATURE_INDICES]")
    code = replace(code, 'def main():', 'def main():\n    global SEED, FEATURES, FEATURE_INDICES')
    code = replace(code, "args = parser.parse_args()", """parser.add_argument('--feature-arm', choices=('C17', 'C12'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    SEED = args.seed
    FEATURE_INDICES = list(range(17)) if args.feature_arm == 'C17' else [*range(8), *range(13, 17)]
    FEATURES = tuple(FULL_FEATURES[i] for i in FEATURE_INDICES)""")
    code = replace(code, "manifest['features'] != list(FEATURES)", "manifest['features'] != list(FULL_FEATURES)")
    code = replace(code, "protocol = {'arm': args.arm,", "protocol = {'feature_arm': args.feature_arm, 'feature_indices': FEATURE_INDICES, 'arm': args.arm,")
    # The existing initialization is exactly paired for C17/C12: core eight
    # input weights are identical and all added crowding weights start at zero.
    path.write_text(code)
    for name in ('run_v36_classifier_ablation.py', 'summarize_v36_classifier_ablation.py',
                 'compare_v36_saved_models.py', 'audit_v36_saved_rows.py'):
        shutil.copy2(Path('scripts') / name, CHECKOUT / 'scripts' / name)
    prepared = CACHE / 'scene_classifier_unbounded_20260922_v1/smooth_prepared_full'
    smoke = ROOT / 'smoke_prepared'
    (smoke / 'cases').mkdir(parents=True)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    manifest['cases'] = [40, 100]
    for case in manifest['cases']:
        name = f'case{case:03d}.npz'
        (smoke / 'cases' / name).symlink_to(prepared / 'cases' / name)
    (smoke / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2('tests/test_v36_classifier_ablation.py', CHECKOUT / 'tests/test_v36_classifier_ablation.py')
    old = CACHE / 'scene_classifier_unbounded_20260922_v1/smooth_classifier_full/selected.pt'
    f8 = CACHE / 'v36_f8_20260924_v1'
    fixed = json.loads((f8 / 'trained_candidate.json').read_text())
    pins = {str(p): digest(p) for p in CHECKOUT.rglob('*.py')}
    pins.update({str(p): digest(p) for p in (CHECKOUT/'configs').glob('*.json')})
    pins.update({str(prepared / 'manifest.json'): digest(prepared / 'manifest.json'),
                 str(smoke / 'manifest.json'): digest(smoke / 'manifest.json'),
                 str(old.parent / 'completed.json'): digest(old.parent / 'completed.json'),
                 str(old): digest(old), str(f8 / 'development_comparison.json'): digest(f8 / 'development_comparison.json')})
    protocol = dict(root=str(ROOT), checkout=str(CHECKOUT), prepared=str(prepared),
        f8_root=str(f8), old_classifier=str(old), fixed=fixed, files=pins,
        arms=['C17', 'C12'], seeds=[20260913, 20260914],
        training_cases=list(range(40, 100)), tune_cases=list(range(100, 120)),
        evaluation_cases=list(range(40, 120)), initialization='paired eight-input core; added weights zero',
        training='original 100-epoch BCE recipe; fresh models; unchanged rows, labels, splits and optimizer',
        selection='minimum tune mean-case BCE; no shear checkpoint selection',
        criteria=dict(point_estimates_must_not_worsen=True, paired_ci_level=.95,
            bias_absolute_degradation_margin_pp=.05, bce_degradation_margin=.0005,
            selected_count_fraction_degradation_margin=.0005,
            cancellation_degradation_margin_pp=.05,
            cross_response_degradation_margin=.0005,
            endpoints='main-cut absolute m, cross response and magnitude-bin cancellation; all three cut counts; both-leg BCE/Brier/calibration and half-shear tuning BCE',
            comparisons='each C12 seed against both same-seed C17 and retained classifier',
            required='all gates and both seeds; inconclusive is not accepted'),
        bootstrap=dict(seed=20260925, replicates=10000),
        no_truth_cuts=True, per_leg_measured_cuts='radius>0.60arcsec and MAG_AUTO<25.8',
        development_only=True, production_accepted=False,
        stages_after_c12='C9 and emulator capacity deferred until C12 result is reviewed',
        maximum_concurrent_gpus=2)
    (ROOT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(ROOT)


if __name__ == '__main__':
    main()
