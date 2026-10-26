"""Freeze the direct F8-input classifier experiment, retaining paired C17 controls."""
import hashlib
import json
from pathlib import Path
import shutil

from scripts.setup_v36_classifier_ablation import replace

CACHE = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
PRIOR = CACHE / 'v36_classifier_occam_20260925_v1'
ROOT = CACHE / 'v36_classifier_f8_inputs_20260925_v1'
CHECKOUT = CACHE.parent / 'sbsi_classifier_f8_inputs_20260925'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ROOT.exists() or CHECKOUT.exists():
        raise FileExistsError('new experiment required')
    ROOT.mkdir()
    old = json.loads((PRIOR/'protocol.json').read_text())
    source = Path(old['checkout'])
    for folder in ('sbsi', 'scripts', 'plots', 'tests', 'configs'):
        for path in (source/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json'):
                out = CHECKOUT/path.relative_to(source)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
    for name in ('pyproject.toml', 'AGENTS.md'):
        shutil.copy2(source/name, CHECKOUT/name)
    path = CHECKOUT/'scripts/train_smooth_classifier.py'
    code = path.read_text()
    code = replace(code, 'from scripts.smooth_scene_classifier_features import FEATURES, FORMAT',
                   "from sbsi.fixed_g0_domain import FLOW_FEATURES as FEATURES\nFORMAT = 'scene_classifier_physical_f8_v1'")
    code = code.replace("'state_major_axis'", "'physical_arcsec'")
    code = replace(code, "choices=('C17', 'C12')", "choices=('C8',)")
    code = replace(code, "FEATURE_INDICES = list(range(17)) if args.feature_arm == 'C17' else [*range(8), *range(13, 17)]",
                   'FEATURE_INDICES = list(range(8))')
    code = replace(code, 'FEATURE_INDICES = list(range(17))', 'FEATURE_INDICES = list(range(8))')
    code = replace(code, "if dimensions == 8:\n        return core, config", """if dimensions == 8:
        # Match the C17 target-feature core; new physical crowding starts at zero.
        with torch.no_grad():
            core.net[0].weight[:, 5:].zero_()
        return core, config""")
    code = code.replace('shared eight-input core; additional geometry weights zero',
                        'shared five-target-feature core; physical crowding weights zero')
    path.write_text(code)
    # Corrected numeric-label scorer; the original C12 checkout stays immutable.
    code = Path('scripts/run_v36_classifier_ablation.py').read_text()
    code = replace(code, "return result, dict(pair_sha256=meta['sha256'], geometry=report)", """for frame in result.values():
        for i, name in enumerate(('nbr_close_050', 'nbr_close_100', 'nbr_close_200')):
            frame[name] = context[:, 8+i]
    return result, dict(pair_sha256=meta['sha256'], geometry=report)""")
    (CHECKOUT/'scripts/run_v36_classifier_ablation.py').write_text(code)
    code = Path('scripts/summarize_v36_classifier_ablation.py').read_text()
    code = code.replace("f'C12_{seed}'", "f'C8_{seed}'")
    code = replace(code, 'np.all(point <= 1e-12) and np.all(hi <= margin)',
                   'np.isfinite(point).all() and np.all(lo <= 0) and np.all(hi <= margin)')
    code = code.replace('Review before C9; retain current classifier unless every gate passes.',
                        'Retain C17 unless both C8 seeds pass every uncertainty-tolerant gate.')
    (CHECKOUT/'scripts/summarize_v36_classifier_ablation.py').write_text(code)
    shutil.copy2('scripts/prepare_v36_classifier_f8_inputs.py', CHECKOUT/'scripts/prepare_v36_classifier_f8_inputs.py')
    shutil.copy2('scripts/amend_v36_classifier_acceptance.py', CHECKOUT/'scripts/amend_v36_classifier_acceptance.py')
    for name in ('test_v36_classifier_acceptance_amendment.py', 'test_v36_classifier_f8_inputs.py'):
        shutil.copy2(Path('tests')/name, CHECKOUT/'tests'/name)
    for seed in old['seeds']:
        (ROOT/f'C17_{seed}').symlink_to(PRIOR/f'C17_{seed}', target_is_directory=True)
    parent = CACHE/'constgold_truth_parent_closecrowd_20260924_v1/parent'
    inputs = [Path(old['prepared'])/'manifest.json', parent/'manifest.json',
              Path(old['old_classifier']), Path(old['old_classifier']).parent/'completed.json',
              Path(old['f8_root'])/'development_comparison.json']
    for seed in old['seeds']:
        inputs.extend(PRIOR/f'C17_{seed}'/n for n in ('selected.pt', 'completed.json', 'protocol.json'))
    files = {str(p): digest(p) for p in CHECKOUT.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')}
    files.update({str(p): digest(p) for p in inputs})
    p = {k: v for k, v in old.items() if k not in ('pretraining_packaging_fix', 'files')}
    p.update(root=str(ROOT), checkout=str(CHECKOUT), prepared=str(ROOT/'prepared'),
        source_prepared=old['prepared'], close_parent=str(parent), files=files,
        arms=['C17', 'C8'], reused_controls=str(PRIOR),
        initialization='same C17 first-five target weights and hidden layers; three physical crowding weights zero',
        raw_features=['e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p', 'circularized_Re_input_p',
                      'nbr_close_050', 'nbr_close_100', 'nbr_close_200'],
        feature_change='exact F8 raw inputs: physical arcsec scales .5/1/2, full rendered neighbours within10arcsec',
        acceptance_rule='owner permits small unresolved worsening: every paired95% CI lower<=0 and upper<=original margin',
        stages_after_c12='C8 directly tests shared F8 inputs; C12 rejected, no C9 promotion',
        projection='identity join original half-shear classifier rows to close-crowding parent; preserve all labels and per-leg core5')
    p['criteria'] = dict(old['criteria'], point_estimates_must_not_worsen=False,
                         comparisons='each C8 seed versus same-seed C17 and retained classifier')
    (ROOT/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
    print(ROOT)


if __name__ == '__main__':
    main()
