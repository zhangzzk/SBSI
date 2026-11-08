#!/usr/bin/env python
"""Freeze the previous 100k image inference with symmetric F8 and fresh selection Z.

Copies hash-verified historical source; patches only model binding, provenance,
and orchestration. The retained V3.9 preset and prior/sampler math stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import textwrap


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip())


def save(path, data):
    write(path, json.dumps(data, indent=2) + '\n')


def patch(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f'expected one patch site: {path}: {old[:80]}')
    path.write_text(text.replace(old, new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--selection-reference', type=Path, required=True)
    parser.add_argument('--flow', type=Path, required=True)
    parser.add_argument('--flow-sha256', required=True)
    args = parser.parse_args()
    root, old, pair = [p.resolve() for p in (args.root, args.reference, args.selection_reference)]
    repo = Path(__file__).resolve().parents[1]
    if root.exists():
        raise FileExistsError('preserve existing study')
    if digest(args.flow) != args.flow_sha256:
        raise ValueError('candidate hash mismatch')
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(parents=True)
    original_hashes = {}
    for line in (old / 'source_snapshot/SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.removeprefix('./')
        source = old / 'source_snapshot' / name
        if digest(source) != expected:
            raise ValueError(f'historical source changed: {source}')
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        original_hashes[name] = expected
    overlays = ['sbsi/measurement_model.py', 'sbsi/flow_disk.py', 'sbsi/flow_physical_disk.py',
                'sbsi/disk_precision.py', 'sbsi/flow_sign_symmetric.py', 'tests/test_flow_sign_symmetric.py']
    for name in overlays:
        shutil.copy2(repo / name, snapshot / name)
    fixed = json.loads((snapshot / 'configs/models_v3_9_like.json').read_text())
    identity = dict(name='symmetric-F8-base-guard-with-retained-C8-and-disk',
        flow=str(args.flow.resolve()), flow_sha256=args.flow_sha256,
        classifier_sha256=fixed['classifier']['sha256'], disk_sha256=fixed['emulator']['sha256'],
        disk_metadata_sha256=fixed['emulator']['metadata_sha256'], production_accepted=False)
    save(snapshot / 'configs/signsym_candidate.json', identity)
    write(snapshot / 'scripts/signsym_candidate.py', '''
        import json
        from dataclasses import replace
        from pathlib import Path
        from sbsi.models import V39_LIKE
        IDENTITY = json.loads((Path(__file__).resolve().parents[1] / 'configs/signsym_candidate.json').read_text())
        assert V39_LIKE.detection_classifier_sha256s == (IDENTITY['classifier_sha256'],)
        assert V39_LIKE.emulator_sha256 == IDENTITY['disk_sha256']
        assert V39_LIKE.emulator_metadata_sha256 == IDENTITY['disk_metadata_sha256']
        CANDIDATE = replace(V39_LIKE, name=IDENTITY['name'],
            flow_checkpoints=(Path(IDENTITY['flow']),), flow_sha256s=(IDENTITY['flow_sha256'],))
    ''')
    for name in ('check_galsbi_local_sampler.py', 'run_galsbi_bright.py'):
        path = snapshot / 'scripts' / name
        patch(path, 'from sbsi.models import V39_LIKE, load_detection_classifier, load_disk_response',
              'from sbsi.models import load_detection_classifier, load_disk_response\n'
              'from signsym_candidate import CANDIDATE, IDENTITY')
        path.write_text(path.read_text().replace('V39_LIKE', 'CANDIDATE'))
    sampler = snapshot / 'scripts/check_galsbi_local_sampler.py'
    patch(sampler, 'protocol = dict(steps=args.steps,', 'protocol = dict(model_identity=IDENTITY, steps=args.steps,')
    selector = snapshot / 'scripts/run_galsbi_bright.py'
    patch(selector, "    flow = load_measurement_model(CANDIDATE.flow_checkpoints[0], device='cuda')",
          "    CANDIDATE.validate()\n    flow = load_measurement_model(CANDIDATE.flow_checkpoints[0], device='cuda')")
    patch(selector, "dict(status='completed_selection', replica=replica,",
          "dict(status='completed_selection', model_identity=IDENTITY, replica=replica,")
    # Freeze a new image reference identity but reuse the exact measured rows and truth bookkeeping.
    reference = root / 'reference'
    reference.mkdir()
    old_result = json.loads((old / 'reference/result.json').read_text())
    for name, key in [('observations.parquet', 'observations_sha256'), ('truth.parquet', 'truth_sha256')]:
        if digest(old / 'reference' / name) != old_result[key]:
            raise ValueError(f'image reference changed: {name}')
        (reference / name).symlink_to(old / 'reference' / name)
    protocol = json.loads((old / 'reference/protocol.json').read_text())
    protocol['model_identity'] = identity
    protocol['comparison_reference'] = str(old / 'reference')
    save(reference / 'protocol.json', protocol)
    old_result.update(protocol_sha256=digest(reference / 'protocol.json'),
                      limitation='Same image observations and GalSBI prior; symmetric F8 candidate, fixed C8/disk')
    save(reference / 'result.json', old_result)
    for name in ('bright_ids.txt', 'faint_ids.txt', 'bright21_ids.txt'):
        shutil.copy2(old / name, root / name)
    for stage in ('c1', 'c2'):
        (root / stage).mkdir()
        shutil.copy2(old / stage / 'units.txt', root / stage / 'units.txt')
        (root / stage / 'source_snapshot').symlink_to(snapshot, target_is_directory=True)
    shutil.copy2(old / 'c1/unit_size.json', root / 'c1/unit_size.json')
    selection_root = root / 'normalization'
    for side in ('original', 'twin'):
        dest = selection_root / side / 'it0'
        dest.mkdir(parents=True)
        plan = json.loads((pair / side / 'it0/plan.json').read_text())
        plan.update(study='Symmetric F8 selection normalization; frozen previous nodes, cuts, draws and seeds',
                    model_identity=identity, GPU_limit=4,
                    resources='Two selection GPUs alongside two inference GPUs; four total',
                    chains='Separate unchanged 100k inference recipe')
        save(dest / 'plan.json', plan)
        (dest / 'reference').symlink_to(reference, target_is_directory=True)
        (selection_root / side / 'reference').symlink_to(reference, target_is_directory=True)
    # The original estimator formulas and bootstrap seeds remain unchanged.
    estimate = root / 'estimate.py'
    shutil.copy2(old / 'estimate.py', estimate)
    patch(estimate, str(old), str(root))
    patch(estimate, str(pair), str(selection_root))
    patch(estimate, 'D = .002',
          "IDENTITY = json.loads((H/'source_snapshot/configs/signsym_candidate.json').read_text())\nD = .002")
    patch(estimate, "        protocol = json.loads((run/'protocol.json').read_text())",
          "        protocol = json.loads((run/'protocol.json').read_text())\n"
          "        if protocol.get('model_identity') != IDENTITY:\n"
          "            raise ValueError('sampler model identity mismatch')")
    patch(estimate, "        bad = {k: plan.get(k) for k, v in expect.items() if plan.get(k) != v}",
          "        if plan.get('model_identity') != IDENTITY:\n"
          "            raise ValueError('selection model identity mismatch')\n"
          "        bad = {k: plan.get(k) for k, v in expect.items() if plan.get(k) != v}")
    patch(estimate, "            receipt = json.loads((s/'result.json').read_text())",
          "            receipt = json.loads((s/'result.json').read_text())\n"
          "            if (receipt.get('model_identity') != IDENTITY\n"
          "                    or receipt['chunks'] != plan['selection_chunks']\n"
          "                    or receipt['data_sha256'] != hashlib.sha256((s/'chunks.npz').read_bytes()).hexdigest()):\n"
          "                raise ValueError('selection artifact identity or completion mismatch')")
    patch(estimate, 'out = dict(definitions=__doc__, stage=stage,',
          'out = dict(model_identity=IDENTITY, definitions=__doc__, stage=stage,')
    worker = root / 'worker.sh'
    shutil.copy2(old / 'worker.sh', worker)
    worker.write_text(worker.read_text().replace(str(old), str(root)))
    patch(worker, '#SBATCH --time 12:00:00', '#SBATCH --time 24:00:00')
    # Units are copied verbatim; no new size selection or waiting stage.
    text = worker.read_text()
    start = text.index('if [ $stage = c1 ]')
    end = text.index('[ -s $R/units.txt ] ||', start)
    text = text[:start] + text[end:]
    worker.write_text(text)
    patch(worker, 'hostname; nvidia-smi -L',
          'hostname; nvidia-smi -L\n$PY "$H/detect_resources.py" -o "$H/logs/resources_${SLURM_JOB_ID}.json"\n'
          '$PY "$H/verify.py"')
    # A failure must prevent dependent estimates; allow exactly one identical retry.
    patch(worker, '        else\n            date +%s > $R/claims/$label/failed\n        fi',
          '''        else
            mv "$R/logs/$label.out" "$R/logs/$label.attempt1.out"
            mv "$R/$label" "$R/${label}_attempt1"
            if SBSI_TEMPER_ROOT=$R taskset -c $set bash $T $label --reference ${REF:-$H/reference} --shear $g1 $g2 \\
                    --seed $((seed0+j)) --initial-bank-seed $((bank0+j)) --observation-indices $ids "${opts[@]}" \\
                    < /dev/null > $R/logs/$label.out 2>&1; then
                date +%s > $R/claims/$label/done
            else
                date +%s > $R/claims/$label/failed
                return 1
            fi
        fi''')
    # Study launch scripts are versioned templates with only root substitution.
    for source, target in [('jobs/job_sign_symmetric_inference.sh', 'pipeline.sh'),
                           ('scripts/check_sign_symmetric_inference.py', 'check.py'),
                           ('scripts/verify_sign_symmetric_inference.py', 'verify.py')]:
        write(root / target, (repo / source).read_text().replace('@STUDY_ROOT@', str(root)))
    shutil.copy2(Path('/home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py'),
                 root / 'detect_resources.py')
    shutil.copy2(repo / 'doc/SYMMETRIC_FLOW_INFERENCE.md', root / 'rule.md')
    (root / 'logs').mkdir()
    snapshot_hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*'))
                       if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
    write(snapshot / 'SHA256SUMS', ''.join(f'{v}  ./{k}\n' for k, v in snapshot_hashes.items()))
    save(root / 'source_changes.json', {k: dict(original=original_hashes.get(k), current=v)
         for k, v in snapshot_hashes.items() if original_hashes.get(k) != v})
    # Freeze all inputs, not generated results/logs/claims; verify before every stage.
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
             if p.is_file() and 'source_snapshot' not in p.relative_to(root).parts}
    save(root / 'frozen_inputs.json', dict(files=files, snapshot=snapshot_hashes,
        reference=str(old), model_identity=identity, maximum_concurrent_gpus=4))
    print(json.dumps(dict(root=str(root), files=len(files), source_files=len(snapshot_hashes))))


if __name__ == '__main__':
    main()
