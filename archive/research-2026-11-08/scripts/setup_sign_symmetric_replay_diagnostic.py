"""Capture failing invariant-cache states without changing the production sampler."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    parent, root = args.parent.resolve(), args.root.resolve()
    root.mkdir(exist_ok=False)
    (root/'logs').mkdir()
    snapshot = root/'source_snapshot'
    shutil.copytree((parent/'source_snapshot').resolve(), snapshot, symlinks=True)
    subprocess.run(['sha256sum', '--check', 'SHA256SUMS', '--quiet'], cwd=snapshot, check=True)
    script = snapshot/'scripts/check_galsbi_local_sampler.py'
    original_hash = digest(script)
    text = script.read_text()
    marker = "                    raise ValueError('invariant cache disagrees with full likelihood replay')"
    assert text.count(marker) == 1
    instrumentation = '''                    valid_rows = np.flatnonzero(valid)
                    errors = np.abs(audit_like[valid]-loglike[valid])
                    worst = int(valid_rows[np.argmax(errors)])
                    save(args.output/'failure_context.json', dict(
                        step=step+1, block=int(block), worst_row=worst,
                        reference_observation=int(indices[(worst % ncold)//nchains]),
                        chain=int(worst % nchains), level=int(worst//ncold),
                        cached_log_kernel=float(loglike[worst]),
                        recomputed_log_kernel=float(audit_like[worst]),
                        largest_current_error=float(np.max(errors)),
                        purpose='diagnostic only; original guard remains fatal'))
                    np.savez_compressed(args.output/'failure_state.npz',
                        coordinates=x, proposal=proposal, valid=valid, auxiliary=aux,
                        observations=observed.to_numpy(dtype=float),
                        observation_columns=observed.columns.to_numpy(dtype=str),
                        reference_indices=np.asarray(indices),
                        cached_log_kernel=loglike, recomputed_log_kernel=audit_like,
                        cached_summary=summaries, recomputed_summary=audit_summary,
                        log_prior=logprior, proposed_log_prior=newprior)
'''
    script.write_text(text.replace(marker, instrumentation+marker))
    hashes = snapshot/'SHA256SUMS'
    lines = hashes.read_text().splitlines()
    rewritten = []
    for line in lines:
        sha, name = line.split(maxsplit=1)
        rewritten.append((digest(script)+'  '+name) if name.lstrip('./') ==
                         'scripts/check_galsbi_local_sampler.py' else line)
    hashes.write_text('\n'.join(rewritten)+'\n')
    subprocess.run(['sha256sum', '--check', 'SHA256SUMS', '--quiet'], cwd=snapshot, check=True)
    shutil.copy2(parent/'detect_resources.py', root/'detect_resources.py')
    planned = []
    for j, line in enumerate((parent/'c1/units.txt').read_text().splitlines()):
        label, filename, first, last = line.split()
        if not (parent/'c1/claims'/label/'failed').exists():
            continue
        previous = json.loads((parent/'c1'/label/'protocol.json').read_text())
        ids = (parent/filename).read_text().splitlines()[int(first):int(last)+1]
        assert list(map(int, ids)) == previous['observations']
        assert previous['seed'] == 2028500001+j
        command = ['bash', str(snapshot/'jobs/job_galsbi_tempered.sh'), label,
                   '--reference', str(parent/'reference'), '--shear', '0', '0',
                   '--seed', str(previous['seed']), '--initial-bank-seed',
                   str(previous['initial_bank_seed']), '--observation-indices', *ids]
        if label.startswith('bright'):
            command += ['--betas', '1', '.7883', '.6167', '.4449', '.2849', '.1473',
                        '.05014', '.005', '--narrow-step-probability', '.5']
        planned.append(dict(label=label, command=command,
            original_protocol_sha256=digest(parent/'c1'/label/'protocol.json'),
            original_failure=json.loads((parent/'c1'/label/'invariant_replay_failure.json').read_text())))
    rule = dict(parent=str(parent), runs=planned, original_sampler_sha256=original_hash,
        diagnostic_sampler_sha256=digest(script),
        policy='Capture only after original guard fails; no tolerance, seed, batch, or sampler changes. '
               'Never substitute diagnostic output into production. A40 diagnostics; original bright failure used V100.')
    (root/'rule.json').write_text(json.dumps(rule, indent=2)+'\n')
    runner = '''import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
run=json.loads((root/'rule.json').read_text())['runs'][int(sys.argv[1])]
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
result=subprocess.run(run['command'],env=dict(os.environ,SBSI_TEMPER_ROOT=str(root)))
folder=root/run['label']
captured=(folder/'failure_state.npz').exists() and (folder/'failure_context.json').exists()
receipt=dict(returncode=result.returncode,failure_captured=captured,production_eligible=False)
(root/(run['label']+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\\n')
sys.exit(0 if captured or result.returncode==0 else result.returncode)
'''
    (root/'run.py').write_text(runner)
    (root/'job.sh').write_text('''#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --constraint=x86-64-v3
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
''' + f'H={root}\n' + '''PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PY" "$H/detect_resources.py" -o "$H/logs/resources_${SLURM_JOB_ID}.json"
exec "$PY" -u "$H/run.py" "$1"
''')
    print(root)


if __name__ == '__main__':
    main()
