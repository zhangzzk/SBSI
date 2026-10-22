"""Hash/copy only: freeze n22 full-SMC experiment without running inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from benchmark_exchange_full_smc import BASE_HASH, OPT_HASH, SEEDS, OBSERVATIONS, digest, load, save, manifest, verify


def prepare(root, destination, candidate):
    start=time.monotonic()
    worktree=Path(__file__).resolve().parents[1]
    dev=load(worktree/'doc/evidence/experiment_v10_corrected_three_way.json')
    original=Path(dev['source'])
    if digest(original/'SHA256SUMS') != dev['source_manifest_sha256']:
        raise ValueError('corrected development source changed')
    entries=manifest(original)
    if entries['sbsi/scene_exchange.py'] != BASE_HASH or digest(candidate) != OPT_HASH:
        raise ValueError('known corrected control and validated categorical candidate required')
    destination.mkdir(parents=True,exist_ok=False)
    arms={}
    for arm in ('baseline','optimized'):
        target=destination/arm; target.mkdir()
        current=dict(entries)
        for name in entries:
            path=target/name; path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original/name,path)
        if arm=='optimized':
            shutil.copyfile(candidate,target/'sbsi/scene_exchange.py')
            current['sbsi/scene_exchange.py']=OPT_HASH
        (target/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(current.items())))
        arms[arm]=dict(source=str(target.resolve()),manifest_sha256=digest(target/'SHA256SUMS'))
    protocol_path=root/'dev_v10_small_0/protocol.json'
    protocol=load(protocol_path)  # Scientific inputs only; no posterior outcomes.
    reference=Path(protocol['reference'])
    external=[protocol_path,original/'SHA256SUMS',
        worktree/'doc/evidence/experiment_v10_corrected_three_way.json',
        *(reference/name for name in ('protocol.json','result.json','observations.parquet'))]
    for key in ('initial_proposal','physical_geometry'):
        path=Path(dev[key]['path'])
        if digest(path)!=dev[key]['sha256']: raise ValueError('changed frozen pilot artifact')
        external.append(path)
    config=load(original/'configs/models_v3_9_like.json')
    roots=dict(SBSI_CACHE_DIR=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches'),
        BLENDEMU_RUNS_DIR=Path('/project/ls-gruen/users/zekang.zhang/blendemu_runs'))
    for key in ('flow','classifier','emulator'):
        item=config[key]; path=roots[item['root']]/item['path']
        if digest(path)!=item['sha256']: raise ValueError('changed model artifact')
        external.append(path)
        if 'metadata_path' in item:
            path=roots[item['root']]/item['metadata_path']
            if digest(path)!=item['metadata_sha256']: raise ValueError('changed model metadata')
            external.append(path)
    external.extend([Path('/home/z/Zekang.Zhang/_cache/cosmo-torrent/Fischbacher+24_abc_posterior/abc_posterior.h5'),
        Path('/home/z/Zekang.Zhang/_cache/cosmo-torrent/HSC_tables/HSC_template_integrals_yfix.h5')])
    installed=Path('/project/ls-gruen/users/zekang.zhang/envs/py31/lib/python3.10/site-packages')
    for package in ('galsbi','PyCosmo'):
        external.extend(p for p in (installed/package).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo','.log','.lock'))
    harness_files={}
    for name in ('benchmark_exchange_full_smc.py','run_exchange_full_smc.sh','prepare_exchange_full_smc.py'):
        target=destination/name; shutil.copyfile(worktree/'scripts'/name,target)
        harness_files[str(target.resolve())]=digest(target)
    d=dict(schema='sbsi-n22-full-smc-paired-v1',arms=arms,seeds=SEEDS,observations=OBSERVATIONS,
        reference=str(reference),initial_proposal=dev['initial_proposal']['path'],
        physical_geometry=dev['physical_geometry']['path'],output=str(root/'exchange_full_smc_runs_20261022'),
        external_files={str(p.resolve()):digest(p) for p in sorted(set(external))},harness_files=harness_files,
        settings=dict(particles=256,stages=64,moves=64,power=2.,scale=1.,shear=[0.,0.],density_batch_size=4096),
        expected_runtime=dict(node='th-cl-nv02',gpu='Tesla V100-SXM2-32GB',threads=2),
        allocation_dependencies='afterany:17435348:17435350',array='0-3%1',
        hypothesis='Identical saved whole-run scientific outputs with lower full-process cost; no efficiency conclusion in advance.',
        exclusions=dict(result=['timing_seconds.initialization','timing_seconds.sampling','timing_seconds.total',
            'final_stage.elapsed_seconds','validated protocol_sha256/particles_sha256/scenes_sha256'],
            progress=['stages[*].elapsed_seconds'],protocol=['runtime.slurm_job_id',
            'validated source-root relocation and known scene_exchange.py hash only']),
        source_preparation_seconds=time.monotonic()-start,
        preparation_cost_scope='File-copy/hash preparation through declaration assembly; charged equally. Final tiny JSON/manifest writes excluded explicitly.',
        limits=['No heldout outcomes or current heldout scientific files read or changed.',
            'Pinned prior packages/HDF inputs and model artifacts; not a whole environment image.',
            'No extra warmup, phase-control evaluation, posterior fit or adaptation.'])
    save(destination/'declaration.json',d)
    verify(d)
    top=sorted(p for p in destination.iterdir() if p.is_file())
    (destination/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in top))
    for p in destination.rglob('*'): p.chmod(p.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    print(json.dumps(dict(destination=str(destination),declaration_sha256=digest(destination/'declaration.json'),
        manifest_sha256=digest(destination/'SHA256SUMS'),submitted=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True); a=p.parse_args()
    prepare(a.root,a.destination,a.candidate)
