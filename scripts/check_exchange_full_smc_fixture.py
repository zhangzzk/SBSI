"""CPU-only saved-development integration fixture; never runs an inference driver."""
import argparse
import copy
from pathlib import Path
import shutil
import os
import benchmark_exchange_full_smc as m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--declaration',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ: raise ValueError('CPU allocation required')
    a.output.mkdir(parents=True,exist_ok=False)
    d=m.load(a.declaration)
    if m.digest(m.__file__) != d['harness_files'][str(a.declaration.parent/'benchmark_exchange_full_smc.py')]:
        raise ValueError('fixture comparator differs from prepared comparator')
    sources=m.verify(d)
    names=['protocol.json','result.json','progress.json','particles.npz','scenes.npz']
    original_hashes={name:m.digest(a.development/name) for name in names}
    original_protocol=m.load(a.development/'protocol.json')
    original_source=Path(d['protocol_template']['source'])
    changes={}
    for arm in ('baseline','optimized'):
        folder=a.output/arm; folder.mkdir()
        for name in names: shutil.copyfile(a.development/name,folder/name)
        protocol=m.load(folder/'protocol.json')
        rebound={}
        for old,checksum in protocol['implementation_sha256'].items():
            relative=Path(old).relative_to(original_source).as_posix()
            if sources['baseline'][relative] != checksum: raise ValueError('original implementation mismatch')
            rebound[str(Path(d['arms'][arm]['source'])/relative)]=sources[arm][relative]
        protocol['implementation_sha256']=rebound
        m.save(folder/'protocol.json',protocol)
        result=m.load(folder/'result.json'); old_protocol_hash=result['protocol_sha256']
        if old_protocol_hash != original_hashes['protocol.json']: raise ValueError('original protocol hash mismatch')
        result['protocol_sha256']=m.digest(folder/'protocol.json'); m.save(folder/'result.json',result)
        changes[arm]=dict(implementation_before=original_protocol['implementation_sha256'],
            implementation_after=rebound,protocol_sha256_before=old_protocol_hash,
            protocol_sha256_after=result['protocol_sha256'],copied_file_sha256={n:m.digest(folder/n) for n in names})
    # Test-only expected-seed mapping lets unchanged old saved values exercise the
    # real comparators. There is no production flag permitting an alternate seed.
    production_seeds=list(m.SEEDS)
    m.SEEDS=[original_protocol['seed'],*production_seeds[1:]]
    try:
        arrays=m.compare_runs(d,a.output,sources,0)
    finally:
        m.SEEDS=production_seeds
    assert original_hashes=={name:m.digest(a.development/name) for name in names}
    m.save(a.output/'fixture_receipt.json',dict(status='passed',inference_executed=False,
        declaration_sha256=m.digest(a.declaration),original_directory=str(a.development),
        original_sha256=original_hashes,fixture_rebinding=changes,
        expected_seed_override=dict(production=production_seeds[0],fixture=original_protocol['seed'],
            scope='Imported test harness constant only; original and copied protocol seed unchanged'),
        compared_arrays=arrays,complete_stages=64,
        allowed_saved_output_changes=['protocol.implementation_sha256','result.protocol_sha256'],
        implementation_sha256={str(Path(x).resolve()):m.digest(x) for x in [__file__,m.__file__]}))
    print('AUDITED_DEVELOPMENT_FIXTURE_PASS',a.output,flush=True)


if __name__=='__main__': main()
