"""Restore pre-extension V3.9 features and original prior weights, with provenance."""
import os
from pathlib import Path
import sys

ROOT = Path(os.environ['SBSI_STUDY_ROOT'])
sys.path.insert(0, str(ROOT))
import run_v39_sampling_study as study  # noqa: E402


def main():
    execution = study.identity()
    study.verify(__file__, execution['reduced_prior_builder_sha256'])
    protocol = study.read(ROOT/'protocol.json')
    source = Path(protocol['reduced_prior_source']['root'])
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    assert 0 <= index < protocol['prior_shards'] == 20
    original = source/f'features/shard_{index:02d}'
    receipt = study.read(original/'manifest.json')
    old = Path(receipt['source_record']['root'])
    for name in ('flow_zero.parquet', 'prior_weight.npy'):
        study.verify(original/name, receipt['output_sha256'][name])
    weights_path = old/'prior_weight.npy'
    study.verify(weights_path, receipt['source_record']['output_sha256']['prior_weight.npy'])
    out = ROOT/f'features/shard_{index:02d}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'flow_zero.parquet').symlink_to(original/'flow_zero.parquet')
    # Restore the pre-extension inverse-inclusion weights, not renormalized
    # extended-prior weights: the extension had changed the bright-stratum mass.
    weights = study.np.load(weights_path)
    if protocol['reduced_prior_source']['mode'] == 'uniform20m':
        study.verify(old/'sample_rank.npy', receipt['source_record']['output_sha256']['sample_rank.npy'])
        ranks = study.np.load(old/'sample_rank.npy')
        weights = (ranks < 20000000).astype(float)/20000000
    study.np.save(out/'prior_weight.npy', weights)
    # Keep aligned zero-mass cache rows only if restoring the uniform base;
    # they are never drawn and are outside proposal active support.
    receipt.update(protocol_sha256=study.file_hash(ROOT/'protocol.json'),
        prior_mass=float(weights.sum()), active_rows=int((weights > 0).sum()),
        reduced_prior_source=dict(root=str(original), manifest_sha256=study.file_hash(original/'manifest.json'),
                                  restored_weights_sha256=study.file_hash(weights_path)),
        output_sha256={name: study.file_hash(out/name) for name in ('flow_zero.parquet', 'prior_weight.npy')})
    # Reuse only unsheared proposal coordinates; fresh p(U) and normalization
    # will be evaluated after the new mock fixes its own starting centre.
    prepared = ROOT/f'coordinate_source/shard_{index:02d}'
    prepared.mkdir(parents=True, exist_ok=False)
    source_record = study.read(source/f'prepared/shard_{index:02d}/manifest.json')
    receipt['coordinate_sha256'] = {}
    for name in ('values.npy', 'dispersion.npy'):
        p = source/f'prepared/shard_{index:02d}'/name
        study.verify(p, source_record['output_sha256'][name])
        (prepared/name).symlink_to(p)
        receipt['coordinate_sha256'][name] = source_record['output_sha256'][name]
    study.write_json(out/'manifest.json', receipt)
    print(f'PRIOR_SHARD_COMPLETE shard={index} stored={len(weights)} active={(weights>0).sum()} mass={weights.sum():.12g}', flush=True)


if __name__ == '__main__':
    main()
