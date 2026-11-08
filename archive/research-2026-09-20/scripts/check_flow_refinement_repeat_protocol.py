"""Check that the declared flow-refinement repeat changes only random seeds."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1')
    protocol = json.loads((root/'flow_refinement_repeat_protocol.json').read_text())
    for name, expected in protocol['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen repeat input changed: '+name)
    original = Path('scripts/refine_joint_guard_probability_flow.py').read_text()
    for old, new in protocol['seed_changes'].items():
        if old not in original:
            raise ValueError('seed absent from original source')
        original = original.replace(old, str(new))
    if original != Path('scripts/refine_joint_guard_probability_flow_repeat.py').read_text():
        raise ValueError('repeat changes more than the declared random streams')
    print('FLOW_REFINEMENT_REPEAT_PROTOCOL_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
