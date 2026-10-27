#!/usr/bin/env python
"""Freeze the retained F8 training recipe with a new sign-symmetric flow.

Restores the historical training modules into a study-local source snapshot;
the active library never imports archived experiment code. Input domains are
read-only links to the original uncut F8 projections.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"expected one patch site: {path}: {old[:60]}")
    path.write_text(text.replace(old, new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--invariant-features", choices=("bounded", "asinh"), default="bounded")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root, reference = args.root.resolve(), args.reference.resolve()
    checkout = root / "source_snapshot"
    if checkout.exists() or (root / "experiment.json").exists():
        raise FileExistsError("preserving existing frozen experiment")
    root.mkdir(parents=True, exist_ok=True)
    old = json.loads((reference / "experiment.json").read_text())
    source = Path(old["checkout"])
    for name, expected in old["checkout_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"reference source changed: {name}")
    for name in old["checkout_hashes"]:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    # Current inference-compatible density paths plus the new architecture.
    overlays = ["sbsi/measurement_model.py", "sbsi/flow_disk.py", "sbsi/flow_physical_disk.py",
        "sbsi/disk_precision.py", "sbsi/flow_sign_symmetric.py", "tests/test_flow_sign_symmetric.py"]
    for name in overlays:
        shutil.copy2(repo / name, checkout / name)
    trainer = checkout / "scripts/train_fixed_g0_flow.py"
    replace_once(trainer, '"flow_type": "physical_disk_affine",',
                 '"flow_type": "sign_symmetric_physical_disk_affine",\n'
                 f'        "invariant_features": "{args.invariant_features}",')
    replace_once(trainer, '    physical = torch.as_tensor(train["target_raw"], dtype=torch.float64)',
        '''    # Fit shape coordinates to the conceptual four-reflection orbit.
    # No reflected rows are materialized and no training labels are changed.
    for values, means, scales in (
        (train["context_raw"], condition.means, condition.scales),
        (train["target_raw"], target.means, target.scales),
    ):
        count = len(values)
        scales[:2] = np.hypot(scales[:2] * np.sqrt((count - 1) / count), means[:2])
        means[:2] = 0
    condition.fill_values[:2] = 0

    physical = torch.as_tensor(train["target_raw"], dtype=torch.float64)''')
    replace_once(trainer, '    disk_scales = shape_coordinate.std(0, correction=1)',
        '''    disk_scales = torch.hypot(shape_coordinate.std(0, correction=1)
        * np.sqrt((len(shape_coordinate) - 1) / len(shape_coordinate)), disk_means)
    disk_means = torch.zeros_like(disk_means)''')
    for part in ("base", "harmonized"):
        (root / part).mkdir(exist_ok=True)
        (root / part / "domain").symlink_to(reference / part / "domain", target_is_directory=True)
    for folder in ("joint", "status", "logs"):
        (root / folder).mkdir(exist_ok=True)
    shutil.copy2(reference / "run_v36_f8_stage.py", root / "run_stage.py")
    # Keep the original label regeneration/replay and complete refinement chain.
    # Its evaluate entry point is deliberately not launched by this study.
    fixed = json.loads((repo / "configs/models_v3_9_like.json").read_text())
    experiment = dict(old)
    experiment.update(name="F8 independent shape-sign symmetry; C8 unchanged",
        checkout=str(checkout), source_checkout=str(source), reference_root=str(reference),
        source_hashes=old["checkout_hashes"],
        checkout_hashes={str(p.relative_to(checkout)): digest(p)
            for p in sorted(checkout.rglob("*")) if p.is_file()},
        overlay_hashes={name: digest(repo / name) for name in overlays},
        runner_sha256=digest(root / "run_stage.py"),
        fixed_components=dict(flow_control_sha256=fixed["flow"]["sha256"],
            classifier_sha256=fixed["classifier"]["sha256"],
            disk_emulator_sha256=fixed["emulator"]["sha256"]),
        symmetry=dict(group="Z2 x Z2 independent e1/e2 sign actions",
            flow_type="sign_symmetric_physical_disk_affine",
            invariant_features=args.invariant_features,
            conditioning="first two signed shapes, zero means/fills; invariant conditioner features",
            physical_shape_centres=[0, 0], preprocessing="shape RMS about zero",
            full_joint_symmetry_claim=False),
        maximum_concurrent_sbsi_gpus=1,
        production_accepted=False,
        limitations=["One fresh training seed, no production promotion.",
            "C8 unchanged: symmetry of the full usable-event kernel is not guaranteed.",
            "No ConstGold rows used for training or checkpoint selection.",
            "Retained training truth-domain bounds and guard losses are unchanged."])
    experiment["training"] = dict(old["training"], recipe="retained full base/labels/refinement chain with sign-symmetric architecture")
    (root / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    print(json.dumps(dict(root=str(root), snapshot=str(checkout), files=len(experiment["checkout_hashes"]))))


if __name__ == "__main__":
    main()
