"""Delete only verified intermediate image directories for one completed chunk.

The script refuses partial, linked, out-of-boundary, or unextracted targets.  Shape,
detection, input, generated, manifest, and extracted truth catalogues are retained.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_bases(items: list[str]) -> dict[str, Path]:
    bases: dict[str, Path] = {}
    for item in items:
        mode, sep, value = item.partition("=")
        if not sep or not mode or not value or mode in bases:
            raise ValueError(f"invalid or duplicate --base {item!r}")
        bases[mode] = Path(value)
    if set(bases) != {"total", "self", "deployed", "other"}:
        raise ValueError(f"unexpected modes {sorted(bases)}")
    return bases


def directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise RuntimeError(f"refusing symlink inside cleanup target: {item}")
        if item.is_file():
            total += item.stat().st_size
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="append", required=True, help="MODE=PATH")
    ap.add_argument("--truth", required=True,
                    help="completed extracted truth feather that authorizes this chunk")
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--signs", type=float, nargs="+", default=[0.02, -0.02])
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args()
    if os.path.exists(args.receipt):
        raise FileExistsError(f"refusing existing cleanup receipt {args.receipt}")
    if len(set(args.cases)) != len(args.cases) or len(set(args.signs)) != len(args.signs):
        raise ValueError("duplicate cases or signs")

    truth_path = Path(args.truth)
    truth_sidecar = truth_path.with_suffix(".json")
    if not truth_path.is_file() or not truth_sidecar.is_file():
        raise FileNotFoundError("verified truth feather and sidecar are both required")
    truth = pd.read_feather(truth_path, columns=["case", "input_index"])
    with open(truth_sidecar, encoding="utf-8") as handle:
        meta = json.load(handle)
    expected_cases = sorted(args.cases)
    if sorted(truth["case"].unique().astype(int).tolist()) != expected_cases:
        raise RuntimeError("truth catalogue does not contain exactly the cleanup cases")
    if sorted(int(case) for case in meta["cases"]) != expected_cases:
        raise RuntimeError("truth sidecar does not contain exactly the cleanup cases")
    if len(truth) != int(meta["n_rows"]):
        raise RuntimeError("truth frame/sidecar row count differs")
    if truth.duplicated(["case", "input_index"]).any():
        raise RuntimeError("truth catalogue has duplicate keys")

    bases = parse_bases(args.base)
    targets: list[Path] = []
    bytes_by_target: dict[str, int] = {}
    for mode, base in bases.items():
        if base.is_symlink():
            raise RuntimeError(f"refusing linked simulation base {base}")
        base_resolved = base.resolve(strict=True)
        if not base_resolved.is_dir():
            raise RuntimeError(f"invalid simulation base {base_resolved}")
        for case in expected_cases:
            for sign in args.signs:
                label = str(float(sign))
                target = base_resolved / f"case{case}_{label}" / "real0" / "images"
                if target.is_symlink() or not target.is_dir():
                    raise RuntimeError(f"missing, linked, or non-directory cleanup target {target}")
                resolved = target.resolve(strict=True)
                relative = resolved.relative_to(base_resolved)
                if relative.parts != (f"case{case}_{label}", "real0", "images"):
                    raise RuntimeError(f"unexpected cleanup target structure {resolved}")
                if resolved in targets:
                    raise RuntimeError(f"duplicate cleanup target {resolved}")
                targets.append(resolved)
                bytes_by_target[str(resolved)] = directory_bytes(resolved)
    expected_count = len(bases) * len(expected_cases) * len(args.signs)
    if len(targets) != expected_count:
        raise RuntimeError(f"resolved {len(targets)} targets, expected {expected_count}")

    total_bytes = sum(bytes_by_target.values())
    print(
        f"validated {len(targets)} exact image directories; "
        f"deleting {total_bytes / 2**30:.2f} GiB", flush=True,
    )
    for target in targets:
        print(f"DELETE {target}", flush=True)
    for target in targets:
        shutil.rmtree(target)
        if target.exists():
            raise RuntimeError(f"cleanup target still exists after deletion: {target}")

    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.receipt, "x", encoding="utf-8") as handle:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "truth": str(truth_path.resolve()), "cases": expected_cases,
            "signs": args.signs, "target_count": len(targets),
            "deleted_bytes": total_bytes, "deleted_paths": [str(p) for p in targets],
            "retained": [
                "generated catalogues and simulation configs", "input catalogues",
                "SExtractor catalogues", "Shapes catalogues", "truth extraction",
                "manifest",
            ],
            "recoverability": "image directories are not recoverable except by rerendering",
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote cleanup receipt {args.receipt}")
    print("V22_CONSTGOLD_Q3_IMAGE_CLEANUP_DONE", flush=True)


if __name__ == "__main__":
    main()
