"""Prepare a paired constant-shear rerender with every input galaxy in emulator support.

The source constant-shear catalogues are already paired across +/-g.  This script keeps their
galaxies, positions, IDs, and ordering fixed and removes only objects outside the deployed
BlendEMU regression box.  It then rewrites the existing MultiBand_ImSim configuration paths into
a new output tree.  It deliberately does not regenerate galaxies from the parent FS2 catalogue:
doing that after filtering the parent would change every sampled scene and destroy the paired
comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _atomic_feather(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_feather(tmp)
    os.replace(tmp, path)


def _replace_ini_value(text: str, section: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"(?ms)(^\[{re.escape(section)}\]\s*.*?)(?=^\[|\Z)"
    )
    match = pattern.search(text)
    if match is None:
        raise KeyError(f"missing INI section [{section}]")
    block = match.group(1)
    key_pattern = re.compile(
        rf"(?m)^(\s*{re.escape(key)}\s*=\s*)([^#\n]*)(#.*)?$"
    )

    def replacement(found: re.Match[str]) -> str:
        old_value = found.group(2)
        comment = found.group(3) or ""
        gap = old_value[len(old_value.rstrip()):]
        if comment and not gap:
            gap = " "
        return f"{found.group(1)}{value}{gap}{comment}"

    replaced, count = key_pattern.subn(replacement, block, count=1)
    if count != 1:
        raise KeyError(f"missing INI key [{section}] {key}")
    return text[:match.start(1)] + replaced + text[match.end(1):]


def _atomic_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-base", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--signs", nargs="+", default=["0.02", "-0.02"])
    parser.add_argument("--mag-min", type=float, default=18.0)
    parser.add_argument("--mag-max", type=float, default=28.0)
    parser.add_argument("--re-min", type=float, default=0.1)
    parser.add_argument("--re-max", type=float, default=1.5)
    args = parser.parse_args()

    source = Path(args.original_base).resolve()
    output = Path(args.output_base).resolve()
    if source == output:
        raise ValueError("source and output bases must differ")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to populate non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_noise = source / "noise.csv"
    if not source_noise.is_file():
        raise FileNotFoundError(source_noise)
    shutil.copy2(source_noise, output / "noise.csv")

    records: list[dict[str, object]] = []
    for case in args.cases:
        kept_ids_by_sign: dict[str, np.ndarray] = {}
        property_reference: pd.DataFrame | None = None
        for sign in args.signs:
            source_cat = source / f"gals{case}_{sign}.feather"
            source_ini = source / f"sim_config_case{case}_{sign}.ini"
            if not source_cat.is_file() or not source_ini.is_file():
                raise FileNotFoundError(f"missing source pair member: {source_cat} or {source_ini}")

            frame = pd.read_feather(source_cat)
            required = {"index", "r", "Re"}
            missing = required - set(frame.columns)
            if missing:
                raise KeyError(f"{source_cat} is missing {sorted(missing)}")
            r = frame["r"].to_numpy(float)
            reff = frame["Re"].to_numpy(float)
            finite = np.isfinite(r) & np.isfinite(reff)
            bright = finite & (r <= args.mag_min)
            faint = finite & (r >= args.mag_max)
            small = finite & (reff <= args.re_min)
            large = finite & (reff >= args.re_max)
            keep = finite & (r > args.mag_min) & (r < args.mag_max)
            keep &= (reff > args.re_min) & (reff < args.re_max)

            ids = frame.loc[keep, "index"].to_numpy(np.int64)
            kept_ids_by_sign[sign] = ids
            properties = frame.loc[:, ["index", "r", "Re"]].reset_index(drop=True)
            if property_reference is None:
                property_reference = properties
            else:
                if not property_reference.equals(properties):
                    raise ValueError(f"case {case}: input identities/properties differ across shear signs")

            target_cat = output / source_cat.name
            target_ini = output / source_ini.name
            if target_cat.exists() or target_ini.exists():
                raise FileExistsError(f"refusing to overwrite {target_cat} or {target_ini}")
            _atomic_feather(frame.loc[keep].reset_index(drop=True), target_cat)

            ini = source_ini.read_text(encoding="utf-8")
            if str(source) not in ini:
                raise ValueError(f"source base path not found in {source_ini}")
            ini = ini.replace(str(source), str(output))
            ini = _replace_ini_value(ini, "GalInfo", "mag_cut", f"{args.mag_min:g}, {args.mag_max:g}")
            ini = _replace_ini_value(ini, "GalInfo", "Re_cut", f"{args.re_min:g}, {args.re_max:g}")
            ini = _replace_ini_value(ini, "CrossMatch", "mag_faint_cut", f"{args.mag_max:g}")
            _atomic_text(ini, target_ini)

            flux = np.power(10.0, -0.4 * (r - 30.0), where=np.isfinite(r),
                            out=np.zeros_like(r))
            records.append({
                "case": case,
                "sign": sign,
                "n_input": int(len(frame)),
                "n_kept": int(keep.sum()),
                "keep_fraction": float(keep.mean()),
                "n_bright": int(bright.sum()),
                "n_faint": int(faint.sum()),
                "n_small": int(small.sum()),
                "n_large": int(large.sum()),
                "n_any_ood": int((~keep).sum()),
                "flux_fraction_kept": float(flux[keep].sum() / flux.sum()),
                "flux_fraction_ood": float(flux[~keep].sum() / flux.sum()),
            })
            print(
                f"case={case} sign={sign}: {keep.sum():,}/{len(frame):,} kept "
                f"({keep.mean():.2%}); OOD bright={bright.sum():,} faint={faint.sum():,} "
                f"small={small.sum():,} large={large.sum():,}; "
                f"OOD flux={flux[~keep].sum() / flux.sum():.2%}",
                flush=True,
            )

        reference = kept_ids_by_sign[args.signs[0]]
        for sign in args.signs[1:]:
            if not np.array_equal(reference, kept_ids_by_sign[sign]):
                raise ValueError(f"case {case}: kept IDs differ for signs {args.signs[0]} and {sign}")

    summary = {
        "original_base": str(source),
        "output_base": str(output),
        "cases": args.cases,
        "signs": args.signs,
        "strict_domain": {
            "mag": [args.mag_min, args.mag_max],
            "Re_arcsec": [args.re_min, args.re_max],
        },
        "method": "filter already-generated paired per-case catalogues; preserve IDs/positions/noise",
        "records": records,
    }
    _atomic_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                 output / "indomain_filter_manifest.json")
    print(f"wrote {len(records)} filtered catalogues and manifest under {output}", flush=True)


if __name__ == "__main__":
    main()
