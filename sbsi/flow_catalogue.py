"""Preparation helpers for compact measurement-flow catalogues.

The simulation and measurement products are external inputs.  This module only
selects flow-usable rows and attaches keyed, user-supplied scene summaries; it
does not render or measure images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc


KEY_MULTIPLIER = 1_000_003


def object_keys(case, input_index):
    return np.asarray(case, dtype=np.int64) * KEY_MULTIPLIER + np.asarray(
        input_index, dtype=np.int64
    )


def load_keyed_columns(path: Path, columns: Sequence[str]):
    """Load and sort a keyed lookup, rejecting conflicting duplicate rows."""

    table = pf.read_table(path, columns=["case", "input_index", *columns])
    frame = table.to_pandas()
    keys = object_keys(frame["case"], frame["input_index"])
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    values = {name: frame[name].to_numpy()[order] for name in columns}
    duplicate = np.r_[False, keys[1:] == keys[:-1]]
    if duplicate.any():
        previous = np.flatnonzero(duplicate) - 1
        current = previous + 1
        for name, array in values.items():
            left, right = array[previous], array[current]
            same = np.isclose(left, right, equal_nan=True) if np.issubdtype(
                array.dtype, np.number
            ) else left == right
            if not np.all(same):
                raise ValueError(f"conflicting duplicate keys in {path} column {name!r}")
        keep = ~duplicate
        keys = keys[keep]
        values = {name: array[keep] for name, array in values.items()}
    return keys, values


def prepare_flow_catalogue(
    input_path: Path,
    output_path: Path,
    *,
    columns: Sequence[str],
    lookups: Mapping[Path, Sequence[str]],
    max_case: Optional[int] = None,
    progress_every: int = 100,
):
    """Stream a compact detected+finite catalogue and attach keyed summaries.

    Rows missing any requested lookup are dropped and reported.  This is
    deliberate: zero-filling a missing crowding value would turn an unmatched
    row into a falsely isolated scene.
    """

    lookup_data = []
    for path, names in lookups.items():
        keys, values = load_keyed_columns(Path(path), names)
        lookup_data.append((str(path), keys, values))
        print(f"lookup {path}: {len(keys):,} unique objects; columns={list(names)}")

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer_holder = [None]
    raw_rows = selected_rows = output_rows = unmatched_rows = 0
    cases_seen = set()
    try:
        with ipc.open_file(input_path) as reader:
            available = set(reader.schema.names)
            required = set(columns) | {
                "case",
                "input_index",
                "detected",
                "measured_ngmix_g1",
                "measured_ngmix_g2",
            }
            missing = sorted(required - available)
            if missing:
                raise KeyError(f"input catalogue lacks required columns: {missing}")
            read_columns = [name for name in columns if name in available]
            for required_name in required:
                if required_name not in read_columns:
                    read_columns.append(required_name)

            for batch_index in range(reader.num_record_batches):
                batch = pa.Table.from_batches([reader.get_batch(batch_index)]).select(
                    read_columns
                ).to_pandas()
                raw_rows += len(batch)
                if max_case is not None:
                    batch = batch[batch["case"].astype(int) < max_case]
                usable = batch["detected"].astype(bool).to_numpy()
                usable &= np.isfinite(batch["measured_ngmix_g1"].to_numpy(float))
                usable &= np.isfinite(batch["measured_ngmix_g2"].to_numpy(float))
                batch = batch.loc[usable].reset_index(drop=True)
                selected_rows += len(batch)
                if batch.empty:
                    continue

                batch_key = object_keys(batch["case"], batch["input_index"])
                matched_all = np.ones(len(batch), dtype=bool)
                pending = {}
                for path, keys, values in lookup_data:
                    position = np.searchsorted(keys, batch_key)
                    in_bounds = position < len(keys)
                    safe_position = np.minimum(position, len(keys) - 1)
                    matched = in_bounds & (keys[safe_position] == batch_key)
                    matched_all &= matched
                    for name, array in values.items():
                        if name in pending:
                            raise ValueError(f"lookup column requested more than once: {name}")
                        pending[name] = array[safe_position]
                unmatched_rows += int((~matched_all).sum())
                batch = batch.loc[matched_all].reset_index(drop=True)
                for name, values in pending.items():
                    batch[name] = values[matched_all]
                if batch.empty:
                    continue
                cases_seen.update(batch["case"].astype(int).unique().tolist())
                table = pa.Table.from_pandas(batch, preserve_index=False)
                if writer_holder[0] is None:
                    writer_holder[0] = ipc.new_file(str(output_path), table.schema)
                writer_holder[0].write_table(table)
                output_rows += len(batch)
                if progress_every and (batch_index + 1) % progress_every == 0:
                    print(
                        f"batches={batch_index + 1:,}, raw={raw_rows:,}, "
                        f"usable={selected_rows:,}, unmatched={unmatched_rows:,}, "
                        f"written={output_rows:,}",
                        flush=True,
                    )
    finally:
        if writer_holder[0] is not None:
            writer_holder[0].close()

    if output_rows == 0:
        raise RuntimeError("no rows were written")
    print(
        f"wrote {output_path}: {output_rows:,} rows, {len(cases_seen):,} cases; "
        f"raw={raw_rows:,}, usable={selected_rows:,}, unmatched_dropped={unmatched_rows:,}",
        flush=True,
    )
    return {
        "raw_rows": raw_rows,
        "usable_rows": selected_rows,
        "unmatched_rows": unmatched_rows,
        "output_rows": output_rows,
        "cases": sorted(cases_seen),
    }


__all__ = ["KEY_MULTIPLIER", "object_keys", "prepare_flow_catalogue"]
