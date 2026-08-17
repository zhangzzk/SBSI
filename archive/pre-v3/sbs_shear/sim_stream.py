"""Shared streaming + reservoir loader for the (multi-GB) SBSI catalogues.

Replaces the ~half-dozen near-identical ``load``/``load_parent``/``load_sheared_sample``
copies that had drifted apart (different thresholds / column sets) across the diagnostic and
training scripts.  One code path => one place to keep consistent.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection


def stream_reservoir(catalogue, columns, max_rows, seed=7, detected=None,
                     shear_threshold=0.0, max_read_batches=None):
    """Stream the column-selected catalogue, apply the standard source cuts, and
    reservoir-sample (uniform weighting via a random key) down to ``max_rows``.

    Parameters
    ----------
    columns : iterable of column names, OR a callable(available_set) -> iterable, for
        schema-dependent column sets (e.g. raw_columns_for_selection_features). Only columns
        present in the file are read.
    detected : None keeps the full parent (detected + undetected); True keeps detected
        only; False keeps undetected only. (Requires the 'detected' column.)
    shear_threshold : if > 0 and gamma columns are present, keep |gamma| > threshold
        (drops the unsheared half). Use 0.0 for g=0 catalogues.
    """
    rng = np.random.default_rng(seed)
    with ipc.open_file(catalogue) as reader:
        available = set(reader.schema.names)
        wanted = columns(available) if callable(columns) else columns
        read_cols = sorted(c for c in wanted if c in available)
        reservoir = None
        for bi in range(reader.num_record_batches):
            if max_read_batches is not None and bi >= max_read_batches:
                break
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(read_cols).to_pandas()
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
            if len(batch) == 0:
                continue
            if detected is not None and "detected" in batch.columns:
                batch = batch[batch["detected"].astype(bool) == detected].reset_index(drop=True)
            if shear_threshold > 0.0 and "gamma1_input_p" in batch.columns:
                gmag = np.hypot(batch["gamma1_input_p"].to_numpy(float),
                                batch["gamma2_input_p"].to_numpy(float))
                batch = batch[gmag > shear_threshold].reset_index(drop=True)
            if len(batch) == 0:
                continue
            batch = batch.copy()
            batch["__key"] = rng.random(len(batch))
            reservoir = batch if reservoir is None else pd.concat([reservoir, batch], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    if reservoir is None:
        raise SystemExit(f"No rows selected from {catalogue}")
    if len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    return reservoir.drop(columns="__key").reset_index(drop=True)
