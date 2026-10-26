"""Loading user-owned catalogues at the SBSI API boundary."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd


Catalogue = Union[pd.DataFrame, str, PathLike]


def load_catalogue(catalogue: Catalogue, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Return a catalogue as a DataFrame.

    SBSI never chooses a training, validation, or inference catalogue.  The
    caller supplies either a DataFrame or a path to a Feather, Parquet, CSV, or
    pickle file.  ``columns`` is forwarded when the file format supports column
    projection.
    """

    if isinstance(catalogue, pd.DataFrame):
        if columns is None:
            return catalogue.copy()
        missing = sorted(set(columns) - set(catalogue.columns))
        if missing:
            raise KeyError(f"catalogue is missing columns: {missing}")
        return catalogue.loc[:, list(columns)].copy()

    path = Path(catalogue).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow"}:
        return pd.read_feather(path, columns=columns)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
        if columns is not None:
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise KeyError(f"catalogue is missing columns: {missing}")
            frame = frame.loc[:, list(columns)]
        return frame
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"pickle does not contain a DataFrame: {path}")
        if columns is not None:
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise KeyError(f"catalogue is missing columns: {missing}")
            frame = frame.loc[:, list(columns)]
        return frame.copy()
    raise ValueError(
        f"unsupported catalogue format {suffix!r}; use Feather, Parquet, CSV, or pickle"
    )


__all__ = ["Catalogue", "load_catalogue"]
