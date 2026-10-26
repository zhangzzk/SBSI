"""Parent-scene coverage for fixed-count, chunked population mocks."""
import numpy as np


def last_retained_parent(selected_ids, retained_count, parent_count, *, final_chunk):
    """Keep every parent in complete chunks; truncate only a partial final one.

An unselected original parent can produce a selected twin. The last selected
original is therefore not the last retained parent of a complete chunk.
"""
    ids = np.asarray(selected_ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or not isinstance(parent_count, (int, np.integer)) or parent_count < 1
            or not isinstance(retained_count, (int, np.integer))
            or not 0 <= retained_count <= len(ids)
            or np.any(ids < 0) or np.any(ids >= parent_count)
            or np.any(np.diff(ids) <= 0)):
        raise ValueError('ordered in-range selected parents and a valid retained count required')
    if retained_count == len(ids):
        return parent_count-1
    if not final_chunk:
        raise ValueError('only the final chunk may retain fewer than all selected originals')
    return int(ids[retained_count-1]) if retained_count else -1
