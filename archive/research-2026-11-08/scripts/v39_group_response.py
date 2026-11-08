"""Bounded reuse of shear-independent disk coefficients for one fixed group."""

import numpy as np


class GroupResponseMemo:
    """Keep the original calculation, retaining its values across shear nodes.

    A full centre scan populates this table before arbitrary candidate/tail
    queries. Its lifetime is one observation group, never the whole catalogue.
    Replay mode checks exact equality with the original implementation.
    """

    def __init__(self, likelihood, raw):
        self.likelihood = likelihood
        self.original = likelihood._coefficients
        self.raw = np.array(raw, dtype=float, copy=True)
        self.values = np.full((len(raw), likelihood.disk_response.n_atoms), np.nan)
        self.replay = False
        self.replay_calls = 0
        self.failure = None

    def __enter__(self):
        self.likelihood._response_cache.clear()
        self.likelihood._coefficients = self
        return self

    def __exit__(self, *exc):
        self.likelihood._coefficients = self.original
        self.likelihood._response_cache.clear()

    def __call__(self, indices, raw, valid):
        if not np.array_equal(raw, self.raw):
            raise ValueError("Response memo belongs to one fixed observation group")
        rows, columns = np.nonzero(valid)
        atoms = indices[rows, columns]
        stored = self.values[rows, atoms]
        if np.isnan(stored).any() or self.replay:
            original = self.original(indices, raw, valid)
            fresh = original[rows, columns]
            if not np.isfinite(fresh).all():
                raise ValueError("Nonfinite disk-response coefficients")
            known = np.isfinite(stored)
            if not np.array_equal(stored[known], fresh[known]):
                self.failure = dict(
                    rows=rows[known], atoms=atoms[known], reference=stored[known], actual=fresh[known]
                )
            np.testing.assert_array_equal(stored[known], fresh[known])
            self.values[rows, atoms] = fresh
            if self.replay:
                self.replay_calls += 1
            return original
        result = np.zeros(indices.shape, dtype=np.float64)
        result[rows, columns] = stored
        return result
