# Research archive — 2026-09-14

This directory preserves the experimental surface removed during the
`V3.5-like` repository cleanup. It is provenance, not supported runtime code,
and is excluded from package installation and test discovery.

Contents:

- `sbsi/` — 20 response-training and experimental flow modules.
- `scripts/` — 63 training, ConstGold, plotting, and branch-review commands.
- `jobs/` — 107 experiment-specific scheduler wrappers.
- `tests/` — 55 tests for the archived experimental surface.
- `doc/` — detailed experiment notes, the complete historical work log, and
  pre-cleanup copies of the canonical documentation.
- `configs/` — retired V3.3-like and fresh-acceptance experiment configurations.
- `examples/` and `figures/` — training examples and generated research figures.

The archive was produced by moves or byte-for-byte copies; scientific results
were not rewritten. Paths inside old scripts and documents are historical and
are not expected to resolve from this directory. Git history remains the
authoritative source for files that were already tracked before the cleanup.

The active repository retains only likelihood construction, model loading,
selection normalization, finite-prior sampling, and inference. `V3.5-like`
is a path-only development model identity and is not a 0.3%-calibration claim.
