# Infer V1 development archive

These files record the experiments that led to the frozen Infer V1 setup.
They are not active package modules, command-line entry points, or CI tests.
Paths quoted in `doc/WORKLOG.md` describe their locations when the experiments
ran; this archive preserves the corresponding source.

- `legacy-score/` is the superseded §5B catalogue-draw study.
- `sampling-diagnostics/` contains the completed K/M, candidate-capture, and
  finite-M studies.
- `optimization/` contains approaches that were benchmarked but not adopted,
  including the JAX path.
- `galsbi-prior/` contains the rejected GalSBI-prior experiments. Infer V1 uses
  the default 200-case FS2 prior instead.

Archived tests are intentionally excluded from normal pytest discovery. They
are provenance fixtures, not a supported historical test environment.
