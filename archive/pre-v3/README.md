# Pre-V3 archive

This directory preserves the exploratory code, Slurm wrappers, diagnostics,
tests, figures, and result products that preceded the frozen V3 milestone.
Nothing here is part of the supported API.

The frozen release record is defined in `MILESTONE.md`; optional external model
paths are listed in `sbs_shear/models.py`. Reproduce new work through the `sbs_shear` package;
do not promote an archived script back into the main tree without first moving
its reusable logic into the package and documenting the change in
`WORKLOG.md`.

Large catalogue and diagnostic files are retained here for provenance.  The
five supported result products remain under the top-level `results/`
directory.
