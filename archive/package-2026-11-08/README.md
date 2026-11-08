# Package review archive — November 8, 2026

## Contents and restoration

Baseline: `7c58bc16ab6128a68116375a2af06e471764d807`. The owner requested
review, reorganization and cleanup of `sbsi/` after the documentation cleanup.
The [manifest](manifest.json) records all original paths, archive paths, SHA256
hashes, move/snapshot actions and the complete internal module migration map.

Nineteen experimental modules and fifteen tests are retired because no retained
entry point or public root export depends on them. This includes particle/SMC,
exchange/relocation, phase controls, gradient diagnostics and unused historical
initialization. Their tests remain alongside their source. `snapshots/` retains
the original implementation, callers, tests, metadata and guides for replay.
All archive files are tracked source or documentation; no model weights moved.

To restore an experiment, use a separate checkout at the baseline or restore
its complete dependency closure to the original paths. Earlier documentation
and study runners are in `archive/research-2026-11-08/`. Do not execute from or
import directly out of this archive. Closed studies still require an owner
request before restarting scientific jobs.

## Current package

The retained code is organized under flows/galsbi/disk/finite. The package-root
loading and prediction API remains. Internal module imports change as recorded
in the manifest; no algorithm or trained parameter change is implied.
The source-capture repair and complete verification are described in the
[current work log](../../doc/WORKLOG.md#2026-11-08--review-and-reorganize-the-python-package).
