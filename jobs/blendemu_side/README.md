# blendemu-side jobs

These jobs do **not** run SBSI code. Each one `cd`s into the blendemu checkout, runs a
script from `blendemu/scripts/`, and writes into `blendemu/models/`. They are kept here
only because the emulator tags they produce (`lsst_r_extnbr_ho`, `lsst_r_extnbr_cw`, ...)
are the ones SBSI's R_blend step consumes, so the provenance of those weights matters to
this project's results.

They are separated from `jobs/` so that the main job directory contains only jobs that
run SBSI itself. If you are reproducing SBSI from finished catalogues you never need
these; if you are regenerating the emulator, prefer running them from the blendemu repo.

| job | what it produces |
|---|---|
| `job_retrain_extnbr.sh` / `job_retrain_extnbr_cpu.sh` | extended-neighbour emulator |
| `job_retrain_ho.sh` | `lsst_r_extnbr_ho` — held-out retrain (gold cases 0-39 excluded) |
| `job_retrain_cw.sh` | `lsst_r_extnbr_cw` — close-weighted variant |
| `job_emu_reweight.sh` | gold-reweighting / mean-residual diagnostics |
| `job_extnbr_perdist.sh` | per-distance emulator mean residual |

Their `python -u scripts/...` lines resolve against **blendemu's** `scripts/` because of
the preceding `cd`; they are not references to `SBSI/scripts/`.
