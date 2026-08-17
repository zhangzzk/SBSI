# Agent instructions for SBSI

SBSI is a general shear-calibration library. It has one workflow; scientific
release names such as V3 and V3b are only convenience references to external
model paths. Read `doc/CONVENTIONS.md` before producing a science number.

All project documentation lives in `doc/`. Only this file, `CLAUDE.md`, and
`README.md` stay at the repository root.

## Scope

- Keep classifiers, conditional flows, response prediction, validation, and
  Bayesian inference in SBSI.
- BlendEMU owns image simulation, measurement, simulation-catalogue production,
  and emulator training. SBSI may call those supported APIs/CLIs but must not
  copy their rendering or measurement implementation.
- SBSI owns forward-response catalogue preparation from a user truth
  catalogue: input validation, nearest-neighbour finding, the one-row-per-primary
  flow scene view, pair construction, training-matched cuts and rescaling, and
  keyed alignment of emulator predictions.
- Do not copy SBSI inference or flow code into BlendEMU, or move BlendEMU
  catalogue builders here, unless the owner explicitly changes this boundary.
- The next emulator training/tuning update belongs in BlendEMU.

## General API contract

- Every training, validation, prior, and inference catalogue is supplied by the
  user as a DataFrame or external path. Never hide a project catalogue in an API
  default.
- Flow checkpoints and emulator artifacts are explicit paths. `get_model("V3")`
  and `get_model("V3b")` are path-only conveniences; no behavior may branch on
  those names.
- Model support is read from checkpoint metadata or supplied explicitly. Never
  infer a domain from a filename.
- Reusable behavior belongs in `sbs_shear/`. `examples/job_blendemu.sh`
  is a deployment example that calls BlendEMU; it is not part of the workflow
  or imported by the library.
- `archive/pre-v3/` is provenance, not supported code.

## Public API

1. `sbsi flow`: config-driven conditional-flow training and explicit tuning,
   backed by the reusable implementation in `sbs_shear.flow`.
2. `sbs_shear.response`: ensemble response prediction and blend-response
   composition.
3. `sbs_shear.inference`: the future simulation-based catalogue likelihood and
   shear inference. Do not claim this is validated until its TODO boundary is
   resolved.

Both response components are always required:

```text
R_model = R_flow + R_blend
m = R_sim / R_model - 1
```

`R_flow` is self-response and is never the complete response. Emulator responses
come from a user-supplied aligned column/array or external catalogue. Unmatched
rows are dropped or rejected, never assigned `R_blend = 0`.

## Numerical integrity

- Do not introduce empirical offsets, scaling factors, or pasted results to make
  a number agree with an expectation.
- Derive ratios per seed and then summarize them.
- Use all flow checkpoints selected by the user for a shape-response result.
- Use common random numbers when differencing stochastic flow evaluations.
- Apply the same forward or antithetic extraction convention to simulation and
  model sides. See `doc/CONVENTIONS.md`.

## Work log

After every substantive change, prepend a dated entry to `doc/WORKLOG.md` covering
files and behavior changed, validation commands and outputs, limitations, and
next steps.

## Resources and environment

- On the project login node, perform only edits, syntax checks, metadata
  inspection, and tiny smoke tests. Submit training, full-catalogue scans,
  response production, simulations, and other nontrivial CPU/GPU work through
  the local scheduler.
- The scheduling layer belongs to the user/deployment, not the SBSI source API.
- Conda environment: `sims1`; the repository is not installed in production.

```bash
conda activate sims1
export PYTHONPATH="$PWD:$PYTHONPATH"        # add BlendEMU only for the emulator step
```

`sims1` has no pytest, so run the suite with the `py31` interpreter (see `CLAUDE.md`):

```bash
PYTHONPATH="$PWD" /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m pytest tests/ -q
```

The suite must pass on a checkout that has SBSI only; the single BlendEMU cross-check
skips when BlendEMU is absent.

Preserve the canonical spin-2 convention
`(q1, q2) = q(cos 2 theta, sin 2 theta)`. Validate analytic gradients against
finite differences before treating them as science results.
