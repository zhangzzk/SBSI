# Agent instructions for SBSI

SBSI is a general shear-calibration library. It has one workflow; scientific
release names such as V3 and V3b are only convenience references to external
model paths.

All project documentation lives in `doc/`. Only this file, `CLAUDE.md`, and
`README.md` stay at the repository root:

- `doc/API.md` — the BlendEMU scope boundary, the general API contract, and the
  public API.
- `doc/CONVENTIONS.md` — the definitions every reported number must cite:
  catalogues, population-cut order, shape kinds, response estimators, seeds,
  and the definition of `m`.
- `doc/ENVIRONMENT.md` — interpreters, running the test suite, the BlendEMU
  import boundary, and the login-node/scheduler policy.
- `doc/WORKLOG.md` — the dated record of every substantive change, newest
  first. Read the top entries for current state; find older material by
  searching for a keyword, date, or `cont.NNN` tag.

`doc/` is pulled on demand, and by section, not cover to cover. Read headlines first and then the relevant entries. When writing a doc, keep it navigable this way: greppable headings, and cite sections by anchor, not by file alone.

## Rules

- Classifiers, conditional flows, response prediction, validation, and Bayesian
  inference stay in SBSI; BlendEMU owns image simulation, measurement,
  simulation-catalogue production, and emulator training. Boundary detail:
  `doc/API.md`.
- Unmatched rows between catalogues should be dropped or rejected and be reported.
- No empirical offsets, scaling factors, or pasted results to make a number
  agree with an expectation. Use common random numbers when differencing
  stochastic flow evaluations, and apply the same forward or antithetic
  extraction convention to simulation and model sides (`doc/CONVENTIONS.md`
  §6c–6d).
- The login node is for edits, syntax checks, and other light work; other nontrivial heavy CPU/GPU work go through the local
  scheduler. Interpreters and test commands: `doc/ENVIRONMENT.md`.

## Work log

After every substantive change, prepend a dated entry to `doc/WORKLOG.md` covering
files and behavior changed, validation commands and outputs, limitations, and
next steps.
