# Branches: `dev` and `master`

The repository has two long-lived branches that track **different sets of files**.

| | `dev` | `master` |
|---|---|---|
| Purpose | the full working repo | the public/release view |
| `sbs_shear/` | tracked | tracked |
| `examples/` | tracked | tracked |
| `tests/` | tracked | tracked |
| `README.md`, `pyproject.toml` | tracked | tracked |
| `doc/` | tracked | ignored |
| `archive/` | tracked | ignored |
| `AGENTS.md`, `CLAUDE.md` | tracked | ignored |
| `notebooks/`, `slides/` | tracked | ignored |
| `models/`, `results/`, `figures/`, `data/`, `logs/` | ignored | ignored |

Each branch carries its own `.gitignore`. `master`'s is a whitelist: it ignores
everything at the top level and then re-admits only the public tree.

## Do your work on `dev`

`dev` is the branch to develop on. Nothing is hidden there, and `doc/WORKLOG.md`
— which `AGENTS.md` requires you to update after every substantive change —
only exists there.

## Never merge `dev` into `master`

This is the part that will bite you if you forget it.

**`.gitignore` does not apply to files that arrive through a merge.** It only
governs *untracked* files. So `git merge dev` while on `master` would re-add
`doc/`, `archive/`, `notebooks/`, `slides/` and the agent instruction files to
`master`, silently undoing the split.

Instead, publish by copying just the public paths out of `dev`:

```bash
git checkout master
git checkout dev -- sbs_shear examples tests README.md pyproject.toml
git commit -m "Sync public tree from dev"
```

That takes `dev`'s version of exactly those paths and leaves everything else on
`master` untouched. It is also safe to run repeatedly — it is a copy, not a merge.

To go the other way (a fix made directly on `master`), a normal
`git merge master` from `dev` is fine: `master` is a subset, so nothing
unwanted travels.

## Files stay on disk

Untracking is an index operation. `doc/`, `archive/` and the rest remain in your
working directory when you are on `master`; they are simply ignored there. You
will not lose them by switching branches, and their history is intact on `dev`.

## Recovery

The tag `pre-branch-split-2026-08-17` marks the last commit before the split, when
a single branch tracked everything.
