# Branches: `dev` and `master`

The repository has two long-lived branches. They track **different sets of files**
and, since 2026-08-17, they have **completely separate histories**.

| | `dev` | `master` |
|---|---|---|
| Purpose | the full working repo | the public/release view |
| `sbsi/` | tracked | tracked |
| `examples/` | tracked | tracked |
| `tests/` | tracked | tracked |
| `models/` (frozen V3 artifacts) | tracked | tracked |
| `README.md`, `pyproject.toml` | tracked | tracked |
| `doc/` | tracked | ignored |
| `archive/` | tracked | ignored |
| `AGENTS.md`, `CLAUDE.md` | tracked | ignored |
| `notebooks/`, `slides/` | tracked | ignored |
| `results/`, `figures/`, `data/`, `logs/` | ignored | ignored |

Each branch carries its own `.gitignore`. `master`'s is a whitelist: it ignores
everything at the top level and then re-admits only the public tree.

## Do your work on `dev`

`dev` is the branch to develop on. Nothing is hidden there, and `doc/WORKLOG.md` —
which `AGENTS.md` requires you to update after every substantive change — only
exists there.

## Never merge between the branches, in either direction

Two independent reasons, both of which will bite you:

1. **`master` is an orphan.** It was rebuilt on 2026-08-17 as a single root commit
   with no ancestors, so that publishing it does not expose the internal working
   history. It shares no commit with `dev`. Git will refuse to merge them without
   `--allow-unrelated-histories`, and forcing it would splice dev's whole history
   back into the public branch — exactly what the rebuild removed.

2. **`.gitignore` does not filter files that arrive through a merge.** It governs
   *untracked* files only. So a merge from `dev` would re-add `doc/`, `archive/`,
   `notebooks/`, `slides/` and the agent instruction files to `master` regardless
   of what the ignore rules say.

## Publishing: copy, do not merge

```bash
git checkout master
git checkout dev -- sbsi examples tests models README.md pyproject.toml
git commit -m "Sync public tree from dev"
```

This takes `dev`'s version of exactly those paths and leaves everything else on
`master` alone. It is a file copy, not a merge, so it is safe to repeat and it
never drags history across. If you add a new public directory, add it both to this
command and as a `!` line in `master`'s `.gitignore`.

To take a fix made directly on `master` back to `dev`, copy it the same way
(`git checkout master -- <path>`) rather than merging.

## Files stay on disk

Untracking is an index operation. `doc/`, `archive/` and the rest remain in your
working directory when you are on `master`; they are simply ignored there. You will
not lose them by switching branches, and their history is intact on `dev`.

## Recovery tags

| Tag | What it marks |
|---|---|
| `pre-branch-split-2026-08-17` | the last commit when a single branch tracked everything |
| `master-with-history-2026-08-17` | the trimmed `master` as it was *before* the orphan rebuild, still carrying the shared history |

Neither is reachable from `master` any more, so do not delete them unless you are
sure you want that history gone.
