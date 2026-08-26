#!/usr/bin/env python
"""Sweep for the shadowing class that actually bit: a name ASSIGNED before a loop, READ
inside it above the rebinding line, and REBOUND inside it to something not derived from it.

WHY THIS EXISTS AND WHAT ruff DOES NOT DO.  `ruff --select PLW2901` fires only when the
rebound name is the loop TARGET.  The bug that killed three GPU jobs 21 s in was neither:
`pre = est.bundle.condition_preprocessor` was an ordinary function local, read on every
draw, clobbered by a later `pre = ev[:, :m + 1]` in the same loop.  A clean PLW2901 run
says nothing about this class, so do not read one as coverage.

WHAT MAKES IT USABLE.  The "read above the rebinding line" clause.  It is the mechanism of
the bug -- iteration 1 uses the pre-loop value, iteration 2 gets the clobbered one, which
is why a one-iteration smoke run survives it -- and it is what separates the real thing
from the ubiquitous harmless reuse of a short name (`s`, `d`) by a later independent loop.
Without the clause the repo returns 59 candidates; with it, 12.

The remaining filters: comprehension targets are excluded (own scope in Python 3, cannot be
victims); self-refines (`batch = batch.to(dev)`, `reservoir = concat([reservoir, x])`) are
excluded because the new value is derived from the old; augmented assignment is a
self-refine by definition.  Assignments nested in `if`/`try` count -- only nested function
and class bodies are skipped.

VALIDATION IS A TEST, NOT A CLAIM.  `tests/test_loop_shadowing_sweep.py` asserts this
sweep fires on `tests/fixtures/shadow_bug_sample.py` and stays silent on
`tests/fixtures/shadow_benign_sample.py`.  A detector observed to fire but never observed
to stay quiet is not evidence.  Two earlier drafts of this file passed casual inspection
and were wrong -- one counted comprehension variables as pre-loop reads, the other dropped
an isinstance guard so the walk descended into `if` bodies and reported the `if` line
instead of the assignment.  Only the fire/silence check caught them.

ALLOWLIST.  `scripts/loop_shadowing_allowlist.txt` holds the hits reviewed and judged
deliberate loop-carried state.  Keyed by `path::function::name`, NOT by line number: line
numbers drift on every edit above them, and an allowlist that goes stale is worse than
none because it silently re-hides reviewed hits or floods the output with false new ones.
Run with --all to see everything including allowlisted entries.

Exit status is 1 when there is a hit NOT in the allowlist, so this can gate CI.
"""
import argparse
import ast
import io
import pathlib
import sys

ASSIGN = (ast.Assign, ast.AnnAssign, ast.AugAssign)
SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
ALLOWLIST = pathlib.Path(__file__).with_name("loop_shadowing_allowlist.txt")


class Hit:
    __slots__ = ("path", "lineno", "func", "name", "source")

    def __init__(self, path, lineno, func, name, source):
        self.path, self.lineno, self.func = path, lineno, func
        self.name, self.source = name, source

    @property
    def key(self):
        return f"{self.path}::{self.func}::{self.name}"

    def __repr__(self):
        return f"<Hit {self.path}:{self.lineno} {self.func}() {self.name!r}>"


def _stores(node):
    return {t.id for t in ast.walk(node)
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}


def _loads(node):
    return {t.id for t in ast.walk(node)
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load)}


def _own_nodes(node, types):
    """Nodes of `types` in `node`'s OWN scope, not descending into nested scopes."""
    out, stack = [], list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, SCOPED):
            continue
        if isinstance(n, types):
            out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _own_assigns(node):
    """Assignment statements inside `node`, not descending into nested scopes."""
    return _own_nodes(node, ASSIGN)


def scan_source(src, path="<string>"):
    """Return the Hits in one module's source, sorted by position."""
    hits = []
    tree = ast.parse(src, str(path))
    # THE MODULE BODY IS A SCOPE.  Until it was added here, this sweep read function bodies
    # only, which made module scope the one place the defect it looks for could sit
    # unreported -- and a checker is SILENT on an unscanned region in exactly the way it is
    # silent on clean code, so the blind spot could not be told from a clean bill.  Found by
    # lifting a loop into a function and watching the sweep fire on code it had been sitting
    # next to, unchanged, all along.  Measured when closed: two hits on the unscanned side,
    # both best-so-far trackers, nothing worse -- but that is a result, not what it was
    # before it was run.
    scopes = [(tree, "<module>")] + [(n, n.name) for n in ast.walk(tree)
                                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn, fname in scopes:
        comp_names = set()
        # SCOPE-RESTRICTED, not ast.walk: with the module in the list, walking would attribute
        # every nested function's loops to `<module>` as well and report them twice.
        for c in _own_nodes(fn, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if isinstance(c, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for g in c.generators:
                    comp_names |= _stores(g.target)
        for loop in _own_nodes(fn, (ast.For, ast.While)):
            loop_targets = _stores(loop.target) if isinstance(loop, ast.For) else set()
            bound_before = set()
            for a in _own_assigns(fn):
                if a.lineno < loop.lineno:
                    bound_before |= _stores(a)
            for stmt in _own_assigns(loop):
                if isinstance(stmt, ast.AugAssign):
                    continue
                value = getattr(stmt, "value", None)
                rhs = _loads(value) if value is not None else set()
                for name in _stores(stmt):
                    if name in loop_targets or name in comp_names:
                        continue
                    if name not in bound_before or name in rhs:
                        continue
                    read_above = any(
                        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                        and n.id == name and loop.lineno < n.lineno < stmt.lineno
                        for n in ast.walk(loop))
                    if read_above:
                        seg = (ast.get_source_segment(src, stmt) or "?").splitlines()[0]
                        hits.append(Hit(str(path), stmt.lineno, fname, name,
                                        seg.strip()[:76]))
    return sorted(hits, key=lambda h: (h.path, h.lineno, h.name))


def scan_paths(paths):
    hits = []
    for p in paths:
        hits.extend(scan_source(io.open(p, encoding="utf-8").read(), p))
    return sorted(hits, key=lambda h: (h.path, h.lineno, h.name))


def default_paths(root="."):
    root = pathlib.Path(root)
    return sorted(root.glob("sbsi/**/*.py")) + sorted(root.glob("scripts/**/*.py"))


def load_allowlist(path=ALLOWLIST):
    if not pathlib.Path(path).exists():
        return set()
    out = set()
    for line in io.open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="files to scan (default: sbsi/ and scripts/)")
    ap.add_argument("--all", action="store_true", help="also print allowlisted hits")
    ap.add_argument("--no-allowlist", action="store_true")
    args = ap.parse_args(argv)

    hits = scan_paths([pathlib.Path(p) for p in args.paths] if args.paths
                      else default_paths())
    allow = set() if args.no_allowlist else load_allowlist()
    new = [h for h in hits if h.key not in allow]
    known = [h for h in hits if h.key in allow]

    for h in new:
        print(f"NEW  {h.path}:{h.lineno}: in {h.func}(): {h.name!r} rebound to something "
              f"not derived from it,\n     after being read above in the same loop\n"
              f"       {h.source}")
    if args.all:
        for h in known:
            print(f"ok   {h.path}:{h.lineno}: in {h.func}(): {h.name!r}  (allowlisted)")

    print(f"\n{len(new)} new, {len(known)} reviewed/allowlisted, {len(hits)} total.")
    if new:
        print("Each NEW hit needs eyes.  Deliberate loop-carried state (a fixed-point\n"
              "update, a best-so-far tracker, a binary search) is fine -- add it to\n"
              f"{ALLOWLIST.name} with a reason.  A rebind that means something DIFFERENT\n"
              "from what the loop reads above it is the bug.")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
