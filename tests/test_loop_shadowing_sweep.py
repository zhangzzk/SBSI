"""The sweep's validation, as a test rather than a claim in a message.

A detector that has been observed to fire but never observed to STAY QUIET is not
evidence -- it could be a detector that always fires.  These tests pin both halves against
fixtures, so the property survives drift in the repo and is reproducible by anyone.

Two earlier drafts of the sweep passed casual inspection and were wrong: one counted
comprehension variables as pre-loop reads (59 candidates instead of 12), the other dropped
an isinstance guard so the walk descended into `if` bodies and reported the `if` line
instead of the assignment.  Only the fire/silence pair caught them.
"""
import io
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sweep_loop_shadowing as sweep  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BUG = FIXTURES / "shadow_bug_sample.py"
BENIGN = FIXTURES / "shadow_benign_sample.py"
REVIEWED = FIXTURES / "shadow_reviewed_sample.py"
MODSCOPE = FIXTURES / "shadow_module_scope_sample.py"


def test_the_sweep_fires_on_the_bug_that_killed_the_gpu_jobs():
    """`pre` bound before the loop, read inside it, rebound to a tensor."""
    hits = sweep.scan_paths([BUG])
    names = {h.name for h in hits}
    assert names == {"pre"}, f"expected exactly the `pre` rebind, got {hits}"
    assert hits[0].func == "draw_loop"
    assert "ev[:, :m + 1]" in hits[0].source


def test_the_sweep_fires_at_MODULE_SCOPE_and_attributes_each_scope_correctly():
    """The blind spot, closed and then validated -- because closing it is not evidence.

    The sweep read function bodies only, so module scope was the one place this defect could
    live unreported.  That is worse than an uncovered file: an unscanned region is silent in
    exactly the way clean code is silent, so the tool's quiet could not be read.

    Two assertions, and the second is the one that would break under the obvious
    implementation.  Adding the module to the scope list while still walking the whole tree
    per scope attributes every nested function's loops to `<module>` as well, and reports
    each one twice; the fixture carries a nested function so that regression is caught here
    rather than as noise in a sweep months later.
    """
    hits = sweep.scan_paths([MODSCOPE])
    assert {(h.func, h.name) for h in hits} == {("<module>", "PRE"), ("helper", "acc")}, hits
    assert len(hits) == 2, f"each hit must be reported once, not once per enclosing scope: {hits}"


def test_the_sweep_stays_silent_on_self_refines():
    """Rebinds DERIVED from the old value must not be reported.

    Without this half, a sweep that returned every assignment would 'pass' the fire test.
    """
    hits = sweep.scan_paths([BENIGN])
    assert hits == [], f"false positives: {hits}"


def test_a_deliberate_fixed_point_IS_reported_and_that_is_correct():
    """The sweep reports intent-dependent rebinds; the allowlist records the judgement.

    Written as a test because the tempting 'fix' is to suppress this shape, and the real
    bug has the same shape.  `sbsi/inference.py::infer_shear::shear` is the live instance.
    """
    hits = sweep.scan_paths([REVIEWED])
    assert [h.name for h in hits] == ["shear", "shear"], hits
    assert all(h.func == "fixed_point" for h in hits)


def test_a_self_refine_is_not_a_hit_even_when_read_above():
    src = (
        "def f(rows, dev):\n"
        "    x = load()\n"
        "    for r in rows:\n"
        "        use(x)\n"
        "        x = x.to(dev)\n"
    )
    assert sweep.scan_source(src, "inline") == []


def test_a_rebind_read_only_BELOW_it_is_not_a_hit():
    """The mechanism needs the read ABOVE: that is what makes iteration 2 differ from 1."""
    src = (
        "def f(rows):\n"
        "    x = load()\n"
        "    for r in rows:\n"
        "        x = compute(r)\n"
        "        use(x)\n"
    )
    assert sweep.scan_source(src, "inline") == []


def test_a_comprehension_variable_is_not_a_victim():
    """Comprehension targets have their own scope in Python 3 -- they flooded draft one."""
    src = (
        "def f(rows):\n"
        "    y = [s for s in rows]\n"
        "    for r in rows:\n"
        "        use(s_outer)\n"
        "        s = r\n"
    )
    assert [h.name for h in sweep.scan_source(src, "inline")] == []


def test_an_assignment_nested_in_an_if_is_still_reported():
    """Draft two reported the `if` line instead of the assignment inside it."""
    src = (
        "def f(rows, flag):\n"
        "    p = setup()\n"
        "    for r in rows:\n"
        "        use(p)\n"
        "        if flag:\n"
        "            p = r\n"
    )
    hits = sweep.scan_source(src, "inline")
    assert [h.name for h in hits] == ["p"]
    assert hits[0].source == "p = r", f"must anchor on the assignment, got {hits[0].source}"


def test_the_repo_has_no_unreviewed_hits():
    """The regression guard.  A NEW hit means someone wrote the pattern again.

    If this fails, read the loop.  Deliberate loop-carried state goes in the allowlist
    WITH a reason; a rebind that means something different from what the loop reads above
    it is the bug, and it should be renamed instead.
    """
    hits = sweep.scan_paths(sweep.default_paths(REPO))
    allow = sweep.load_allowlist()
    # keys are repo-relative, so scan from the repo root the same way the CLI does
    new = [h for h in hits
           if str(pathlib.Path(h.path).relative_to(REPO)) + f"::{h.func}::{h.name}"
           not in allow]
    assert not new, ("unreviewed loop-shadowing hits:\n  "
                     + "\n  ".join(f"{h.path}:{h.lineno} {h.func}() {h.name!r}"
                                   for h in new))


def test_the_allowlist_has_no_dead_entries():
    """A stale allowlist silently re-hides a reviewed line that moved or was deleted."""
    hits = sweep.scan_paths(sweep.default_paths(REPO))
    live = {str(pathlib.Path(h.path).relative_to(REPO)) + f"::{h.func}::{h.name}"
            for h in hits}
    dead = sorted(sweep.load_allowlist() - live)
    assert not dead, f"allowlist entries that no longer match any hit: {dead}"


def test_the_cli_exits_nonzero_only_on_a_new_hit():
    """So it can gate CI without anyone reading the output."""
    ok = subprocess.run([sys.executable, str(REPO / "scripts" / "sweep_loop_shadowing.py")],
                        cwd=REPO, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "sweep_loop_shadowing.py"),
                          str(BUG), "--no-allowlist"], cwd=REPO,
                         capture_output=True, text=True)
    assert bad.returncode == 1
    assert "pre" in bad.stdout
