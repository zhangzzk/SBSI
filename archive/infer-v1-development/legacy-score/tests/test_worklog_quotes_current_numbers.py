"""Archived prose-regression test for the legacy finite-M study.

Three times a statement that was correct WHEN WRITTEN silently stopped being true, and
nothing errored:

  * a grouping key that outlived the dimension it assumed (closed permanently by
    `assert_group_is_replicates`);
  * a docstring describing `C = C_jk + C_seed/S` after the body stopped computing it;
  * prose literals from a 40-draw run surviving into a 150-draw section of cont.191.

The first was structural and is fixed structurally.  The third is the same class acting on
PROSE, and it is the one most likely to reach a reader, because prose is what gets quoted.

The fix has the same shape: do not hand-carry values a run produces.
`scripts/ess_exponent_control.py` EMITS every number cont.191 is allowed to quote into
`doc/generated/ess_exponent_control.numbers`, and this test asserts each emitted value still
appears in the entry that quotes it.  A re-run that moves a number then either forces the
prose to be updated or fails the suite, instead of leaving a section that reads correctly and
is wrong.

THREE CHECKS, AND EACH ONE EXISTS BECAUSE THE PREVIOUS ONE HAD A HOLE.  A bare value search over the entry
FAILED TO FIRE on the historical case: "1.49x" is a stale Tier-5 median and also a live
grid-scan figure in the same entry, so the search found it and passed.  Numbers that travel
together are therefore emitted as the EXACT lines the entry carries and compared VERBATIM --
a coincidence cannot fool that, because it would have to be the whole row.  The value search
is kept for the scattered prose figures, where it is a tripwire rather than a proof: it
cannot tell a value used in the right sentence from the same digits elsewhere.

AND BOTH OF THOSE ONLY PIN THE PRESENT.  "Everything emitted still appears" says nothing
about a figure somebody types into the prose by hand that no emitter covers -- it is checked
by nobody and passes forever.  That is the FUTURE failure mode, and it is closed by the
COMPLEMENT check: scan the covered sections for anything shaped like a quoted result and
require each one to be claimed by an emitted value, with an explicit allowlist for the few
that legitimately are not run outputs.  Same safe-by-default property as `run_identity`: a
new number is UNCHECKED UNTIL SOMEONE CONSCIOUSLY EXEMPTS IT, rather than checked only if
someone remembers to emit it.

COVERAGE IS DECLARED, NOT ASSUMED.  The complement runs over `COVERED` only -- the sections
whose numbers this emitter produces.  A figure added to a section outside that list is not
checked, and widening the guarantee means emitting that section's numbers and adding it here.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EMITTED = ROOT / "doc" / "generated" / "ess_exponent_control.numbers"
TABLES = ROOT / "doc" / "generated" / "ess_exponent_control.tables"
WORKLOG = ROOT / "doc" / "WORKLOG.md"
ENTRY = "cont.191"
# Sections whose numbers scripts/ess_exponent_control.py emits.  ORDERED BY WHAT GETS QUOTED,
# not by where drift happened to be caught first: a check is worth what it protects, and the
# figures that leave this document are the headline and the summary tables, not the section
# where the first stale literal turned up.  Section 3 was the last one out, on the grounds that
# its numbers came from a MANUAL scan across grid tops rather than from one run -- which was a
# description of the script, not a property of the question.  The previous entry's section 6
# said to move the scan into the run; doing so found three of that table's four columns had
# drifted, in the one section the guard did not cover.
COVERED = ("headline", "1", "2", "3", "4", "4a", "4b", "4c", "4e")

# Result-shaped: a ratio, a bracketed interval, a bounded/approximate figure, a 3dp decimal.
CANDIDATE = re.compile(
    r"\d+\.\d+x|\[\s*\d+\.\d+\s*,\s*\d+\.\d+\s*\]"
    r"|[<~]\s*\d+(?:\.\d+)?(?:-\d+)?|(?<![\d.])\d+\.\d{3}(?![\dx])")
NUM = re.compile(r"-?\d+(?:\.\d+)?")

# NOT run outputs.  Each is exempt for a stated reason; an unexplained entry here would
# reintroduce exactly the hole this check closes.
ALLOWED = {
    "0": "the P value the entry explicitly REFUSES to quote (0.000), shown as a counterexample",
    "1": "1.0x, the no-advantage point the extrapolation runs to -- a constant of the question",
    "1.6": "'1.6x' as a rounded MAGNITUDE in a sentence about not quoting it to two figures",
    "6": "~6% endpoint noise, an order-of-magnitude statement rather than a reported figure",
    "200": "the jackknife block count -- an INPUT to the run, not a result of it",
}


def _canon(tok):
    return tok.rstrip("0").rstrip(".") if "." in tok else tok


def _tokens(text):
    return {_canon(t) for t in NUM.findall(text)}


def _emitted():
    if not EMITTED.exists():
        pytest.skip(f"{EMITTED.relative_to(ROOT)} absent -- run scripts/ess_exponent_control.py")
    out = {}
    for line in EMITTED.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _entry_text():
    """The cont.191 entry only -- an entry is the unit a re-run regenerates."""
    lines = WORKLOG.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("## ") and ENTRY in ln), None)
    assert start is not None, f"{ENTRY} entry not found in doc/WORKLOG.md"
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def _covered_text():
    lines = WORKLOG.read_text().splitlines()
    out = []
    for tag in COVERED:
        if tag == "headline":
            st = next((i for i, ln in enumerate(lines)
                       if ln.startswith("## ") and ENTRY in ln), None)
        else:
            st = next((i for i, ln in enumerate(lines) if ln.startswith(f"### {tag} ")), None)
        assert st is not None, f"section {tag} not found -- COVERED is stale"
        en = next((j for j in range(st + 1, len(lines)) if lines[j].startswith("### ")), len(lines))
        out.append("\n".join(lines[st:en]))
    return "\n".join(out)


def _emitted_tokens():
    toks = set()
    for path in (EMITTED, TABLES):
        if not path.exists():
            pytest.skip(f"{path.relative_to(ROOT)} absent -- run scripts/ess_exponent_control.py")
        for line in path.read_text().splitlines():
            if line.startswith(("#", "<<<", ">>>")):
                continue
            toks |= _tokens(line)
    return toks


def test_every_emitted_number_still_appears_in_the_worklog_entry():
    """The tripwire itself: re-running must not leave the prose behind."""
    quoted, text = _emitted(), _entry_text()
    assert quoted, "the emitter produced no values -- the check would pass vacuously"
    missing = []
    for key, val in sorted(quoted.items()):
        # not flanked by digits or a decimal point, so "17" does not match inside "1.17"
        pat = r"(?<![\d.])" + re.escape(val).replace(r"\ ", r"\s+") + r"(?![\d])"
        if not re.search(pat, text):
            missing.append(f"{key} = {val}")
    assert not missing, (
        "doc/WORKLOG.md " + ENTRY + " no longer quotes these emitted values -- the run moved "
        "and the prose did not:\n  " + "\n  ".join(missing)
    )


def test_the_emitted_file_is_marked_generated_so_nobody_hand_edits_it():
    if not EMITTED.exists():
        pytest.skip("emitted file absent")
    assert "GENERATED" in EMITTED.read_text().splitlines()[0]


def _blocks():
    if not TABLES.exists():
        pytest.skip(f"{TABLES.relative_to(ROOT)} absent -- run scripts/ess_exponent_control.py")
    return dict(re.findall(r"<<<(\w+)\n(.*?)\n>>>\n", TABLES.read_text(), re.S))


def test_every_emitted_TABLE_ROW_matches_the_worklog_entry_field_by_field():
    """The strong check: whole ROWS, not loose digits -- what the value search missed.

    Compared as PARSED NUMERIC FIELDS rather than raw strings.  Byte-comparison would fail on
    a column width or a rounding change -- a failure that is not drift, and the kind that
    teaches people to loosen a check.  This one was built specifically so it cannot be
    loosened, so formatting is free while the property that matters is kept: a coincidence
    would have to reproduce every field of the row, in order.
    """
    blocks, lines = _blocks(), _entry_text().splitlines()
    assert blocks, "the emitter produced no tables -- the check would pass vacuously"
    have = [NUM.findall(ln) for ln in lines]
    missing = []
    for name, block in blocks.items():
        for row in block.splitlines():
            want = NUM.findall(row)
            if want and want not in have:
                missing.append(f"{name}: {row.strip()}")
    assert not missing, (
        "doc/WORKLOG.md " + ENTRY + " no longer carries these emitted rows -- the run moved "
        "and the prose did not:\n  " + "\n  ".join(missing)
    )


def test_every_result_shaped_figure_in_the_covered_sections_IS_EMITTED():
    """The complement, and the only check that pins the FUTURE rather than the present.

    The other two ask whether what the run emits still appears.  Neither notices a figure
    typed into the prose by hand that no emitter covers -- unchecked, and passing forever.
    This one inverts it: everything appearing must be emitted, or consciously exempted.
    """
    emitted, text = _emitted_tokens(), _covered_text()
    unclaimed = {}
    for cand in CANDIDATE.findall(text):
        for tok in _tokens(cand):
            if tok not in emitted and tok not in ALLOWED:
                unclaimed.setdefault(tok, cand)
    assert not unclaimed, (
        "result-shaped figures in " + ENTRY + " sections " + "/".join(COVERED) + " that no "
        "emitted value claims -- either emit them from the run, or add them to ALLOWED with "
        "a reason:\n  " + "\n  ".join(f"{k}  (in {v!r})" for k, v in sorted(unclaimed.items()))
    )


def test_the_allowlist_is_small_and_every_entry_carries_a_reason():
    """An allowlist that grows silently is the hole reopening under a different name."""
    assert len(ALLOWED) <= 12, "allowlist is growing -- emit these numbers instead of exempting"
    for tok, why in ALLOWED.items():
        assert len(why) > 25, f"ALLOWED[{tok}] needs a real reason, not a placeholder"


def test_the_ROW_check_catches_the_drift_the_value_check_MISSED():
    """Guard the guard, on the case that actually got through.

    A one-cell edit to a Tier-5 median is invisible to a whole-entry value search, because
    the replacement digits occur elsewhere legitimately -- "1.49x" is both a stale median and
    a live grid-scan figure.  It is not invisible to a row comparison.  The fixture is built
    by locating the row through its PARSED FIELDS, so this test does not itself depend on
    column widths -- the brittleness the row check was reformulated to avoid.
    """
    text, blocks = _entry_text(), _blocks()
    row = next(r for r in blocks["tier5"].splitlines() if "1.63x" in r)
    want = NUM.findall(row)
    lines = text.splitlines()
    k = next((j for j, ln in enumerate(lines) if NUM.findall(ln) == want), None)
    assert k is not None, "the fixture row is not in the entry; the row check would have failed"
    lines[k] = lines[k].replace("1.63x", "1.49x", 1)
    drifted = "\n".join(lines)
    assert re.search(r"(?<![\d.])1\.49x(?![\d])", drifted), \
        "precondition: the stale value must occur, or the weak check would have caught it"
    assert want not in [NUM.findall(ln) for ln in lines], "the row check missed a changed cell"
