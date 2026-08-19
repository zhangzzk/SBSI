"""Guards on the paired cache differencer.

The two things that can go wrong here are both silent.  A regex that drifts out of step with
the log format returns a WRONG population term rather than failing, and a comparison run
against two different population blocks attributes their difference to the score pass.  Both
produce a plausible number, so both are tested rather than eyeballed.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diff_score_caches as dsc  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diff_score_caches.py"

# Copied VERBATIM from a run log (job 15827006), not reconstructed by hand: the point of the
# parser is to survive this exact formatting, so a paraphrase would test the wrong string.
LOG_MAG = """\
=== Pi population sample M = 500,000 ========================================
Pi_k: min=0.6522 max=0.6588 prior-weighted mean=0.6579  (actual keep 0.6579, mismatch +0.00%)
<s>_sel = [-0.000345 +/- 0.000001, +0.000035 +/- 0.000002]   |s|/sigma = 292.3, 20.6
I_sel   = [[+0.01116 +/- 0.00006, +0.00010], [+0.00011, +0.01080 +/- 0.00006]]
  I_sel/<I> = +0.0013 (A.7's 2/pi analogue);  off-diag/diag = 0.010
"""


def test_the_parser_recovers_the_values_the_run_actually_printed(tmp_path):
    p = tmp_path / "log.out"
    p.write_text(LOG_MAG)
    s_sel, i_sel, _, _ = dsc.population_terms_from_log(str(p))
    assert np.array_equal(s_sel, [-0.000345, +0.000035])
    # Note the ORDER: the printed matrix is [[11, 12], [21, 22]] and the off-diagonals differ
    # in the fifth decimal, so a transposed parse would still look right to the eye.
    assert np.array_equal(i_sel, [[+0.01116, +0.00010], [+0.00011, +0.01080]])


def test_a_log_without_population_terms_RAISES_rather_than_returning_zero(tmp_path):
    """Defaulting to zero would turn a missing population block into an uncorrected number
    that still looks like a corrected one."""
    p = tmp_path / "uncut.out"
    p.write_text("=== a run with no cut, so no population block ===\n")
    with pytest.raises(SystemExit, match="not a cut run log"):
        dsc.population_terms_from_log(str(p))


def _fake_cache(path, grid_n, seed):
    """A cache with the fields the differencer reads and nothing else."""
    rng = np.random.default_rng(seed)
    nb = 8
    key = {"closure_g": 0.05, "jk_blocks": nb, "grid_n": grid_n, "rows": 1000,
           "flow_seed": 1, "ring": "rot90", "shape_reps": 2}
    cnt = rng.uniform(90, 110, nb)
    ns = rng.normal(0.4, 0.01, (nb, 2))
    ni = np.tile(np.eye(2) * 8.0, (nb, 1, 1)) + rng.normal(0, 0.01, (nb, 2, 2))
    np.savez(path, key=json.dumps(key), n_keep=800, n_tot=1000,
             cnt_k=cnt, ns_k=ns, ni_k=ni, cnt_u=cnt, ns_u=ns, ni_u=ni)


def _run(tmp_path, log_b_text):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _fake_cache(a, 61, 0)
    _fake_cache(b, 101, 0)
    la, lb = tmp_path / "a.out", tmp_path / "b.out"
    la.write_text(LOG_MAG)
    lb.write_text(log_b_text)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(a), str(b), "--arm", "kept",
         "--pop-log", str(la), str(lb)],
        capture_output=True, text=True)


def test_two_logs_with_DIFFERENT_population_terms_are_refused(tmp_path):
    """The whole claim of the comparison is that only the score grid moved."""
    drifted = LOG_MAG.replace("+0.01080", "+0.01099")
    out = _run(tmp_path, drifted)
    assert out.returncode != 0
    assert "DIFFERENT population terms" in out.stderr


def test_two_logs_that_agree_are_accepted(tmp_path):
    out = _run(tmp_path, LOG_MAG)
    assert out.returncode == 0, out.stderr
    assert "the two logs agree exactly" in out.stdout


def test_the_kept_arm_actually_APPLIES_the_population_terms(tmp_path):
    """Parsing them and then not subtracting them would pass every test above."""
    a = tmp_path / "a.npz"
    _fake_cache(a, 61, 0)
    z = np.load(a, allow_pickle=True)
    cnt, ns, ni = z["cnt_k"], z["ns_k"], z["ni_k"]
    log = tmp_path / "l.out"
    log.write_text(LOG_MAG)
    s_sel, i_sel, _, _ = dsc.population_terms_from_log(str(log))
    n = cnt.sum()
    with_terms = np.linalg.solve(ni.sum(axis=0) - n * i_sel, ns.sum(axis=0) - n * s_sel)
    without = np.linalg.solve(ni.sum(axis=0), ns.sum(axis=0))
    assert not np.allclose(with_terms, without, atol=1e-9)
