"""Sharding the score pass, and the two ways merging the shards back goes wrong silently.

The expensive half of (5.3) is scoring the catalogue, so it is cut into disjoint row shards
that run side by side and are added afterwards.  That addition is EXACT -- the estimator needs
only `sum s`, `sum I` and a count, all additive, and the jackknife is defined over blocks, so
`k` shards of `B` blocks give a `k*B`-block jackknife over their union.

The reason this file exists is that the exactness is the easy part.  Every identity in
`score_inference` had tests while this pure bookkeeping had none, and it is where the two worst
failures of the §5B line came from (`WORKLOG.md` cont.178, cont.179):

  * a cache-key regression that made jobs `COMPLETE` in 53 s having printed nothing -- a
    mismatch that should have been loud instead skipped every result, which reads as success;
  * the double-count guard, without which merging a shard twice shrinks the error bar by
    `sqrt(2)` with no new information behind it.

Both are failures of REFUSAL, not of arithmetic, so the tests below spend most of their effort
on merges that must raise.  The payload is deliberately trivial (`blocked_sums` output on
random `s`/`I`); what is under test is the bookkeeping around it.
"""

import json

import numpy as np
import pytest

from sbsi.score_inference import (
    ScoreCacheMismatch, blocked_sums, jackknife_blocks, merge_block_sum_caches,
)

NBLOCK = 20
BASE_KEY = dict(cut=0.6, closure_g=0.05, rows=1000, ring="rot90", shape_reps=2,
                jk_blocks=NBLOCK, grid_n=61, uncut=1, row_shard=0, row_shards=3)


def make_shard(tmp_path, name, key, n=1000, seed=0, uncut=True):
    """Write one cache exactly as `eval_score_select.py --save-scores` does."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, 2))
    info = np.tile(np.eye(2)[None], (n, 1, 1)) * rng.uniform(0.5, 1.5, (n, 1, 1))
    block = np.arange(n) % NBLOCK
    cnt, ns, ni = blocked_sums(s, info, block, NBLOCK)
    path = str(tmp_path / f"{name}.npz")
    extra = dict(cnt_u=cnt, ns_u=ns, ni_u=ni) if uncut else {}
    np.savez(path, key=json.dumps(key, sort_keys=True), n_keep=n // 2, n_tot=n,
             cnt_k=cnt, ns_k=ns, ni_k=ni, **extra)
    return path, (cnt, ns, ni)


def key_for(shard, **over):
    k = dict(BASE_KEY, row_shard=shard)
    k.update(over)
    return k


# -- the exactness claim ---------------------------------------------------------------

def test_merging_shards_concatenates_blocks_and_adds_counts(tmp_path):
    """`k` shards of `B` blocks -> one `k*B`-block jackknife, sums added."""
    paths, parts = [], []
    for i in range(3):
        p, blk = make_shard(tmp_path, f"s{i}", key_for(i), seed=i)
        paths.append(p)
        parts.append(blk)
    blk_k, blk_u, n_keep, n_tot, shards = merge_block_sum_caches(
        paths, key_for(0), want_uncut=True)

    assert shards == [0, 1, 2]
    assert n_keep == 3 * 500 and n_tot == 3 * 1000
    assert len(blk_k[0]) == 3 * NBLOCK
    for axis in range(3):
        np.testing.assert_allclose(
            blk_k[axis], np.concatenate([p[axis] for p in parts], axis=0))
    # the uncut control must travel with the kept sums, block for block, or the paired
    # cut-minus-uncut difference silently compares different galaxies
    for axis in range(3):
        np.testing.assert_allclose(blk_u[axis], blk_k[axis])


def test_merged_estimate_equals_scoring_the_union_in_one_pass(tmp_path):
    """The whole point: sharding must not change the answer, only the wall clock."""
    rng = np.random.default_rng(7)
    n = 900
    s = rng.standard_normal((n, 2))
    info = np.tile(np.eye(2)[None], (n, 1, 1)) * rng.uniform(0.5, 1.5, (n, 1, 1))

    # one pass over everything, blocked as 3 x NBLOCK exactly as the merge would be
    block_all = np.concatenate([np.arange(n // 3) % NBLOCK + i * NBLOCK for i in range(3)])
    whole = jackknife_blocks(*blocked_sums(s, info, block_all, 3 * NBLOCK))

    paths = []
    for i in range(3):
        lo, hi = i * (n // 3), (i + 1) * (n // 3)
        cnt, ns, ni = blocked_sums(s[lo:hi], info[lo:hi],
                                   np.arange(hi - lo) % NBLOCK, NBLOCK)
        p = str(tmp_path / f"part{i}.npz")
        np.savez(p, key=json.dumps(key_for(i), sort_keys=True), n_keep=1, n_tot=hi - lo,
                 cnt_k=cnt, ns_k=ns, ni_k=ni)
        paths.append(p)
    blk_k, _, _, _, _ = merge_block_sum_caches(paths, key_for(0), want_uncut=False)
    merged = jackknife_blocks(*blk_k)

    np.testing.assert_allclose(merged[0], whole[0], rtol=1e-12)   # ghat
    np.testing.assert_allclose(merged[1], whole[1], rtol=1e-12)   # and its error bar


# -- the refusals, which is what actually broke -----------------------------------------

def test_repeating_a_shard_is_refused(tmp_path):
    """Double-counting is invisible in the sums; only the registry catches it."""
    p0, _ = make_shard(tmp_path, "a", key_for(0))
    p0_again, _ = make_shard(tmp_path, "a_copy", key_for(0), seed=99)
    with pytest.raises(ScoreCacheMismatch, match="double-count"):
        merge_block_sum_caches([p0, p0_again], key_for(0), want_uncut=True)


@pytest.mark.parametrize("field,value", [
    ("closure_g", 0.10),        # the g-scan bug: same path, different injected shear
    ("cut", 0.4),
    ("grid_n", 101),
    ("rows", 2000),
    ("ring", "none"),
    ("row_shards", 6),          # a whole-catalogue cache is NOT a shard of a 3-way run
])
def test_incompatible_settings_are_refused(tmp_path, field, value):
    """Anything that shapes a score must match, or one run's galaxies meet another's Pi."""
    p, _ = make_shard(tmp_path, "x", key_for(0, **{field: value}))
    with pytest.raises(ScoreCacheMismatch, match="different settings"):
        merge_block_sum_caches([p], key_for(0), want_uncut=True)


def test_row_shard_is_the_only_exempt_key(tmp_path):
    """Shards differ in `row_shard` BY CONSTRUCTION, so that one key must not be checked."""
    p, _ = make_shard(tmp_path, "s2", key_for(2))
    blk_k, _, _, _, shards = merge_block_sum_caches([p], key_for(0), want_uncut=True)
    assert shards == [2] and len(blk_k[0]) == NBLOCK


def test_pre_sharding_caches_read_as_whole_catalogue_passes(tmp_path):
    """The regression that cost two jobs: caches predating sharding carry neither key.

    Adding `row_shard`/`row_shards` to the key made every older cache mismatch on
    `{'row_shards': (1, None)}`.  Jobs then skipped every rung and exited 0 -- a silent
    no-op that reads as success.  A missing key must read as the UNSHARDED value.
    """
    old = {k: v for k, v in BASE_KEY.items() if not k.startswith("row_shard")}
    p, _ = make_shard(tmp_path, "legacy", old)

    blk_k, _, _, _, shards = merge_block_sum_caches(
        [p], key_for(0, row_shards=1), want_uncut=True)
    assert shards == [0] and len(blk_k[0]) == NBLOCK

    # ...but it is still not a shard of a MULTI-shard run
    with pytest.raises(ScoreCacheMismatch, match="different settings"):
        merge_block_sum_caches([p], key_for(0, row_shards=3), want_uncut=True)


def test_missing_keys_other_than_row_shard_still_mismatch(tmp_path):
    """The pre-sharding fallback must not become a general 'absent is fine' rule."""
    short = {k: v for k, v in BASE_KEY.items() if k != "grid_n"}
    p, _ = make_shard(tmp_path, "nogrid", short)
    with pytest.raises(ScoreCacheMismatch, match="different settings"):
        merge_block_sum_caches([p], key_for(0), want_uncut=True)


def test_uncut_control_mismatch_is_caught_by_the_key_not_a_keyerror(tmp_path):
    """A cache saved without the control must fail on the KEY, before the array lookup.

    `want_uncut` reaches for `cnt_u`, which a control-free cache does not have.  That is
    safe only because `uncut` is part of the key and is checked first; if it were ever
    dropped from the key this would surface as a bare `KeyError` instead.
    """
    p, _ = make_shard(tmp_path, "nocontrol", key_for(0, uncut=0), uncut=False)
    with pytest.raises(ScoreCacheMismatch, match="different settings"):
        merge_block_sum_caches([p], key_for(0, uncut=1), want_uncut=True)


def test_empty_path_list_is_refused(tmp_path):
    with pytest.raises(ScoreCacheMismatch, match="no cache paths"):
        merge_block_sum_caches(["", None], key_for(0))
