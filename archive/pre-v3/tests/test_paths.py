"""The slash contract on the shared data paths.

~50 call sites build catalogue paths by CONCATENATION (`CAT + "name.feather"`), so a
trailing slash silently added or removed here repoints every one of them -- to a path that
either does not exist or, worse, exists and is the wrong tree. That is exactly why these
constants could not simply be merged when they lived in 25 separate files in two spellings.
"""

from sbs_shear.paths import (
    CATALOGUES,
    CONST_SIM_BASE,
    CONST_SIM_DIR,
    SIM_BASE,
    catalogue,
)


def test_concatenation_roots_keep_their_trailing_slash():
    assert CATALOGUES.endswith("/")
    assert CONST_SIM_BASE.endswith("/")


def test_fstring_root_has_no_trailing_slash():
    # `f"{CONST_SIM_DIR}/case0_0.02/..."` would double the separator if this ever gained one.
    assert not CONST_SIM_DIR.endswith("/")


def test_the_two_constant_tree_spellings_are_the_same_location():
    assert CONST_SIM_DIR + "/" == CONST_SIM_BASE


def test_catalogue_joiner_matches_plain_concatenation():
    name = "det_meas_ngmix_g0.0_train.feather"
    assert catalogue(name) == CATALOGUES + name
    assert "//" not in catalogue(name)


def test_halfshear_and_constant_trees_are_distinct():
    # Substituting one for the other silently changes the population (CONVENTIONS.md).
    assert SIM_BASE.rstrip("/") != CONST_SIM_DIR
