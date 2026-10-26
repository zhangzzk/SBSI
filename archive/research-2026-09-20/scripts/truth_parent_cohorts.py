"""Explicit cohort availability for response audits with or without g0 data."""

import numpy as np


COHORTS = ("truth_parent", "usable_g0", "fixed_cut_g0")
G0_FIELDS = {"usable_g0": "g0_usable", "fixed_cut_g0": "g0_selected"}


def resolve_cohorts(manifest, requested=COHORTS):
    """Require an explicit restriction when an endpoint has no g0 catalogue."""
    requested = tuple(requested)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("cohorts must be nonempty and distinct")
    if not set(requested) <= set(COHORTS):
        raise ValueError("unknown response cohort")
    available = set(manifest.get("cohorts", COHORTS))
    if manifest.get("g0_diagnostics_available") is False:
        available -= set(G0_FIELDS)
    if not set(requested) <= available:
        raise ValueError("requested cohorts unavailable in parent manifest; select truth_parent explicitly when g0 is absent")
    return requested


def parent_cohort_masks(stored, requested):
    """Read only requested masks; never synthesize an observed-g0 population."""
    requested = resolve_cohorts({}, requested)
    count = len(stored["input_index"])
    masks = {}
    for name in requested:
        if name == "truth_parent":
            masks[name] = np.ones(count, dtype=bool)
            continue
        field = G0_FIELDS[name]
        if field not in stored:
            raise ValueError(f"requested cohort {name} requires parent field {field}")
        mask = np.asarray(stored[field])
        if mask.dtype != np.dtype(bool) or mask.shape != (count,):
            raise ValueError(f"parent field {field} must be a boolean mask aligned with identities")
        masks[name] = mask
    if "usable_g0" in masks and "fixed_cut_g0" in masks:
        if np.any(masks["fixed_cut_g0"] & ~masks["usable_g0"]):
            raise ValueError("fixed_cut_g0 must be a subset of usable_g0")
    return masks
