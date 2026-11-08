"""Selection-aware matched self/neighbor response diagnostics.

A Rademacher probe shears every generated galaxy along one fixed sky axis.
Multiplying each target's influence difference by its own probe sign estimates
its diagonal image-response contribution. Independent signs are essential;
exactly balanced signs introduce off-diagonal bias. This module neither
simulates images nor fits, rescales, or promotes a calibration model.
"""
import numpy as np


def contributions(shape, selected):
    """Zero excluded rows without multiplying an invalid shape by zero."""
    shape, selected = np.asarray(shape, dtype=float), np.asarray(selected, dtype=float)
    if (shape.ndim != 2 or shape.shape[1] != 2 or selected.shape != (len(shape),)
            or not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1))):
        raise ValueError('aligned two-output shapes and finite selection probabilities required')
    present = selected > 0
    if not np.isfinite(shape[present]).all():
        raise ValueError('selected shapes must be finite')
    result = np.zeros_like(shape)
    result[present] = selected[present, None]*shape[present]
    return result, selected


def baseline_reference(shape, selected):
    weighted, mass = contributions(shape, selected)
    denominator = mass.sum()
    if denominator <= 0:
        raise ValueError('positive baseline selected mass required')
    return weighted.sum(0)/denominator, float(denominator)


def influence_difference(plus, minus, selected_plus, selected_minus, *, mean_zero, mass_zero, h):
    """Each row is an influence derivative using a declared common baseline.

Shapes may be invalid only where that leg's own selection is exactly zero.
Use hard U*C for the primary measurement cut. Soft cuts are separate diagnostics.
"""
    pos, wp = contributions(plus, selected_plus)
    neg, wm = contributions(minus, selected_minus)
    mean_zero = np.asarray(mean_zero, dtype=float)
    if (pos.shape != neg.shape or mean_zero.shape != (2,) or not np.isfinite(mean_zero).all()
            or not np.isfinite([mass_zero, h]).all() or mass_zero <= 0 or not 0 < h < 1):
        raise ValueError('matched legs, finite baseline, positive mass and shear required')
    return (pos-neg-(wp-wm)[:, None]*mean_zero)/(2*h*mass_zero)


def probe_response(plus, minus, selected_plus, selected_minus, *, parent_signs, mean_zero, mass_zero, h):
    rows = influence_difference(plus, minus, selected_plus, selected_minus,
                                mean_zero=mean_zero, mass_zero=mass_zero, h=h)
    signs = np.asarray(parent_signs)
    if signs.shape != (len(rows),) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('one Rademacher sign per aligned truth parent required')
    return (rows*signs[:, None]).sum(0)


def coherent_response(plus, minus, selected_plus, selected_minus, *, mean_zero, mass_zero, h):
    """Return both the baseline-influence diagnostic and exact endpoint ratio."""
    rows = influence_difference(plus, minus, selected_plus, selected_minus,
                                mean_zero=mean_zero, mass_zero=mass_zero, h=h)
    mu_plus, _ = baseline_reference(plus, selected_plus)
    mu_minus, _ = baseline_reference(minus, selected_minus)
    return {'baseline_influence': rows.sum(0), 'endpoint_ratio': (mu_plus-mu_minus)/(2*h)}


def summarize_probes(probes, coherent):
    """Probe SE is conditional on this one scene, never a scene-based bias SE."""
    probes, coherent = np.asarray(probes, dtype=float), np.asarray(coherent, dtype=float)
    if (probes.ndim != 2 or probes.shape[1] != 2 or len(probes) < 2
            or coherent.shape != (2,) or not np.isfinite(probes).all() or not np.isfinite(coherent).all()):
        raise ValueError('at least two independent finite two-output probes required')
    self_response = probes.mean(0)
    return {'self_response': self_response, 'neighbor_response': coherent-self_response,
            'conditional_probe_se': probes.std(0, ddof=1)/np.sqrt(len(probes)),
            'probe_count': len(probes), 'calibration_claim': False}
