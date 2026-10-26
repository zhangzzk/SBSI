"""Normalization-aware carry for the measured-bright BFGS continuation."""
import numpy as np


def carry(score, information, delta, old_gradient, old_hessian, gradient, hessian):
    """Carry numerator moments once from a fixed full pass, then replace log B."""
    score, information = np.asarray(score), np.asarray(information)
    if score.shape[-1:] != (2,) or information.shape != score.shape+(2,):
        raise ValueError('aligned full two-component moments required')
    numerator_information = information-np.asarray(old_hessian)
    result_score = (score+np.asarray(old_gradient)
                    -np.einsum('...ij,j->...i', numerator_information, delta)
                    -np.asarray(gradient))
    result_information = numerator_information+np.asarray(hessian)
    if not np.isfinite(result_score).all() or not np.isfinite(result_information).all():
        raise ValueError('nonfinite carried moments')
    return result_score, result_information
