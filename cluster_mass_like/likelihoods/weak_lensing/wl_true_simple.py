import numpy as np


def ln_prob_mwl_mtrue(logm_wl, logm_true, parameters):
    """Log of the probability of having a WL mass given a true mass"""
    A, B, C = parameters
    sig = C
    return -0.5 * ((logm_wl - (A + B * logm_true)) / sig) ** 2 - 0.5 * np.log(
        2 * np.pi * sig
    )
