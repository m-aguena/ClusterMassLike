import numpy as np
from scipy.optimize import curve_fit


def safe_fit(*args, **kwargs):
    try:
        return curve_fit(*args, **kwargs)
    except:
        return np.array([[None], [None]])
