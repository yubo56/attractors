import numpy as np
import scipy.optimize as opt

def get_phis(thetas):
    return [np.pi if q > 0 else 0 for q in thetas]

def roots(eta, I):
    ''' returns theta roots from EOM '''
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    try:
        roots = []
        # try to root find w/ 4 intervals
        args_4_arr = [
            (-np.pi, -np.pi / 2),
            (-np.pi/2, -np.pi / 4),
            (-np.pi / 4, 0),
            (0, np.pi),
        ]
        for args in args_4_arr:
            roots.append(opt.brentq(f, *args))
        return np.array(roots), get_phis(roots)

    # throws ValueError when no zero in brackets
    except ValueError:
        roots = []
        # if not, fallback to only 2
        args_2_arr = [
            (-np.pi, 0),
            (0, np.pi),
        ]
        for args in args_2_arr:
            roots.append(opt.brentq(f, *args))
        return np.array(roots), get_phis(roots)
