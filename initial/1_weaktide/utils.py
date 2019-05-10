import numpy as np
from scipy.integrate import solve_ivp

def to_cart(q, phi):
    return [
        np.sin(q) * np.cos(phi),
        np.sin(q) * np.sin(phi),
        np.cos(q),
    ]

def to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    q = np.arccos(z / r)
    phi = (np.arctan2(y / np.sin(q), x / np.sin(q)) + np.pi)\
        % (2 * np.pi)
    return q, phi

def get_dydt(I, s_c, eps):
    '''
    in units where Omega_1 = 1
    '''
    def dydt(t, v):
        x, y, z, s = v
        tide = eps * 2 / s * (1 - s * z / 2)
        rat = s / s_c
        return [
            rat * y * z - y * np.cos(I) - tide * z * x,
            -rat * x * z + (x * np.cos(I) - z * np.sin(I)) - tide * z * y,
            y * np.sin(I) + tide * (1 - z**2),
            2 * eps * (z - s * (1 + z**2) / 2),
        ]
    return dydt

def solve_ic(I, s_c, eps, y0, tf, method='RK45', rtol=1e-6, **kwargs):
    '''
    wraps solve_ivp and returns sim time
    '''
    dydt = get_dydt(I, s_c, eps)
    ret = solve_ivp(dydt, [0, tf], y0, rtol=rtol, method=method, **kwargs)
    return ret.t, ret.y[0:3, :], ret.y[3, :]
