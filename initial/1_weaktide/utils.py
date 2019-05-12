import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt

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

def find_cs(I, eta, q0):
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)
    return np.cos(opt.newton(f, q0, fprime=fp))

def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

def get_crit(I):
    eta_c = get_etac(I)

    eta = eta_c - 1e-3 # small displacement since no CS4 at eta_c

    # search for CS4
    mu_4 = find_cs(I, eta, -np.pi / 2)
    return eta_c, mu_4

def get_mu4(I, s_c, s):
    eta = s_c / s
    return [find_cs(I, eta_i, -np.pi / 2) for eta_i in eta]

def dydt_avg(t, y):
    '''
    evolves dmu/dt, ds/dt forwards in time, precession-averaged
    '''
    mu, s = y
    return [
        (1 - mu**2) * (2 / s - mu),
        2 * mu - s * (1 + mu**2),
    ]

def get_crits(I, s_c):
    eta_c, mu_4 = get_crit(I)
    ret = solve_ivp(dydt_avg, [0, -2], [mu_4, s_c/eta_c], max_step=0.01)
    mu, s = ret.y
    return mu, s, get_mu4(I, s_c, s)

def get_upper_sc(I):
    eta_c, mu_4 = get_crit(I)
    ret = solve_ivp(dydt_avg, [0, 10], [0, 20], max_step=0.01,
                    dense_output=True)

    t, (mu, s), interp_sol = ret.t, ret.y, ret.sol
    # find s(t) where mu(t) = mu_4 at separatrix crossing
    idx_4 = np.where(mu > mu_4)[0][0]

    root_func = lambda t: interp_sol(t)[0] - mu_4
    t_exact = opt.brentq(root_func, t[idx_4 - 1], t[idx_4])
    return interp_sol(t_exact)[1] * eta_c
