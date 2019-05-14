import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt

def stringify(*args):
    return 'x'.join(['%.1f' % arg for arg in args]).replace('.', '_')

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

def dmu_ds(s, y):
    '''
    solves for mu(s) without ever introducing t
    '''
    mu = y[0]
    return [((1 - mu**2) * (2 / s - mu)) / (2 * mu - s * (1 + mu**2))]

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
    '''
    gets mu4 for a list of spins s, returning -1 if no CS4 for given s
    '''
    eta = s_c / s
    eta_c = get_etac(I)
    mu4 = []
    for eta_i in eta:
        if eta_i > eta_c:
            mu4.append(-1)
        else:
            mu4.append(find_cs(I, eta_i, -np.pi / 2))
    return np.array(mu4)

def get_inf_avg_sol(smax=10):
    '''
    solve averaged equations for IC mu = 0, s >> 1
    '''
    ret = solve_ivp(dmu_ds, [smax, 1.001], [0], max_step=0.1, dense_output=True)
    s, [mu], interp_sol = ret.t, ret.y, ret.sol
    return mu, s, interp_sol
