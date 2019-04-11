import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp
import time

def get_phis(thetas):
    return [np.pi if q > 0 else 0 for q in thetas]

def roots(eta, I):
    ''' returns theta roots from EOM '''
    eta_c = (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

    # function to minimize and derivatives
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)

    if eta < eta_c:
        roots = []
        inits = [0, np.pi / 2, -np.pi, -np.pi / 2]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots), get_phis(roots)

    else:
        roots = []
        inits = [np.pi / 2 - I, -np.pi + I]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots), get_phis(roots)

def get_dydt(I, eta, tide):
    ''' get dy/dt for params '''
    def dydt(t, s):
        x, y, z = s
        return [
            y * z - eta * y * np.cos(I) - tide * z * x,
            -x * z + eta * (x * np.cos(I) - z * np.sin(I)) - tide * z * y,
            eta * y * np.sin(I) + tide * (1 - z**2),
        ]
    return dydt

def get_jac(I, eta, tide):
    ''' get jacobian d(dy/dt)_i/dy_j for params '''
    def jac(t, s):
        x, y, z = s
        return [
            [-tide * z, z - eta * np.cos(I), y - tide * x],
            [-z + eta * np.cos(I), -tide * z, -x - eta * np.sin(I) - tide * y],
            [0, eta * np.sin(I), -2 * tide * z],
        ]
    return jac

def solve_ic(I, eta, tide, y0, tf, method='RK45', **kwargs):
    '''
    wraps solve_ivp and returns sim time
    '''
    time_i = time.time()
    dydt = get_dydt(I, eta, tide)
    jac = get_jac(I, eta, tide)
    if 'RK' in method:
        ret = solve_ivp(dydt, [0, tf], y0, method=method, **kwargs)
    else:
        ret = solve_ivp(dydt, [0, tf], y0, method=method, jac=jac, **kwargs)
    return time.time() - time_i, ret.t, ret.y
