import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

def to_cart(q, phi):
    return [
        np.sin(q) * np.cos(phi),
        np.sin(q) * np.sin(phi),
        np.cos(q),
    ]

def to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    q = np.arccos(z / r)
    phi = (np.arctan2(y / np.sin(q), x / np.sin(q)) + 2 * np.pi)\
        % (2 * np.pi)
    return q, phi

def get_phi(q, phi=0):
    return ((phi + np.pi if q > 0 else phi) + 2 * np.pi) % (2 * np.pi)

def get_phis(qs, phis):
    return np.array([get_phi(th, phi=f) for th, f in zip(qs, phis)])

def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

def roots(I, eta):
    ''' returns theta roots from EOM '''
    eta_c = get_etac(I)

    # function to minimize and derivatives
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)

    if eta < eta_c:
        roots = []
        inits = [0, np.pi / 2, -np.pi, -np.pi / 2]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots), np.zeros_like(roots)

    else:
        roots = []
        inits = [np.pi / 2 - I, -np.pi + I]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots), np.zeros_like(roots)

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

def solve_ic(I, eta, tide, y0, tf, method='RK45', rtol=1e-6, **kwargs):
    '''
    wraps solve_ivp and returns sim time
    '''
    time_i = time.time()
    dydt = get_dydt(I, eta, tide)
    jac = get_jac(I, eta, tide)
    if 'RK' in method:
        ret = solve_ivp(dydt, [0, tf], y0,
                        rtol=rtol, method=method, **kwargs)
    else:
        ret = solve_ivp(dydt, [0, tf], y0,
                        rtol=rtol, method=method, jac=jac, **kwargs)
    return time.time() - time_i, ret.t, ret.y

def get_four_subplots():
    ''' keep using four subplots w/ same settings '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    axs = [ax1, ax2, ax3, ax4]
    for ax in axs:
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([-1, 1])
    f.subplots_adjust(wspace=0.07)
    ax1.set_ylabel(r'$\cos \theta$')
    ax3.set_xlabel(r'$\phi$')
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax3.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax3.set_ylabel(r'$\cos \theta$')
    ax4.set_xlabel(r'$\phi$')

    return f, axs

def plot_point(ax, q, *args, **kwargs):
    ''' plot Cassini state including wrap around logic '''
    phi = get_phi(q)
    phi_arr = [phi] if abs(phi % (2 * np.pi)) > 0.5 else [phi, phi + 2 * np.pi]
    for phi_plot in phi_arr:
        ax.plot(phi_plot, np.cos(q), *args, **kwargs)

def H(I, eta, x, phi):
    return 0.5 * x**2 - eta * (
        x * np.cos(I) -
        np.sqrt(1 - x**2) * np.sin(I) * np.cos(phi))

def get_grids(N=50):
    _phi = np.linspace(0, 2 * np.pi, N)
    _x = np.linspace(-1, 1, N)
    phi_grid = np.outer(_phi, np.ones_like(_x))
    x_grid = np.outer(np.ones_like(_phi), _x)
    return x_grid, phi_grid

def is_below(I, eta, q, phi):
    '''
    'below separatrix' defined here as x < x[3] (Cassini state 4) &
    H(q, phi) > H[3] (energy of Cassini state 4)
    '''
    qs, phis = roots(I, eta)
    x3, phi3 = np.cos(qs[3]), phis[3]
    x = np.cos(q)

    return np.logical_and(x < x3 , H(I, eta, x, phi) > H(I, eta, x3, phi3))
