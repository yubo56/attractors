import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from scipy.integrate import solve_ivp
from utils import roots

def solve_ic(eta, I, q0, phi0, tide=0):
    '''
    solves ivp at given (eta, I, tide) for IC (phi0, q0)
    '''
    def dydt(t, y):
        q, phi = y
        return [
            -eta * np.sin(I) * np.sin(q) * np.sin(phi),
            np.cos(q) - eta * (
                np.cos(I) +
                np.sin(I) * np.cos(phi) / np.tan(q))
        ]
    tf = 150
    ret = solve_ivp(dydt, [0, tf], [q0, phi0])
    return ret.t, ret.y

def solve_cart(eta, I, x0, y0, z0, tide=0):
    '''
    solves IVP at given (eta, I, tide) for cartesian IC (x0, y0, z0)
    '''
    def dydt(t, vec):
        x, y, z = vec
        r = np.sqrt(x**2 + y**2 + z**2)
        q = np.arccos(z / r)
        phi = -np.arctan(y / x) * np.sign(y)
        return [
            -np.sin(q) * np.sin(phi) * np.cos(q)
                + eta * np.sin(q) * np.sin(phi) * np.cos(I),
            np.cos(q) * np.sin(q) * np.cos(phi)
                - eta * (
                    np.sin(q) * np.cos(phi) * np.cos(I)
                    + np.cos(q) * np.sin(I)),
            -eta * np.sin(q) * np.sin(phi) * np.sin(I),
        ]
    tf = 100
    ret = solve_ivp(dydt, [0, tf], [x0, y0, z0])
    return ret.t, ret.y

if __name__ == '__main__':
    eta = 0.1
    I = np.radians(20)
    qs, phis = roots(eta, I)

    pert = 0.08 # perturbation strength

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

    for q0, phi0, ax in zip(qs, phis, [ax1, ax2, ax3, ax4]):
        q = q0 + pert
        phi = phi0 + pert
        x, y, z = [
            -np.sin(q) * np.cos(phi),
            -np.sin(q) * np.sin(phi),
            np.cos(q)]
        t, sol = solve_cart(eta, I, x, y, z)
        x, y, z = sol
        r = np.sqrt(x**2 + y**2 + z**2)
        q = np.arccos(z / r)
        phi = np.arctan(y / x)

        ax.plot(phi0 % (2 * np.pi), np.cos(q0), 'ro', markersize=4)
        ax.plot(phi % (2 * np.pi), np.cos(q), 'bo', markersize=1)
        ax.set_title('Init: (%.3f, %.3f)' % (phi0, np.cos(q0)), fontsize=8)
        ax.set_xticks([0, np.pi, 2 * np.pi])
    plt.suptitle(r'(I, $\eta$)=($%d^\circ$, %.3f)' % (np.degrees(I), eta),
                 fontsize=10)
    ax1.set_ylabel(r'$\cos \theta$')
    ax3.set_xlabel(r'$\phi$')
    ax3.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax3.set_ylabel(r'$\cos \theta$')
    ax4.set_xlabel(r'$\phi$')
    plt.savefig('2evolution.png', dpi=400)
