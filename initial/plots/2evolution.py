import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from scipy.integrate import solve_ivp
from utils import roots

T_F = 500

def dydt_pol(t, y):
    q, phi = y
    return [
        -eta * np.sin(I) * np.sin(phi),
        np.cos(q) - eta * (
            np.cos(I) +
            np.sin(I) * np.cos(phi) / np.tan(q)),
    ]

def jac_pol(t, y):
    q, phi = y
    return [
        [0, -eta * np.sin(I) * np.cos(phi)],
        [
            -np.cos(q) + eta * np.sin(I) * np.cos(phi) * np.sin(q)**2,
            eta * np.sin(I) * np.sin(phi) / np.tan(q),
        ],
    ]

def dydt_cart(t, s):
    x, y, z = s
    return [
        y * z - eta * y * np.cos(I),
        -x * z + eta * (x * np.cos(I) - z * np.sin(I)),
        eta * y * np.sin(I),
    ]

def jac_cart(t, s):
    x, y, z = s
    return [
        [0, z - eta * np.cos(I), y],
        [-z + eta * np.cos(I), 0, -x - eta * np.sin(I)],
        [0, eta * np.sin(I), 0],
    ]

def solve_ic(eta, I, dydt, jac, y0, tide=0, method='RK45'):
    '''
    wraps solve_ivp and returns sim time
    '''
    time_i = time.time()
    ret = solve_ivp(dydt, [0, T_F], y0, method=method)
    return time.time() - time_i, ret.t, ret.y

if __name__ == '__main__':
    eta = 0.1
    I = np.radians(20)
    qs, phis = roots(eta, I)

    pert = 0.08 # perturbation strength

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    f.subplots_adjust(wspace=0)

    for q0, phi0, ax in zip(qs, phis, [ax1, ax2, ax3, ax4]):
        q_i = q0 + pert
        phi_i = phi0 - pert

        s0 = [
            -np.sin(q_i) * np.cos(phi_i),
            -np.sin(q_i) * np.sin(phi_i),
            np.cos(q_i)]
        sim_time_cart, t_cart, sol = solve_ic(eta, I, dydt_cart, jac_cart, s0)
        x, y, z = sol
        r = np.sqrt(x**2 + y**2 + z**2)
        q = np.arccos(z / r) * np.sign(q_i)
        phi = (np.arctan2(-y / np.sin(q), -x / np.sin(q)) + 2 * np.pi)\
            % (2 * np.pi)

        ax.plot(phi0 % (2 * np.pi),
                np.cos(q0),
                'ro',
                markersize=4)
        ax.plot(phi % (2 * np.pi),
                np.cos(q),
                'bo',
                markersize=0.3)

        sim_time_pol, t_pol, sol = solve_ic(
            eta, I, dydt_pol, jac_pol, [q_i, phi_i], method='Radau')
        q_pol, phi_pol = sol
        ax.plot(phi_pol % (2 * np.pi),
                np.cos(q_pol),
                'go',
                markersize=0.3)
        print(sim_time_cart, sim_time_pol)
        ax.set_title(r'Init: $(\phi_0, \theta_0) = (%.3f, %.3f)$'
                     % (phi0, q0), fontsize=8)
        ax.set_xticks([0, np.pi, 2 * np.pi])

    plt.suptitle(r'(I, $\eta$)=($%d^\circ$, %.3f)' % (np.degrees(I), eta),
                 fontsize=10)
    ax1.set_ylabel(r'$\cos \theta$')
    ax3.set_xlabel(r'$\phi$')
    ax3.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax3.set_ylabel(r'$\cos \theta$')
    ax4.set_xlabel(r'$\phi$')
    plt.savefig('2evolution.png', dpi=400)
