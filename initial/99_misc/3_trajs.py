'''
plot various trajectories in phase space
'''
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

def get_dydt(I, eta, eps):
    ''' get dy/dt for params '''
    def dydt(t, s):
        x, y, z = s
        return [
            y * z - eta * y * np.cos(I) - eps * z * x,
            -x * z + eta * (x * np.cos(I) - z * np.sin(I)) - eps * z * y,
            eta * y * np.sin(I) + eps * (1 - z**2),
        ]
    return dydt

def to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    q = np.arccos(z / r)
    phi = (np.arctan2(y / np.sin(q), x / np.sin(q)) + np.pi)\
        % (2 * np.pi)
    return np.array([q, phi])

def to_cart(q, phi):
    return [
        -np.sin(q) * np.cos(phi),
        -np.sin(q) * np.sin(phi),
        np.cos(q),
    ]

def trajs(I, eta, eps_mult=0, tf=300, n_pts=101, n_t=101, phi_c=None):
    '''
    for a range of ICs, integrate trajectory until returning to initial phi,
    then colorize plot of trajectory + period times
    '''
    eps_crit = eta * np.sin(I) / (1 - (eta * np.cos(I))**2)
    eps = eps_mult * eps_crit
    # default central phi = CS2's phi
    if phi_c is None:
        eps_m = min(eps_mult, 0.99) * eps_crit # eps_mult > 1 fails arcsin
        mu2 = eta * np.cos(I) / (1 + eta * np.sin(I))
        phi_c = np.pi - np.arcsin(eps_m * (1 - mu2**2) / (eta * np.sin(I)))

    dydt = get_dydt(I, eta, eps)
    t_lengths = []
    plots = []
    for q in np.linspace(0, np.pi, n_pts + 2)[1: -1]:
        y0 = to_cart(q, phi_c)
        events = [lambda t, y: to_ang(*y0)[1] - to_ang(*y)[1]]
        events[0].direction = +1
        ret = solve_ivp(dydt, [0, tf], y0,
                        events=events, dense_output=True)

        t_vals = np.linspace(ret.t_events[0][0], ret.t_events[0][1], n_t)
        q, phi = to_ang(*ret.sol(t_vals))
        plots.append((q, np.unwrap(phi)))
        t_lengths.append(ret.t_events[0][1] - ret.t_events[0][0])
    norm = cm.colors.Normalize(vmin=min(t_lengths), vmax=max(t_lengths))
    cmap = cm.RdBu
    for t_len, (q, phi) in zip(t_lengths, plots):
        gt_idx = np.where(phi > 2 * np.pi)[0]
        lt_idx = np.where(phi < 0)[0]
        mid_idx = np.where(np.logical_and(
            phi > 0,
            phi < 2 * np.pi,
        ))[0]
        plt.plot(phi[lt_idx] + 2 * np.pi, np.cos(q[lt_idx]),
                 c=cmap(norm(t_len)), linewidth=0.5)
        plt.plot(phi[gt_idx] - 2 * np.pi, np.cos(q[gt_idx]),
                 c=cmap(norm(t_len)), linewidth=0.5)
        plt.plot(phi[mid_idx], np.cos(q[mid_idx]),
                 c=cmap(norm(t_len)), linewidth=0.5)
    plt.axvline(phi_c, c='k')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos\theta$')
    eps_str = '0' if eps == 0 else r'10^{%.2f}' % np.log10(eps)
    plt.title(r'$I = %d^\circ, \eta = %.2f, \epsilon = %s (%.2f \epsilon_c)$' %
              (np.degrees(I), eta, eps_str, eps_mult))
    plt.savefig('3trajs_%s.png' %
                ('0_00' if eps_mult == 0 else
                 ('%.2f' % eps_mult).replace('.','_')))
    plt.clf()

if __name__ == '__main__':
    I = np.radians(5)
    eta = 0.15

    trajs(I, eta, eps_mult=1.05)
    trajs(I, eta, eps_mult=0.95)
    trajs(I, eta, eps_mult=0.45)
    trajs(I, eta, eps_mult=0.2)
    trajs(I, eta, eps_mult=0.05)
    trajs(I, eta)
