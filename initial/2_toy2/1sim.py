import os
import scipy.optimize as opt
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

PLOT_DIR = '1plots'
from utils import to_cart, to_ang, solve_ic, get_areas, get_plot_coords,\
    roots, H

def plot_traj(I, ret, filename):
    n_pts = 100 # number of points used to draw equator
    t = ret.t
    y = ret.y

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

    # plot projection
    q, phi = to_ang(*y[0:3])
    plot_x, plot_y = get_plot_coords(q, phi)
    ax1.plot(plot_x, plot_y, 'r', linewidth=1)

    # use this to plot equator + south pole
    bound_phi = np.linspace(0, 2 * np.pi, n_pts)
    bound_x = np.cos(bound_phi)
    bound_y = np.sin(bound_phi)
    ax1.plot(bound_x, bound_y, 'k', linewidth=5)

    # plot separatrix @ crossing
    t_areas, areas, t_cross = get_areas(ret)
    sep_idx = np.where(t > t_cross)[0][0]
    eta = y[3, sep_idx]
    cs_qs = roots(I, eta)
    if len(cs_qs) == 4:
        q4 = cs_qs[-1]
        # convention: -np.pi / 2 < q4 < 0

        phi_sep = np.linspace(0, 2 * np.pi, n_pts)[1: -1] # omit endpoints
        q_sep_top, q_sep_bot = np.zeros_like(phi_sep), np.zeros_like(phi_sep)
        for idx, phi in enumerate(phi_sep):
            def dH(q):
                return H(I, eta, q, phi) - H(I, eta, q4, 0)
            q_sep_bot[idx] = opt.brentq(dH, -np.pi, q4)
            q_sep_top[idx] = opt.brentq(dH, q4, 0)

        sep_top_x, sep_top_y = get_plot_coords(
            q_sep_top, phi_sep)
        sep_bot_x, sep_bot_y = get_plot_coords(
            q_sep_bot, phi_sep)

        x_4, y_4 = get_plot_coords(q4, 0)
        ax1.plot(sep_top_x, sep_top_y, 'k:', linewidth=2)
        ax1.plot(sep_bot_x, sep_bot_y, 'k:', linewidth=2)
        ax1.plot(x_4, y_4, 'go', markersize=8)

    ax1.set_xlabel(r'$\sin(\theta/2)\cos(\phi)$')
    ax1.set_ylabel(r'$\sin(\theta/2)\sin(\phi)$')

    # plot sep area + eta
    eta = y[3]
    ln1 = ax2.plot(t_areas, areas / (4 * np.pi), 'ro',
                   markersize=4, label=r'$A_{traj}$')
    ax2.set_ylabel(r'$A_{enc} / 4\pi$')

    circ_idx = np.where(t <= t_cross)[0]
    lib_idx = np.where(t > t_cross)[0]
    ln2 = ax2.plot(t[circ_idx], 2 * np.sqrt(eta[circ_idx] * np.sin(I)) / np.pi,
                   'b', linewidth=2, label=r'$A_{sep}$')
    ax2.plot(t[lib_idx], 4 * np.sqrt(eta[lib_idx] * np.sin(I)) / np.pi, 'b',
             linewidth=2)

    ax3 = ax2.twinx()
    ln3 = ax3.plot(t, eta, 'g:', linewidth=2, label=r'$\eta$')
    ax3.set_xlabel(r'$t$')
    ax3.set_ylabel(r'$\eta$')

    lns = ln1 + ln2 + ln3
    ax2.legend(lns, [l.get_label() for l in lns],
               loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

if __name__ == '__main__':
    I = np.radians(20)
    eps = 1e-4
    tf = 3000

    y0 = [*to_cart(np.pi/2 - 0.5, 0), 0.1]
    ret = solve_ic(I, eps, y0, tf)

    plot_traj(I, ret, '1testo.png')
