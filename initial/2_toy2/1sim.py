import os
import scipy.optimize as opt
import numpy as np
from multiprocessing import Pool

import matplotlib
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

PLOT_DIR = '1plots'
from utils import to_cart, to_ang, solve_ic, get_areas, get_plot_coords,\
    roots, H, get_mu4

def plot_traj(I, ret, filename):
    ''' convention: -np.pi / 2 < q4 < 0 '''
    n_pts = 100 # number of points used to draw equator
    t = ret.t
    y = ret.y
    lw = 1

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

    # plot projection
    q, phi = to_ang(*y[0:3])
    plot_x, plot_y = get_plot_coords(q - np.pi, phi) # consistency w/ convention
    ax1.plot(plot_x, plot_y, 'r', linewidth=1)
    ax1.plot(plot_x[-1], plot_y[-1], 'co', markersize=3)

    # use this to plot equator + south pole
    bound_phi = np.linspace(0, 2 * np.pi, n_pts)
    bound_x = np.cos(bound_phi)
    bound_y = np.sin(bound_phi)
    ax1.plot(bound_x, bound_y, 'k', linewidth=5)

    # plot separatrix @ end
    t_areas, areas, t_cross, ends_circ = get_areas(ret)
    # print('Initial area', areas[0])
    eta_f = y[3, -1]
    cs_qs = roots(I, eta_f)
    q4 = cs_qs[-1]

    phi_sep = np.linspace(0, 2 * np.pi, n_pts)[1: -1] # omit endpoints
    q_sep_top, q_sep_bot = np.zeros_like(phi_sep), np.zeros_like(phi_sep)
    for idx, phi in enumerate(phi_sep):
        def dH(q):
            return H(I, eta_f, q, phi) - H(I, eta_f, q4, 0)
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
                   markersize=1, label=r'$A_{traj}$')
    ax2.set_ylabel(r'$A_{enc} / 4\pi$')

    circ_idx = np.where(t <= t_cross)[0]
    post_idx = np.where(t > t_cross)[0]
    # area enclosed in librating case is just sep area, but action in
    # circulating case has to subtract out mu4
    mu4 = get_mu4(I, eta[circ_idx])
    circ_area = np.sign(y[2][0] - mu4) * 2 * np.pi * mu4\
        + 8 * np.sqrt(eta[circ_idx] * np.sin(I))
    ln2 = ax2.plot(t[circ_idx], circ_area / (4 * np.pi),
                   'b', linewidth=lw, label=r'$A_{sep}$')
    # account for post_idx depending on ends_circ (whether ends circulating)
    if ends_circ:
        mu4_post = get_mu4(I, eta[post_idx])
        circ_area_post = np.sign(y[2][-1] - mu4_post) * 2 * np.pi * mu4_post\
            + 8 * np.sqrt(eta[post_idx] * np.sin(I))
        ax2.plot(t[post_idx], circ_area_post / (4 * np.pi),
                 'b', linewidth=lw)
    else:
        ax2.plot(t[post_idx], 4 * np.sqrt(eta[post_idx] * np.sin(I)) / np.pi,
                 'b', linewidth=lw)

    ax3 = ax2.twinx()
    ln3 = ax3.plot(t, eta, 'g:', linewidth=lw, label=r'$\eta$')
    ax3.set_xlabel(r'$t$')
    ax3.set_ylabel(r'$\eta$')

    lns = ln1 + ln2 + ln3
    ax2.legend(lns, [l.get_label() for l in lns],
               loc='upper left', fontsize=8)

    # label title w/ result
    q_f, phi_f = to_ang(y[0][-1], y[1][-1], y[2][-1])
    dH_f = H(I, eta_f, q_f, phi_f) - H(I, eta_f, q4, 0)
    ax1.set_title(r'$\mu_0 = %.3f$ (%s)' %
                  (y[2][0], 'Capture' if dH_f > 0 else 'Escape'))
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def run_single(I, eps, tf, eta0, q0):
    q4 = roots(I, eta0)[3]

    y0 = [*to_cart(q0, 0.2), eta0]
    return solve_ic(I, eps, y0, tf)

def plot_single(I, eps, tf, eta0, q0, filename):
    ret = run_single(I, eps, tf, eta0, q0)
    plot_traj(I, ret, filename)

def stats_runner(I, q0):
    tf = 10000
    def result_single(eps, tf, eta0):
        ret = run_single(I, eps, tf, eta0, q0)

        q_f, phi_f = to_ang(ret.y[0][-1], ret.y[1][-1], ret.y[2][-1])
        eta_f = ret.y[3, -1]
        q4 = roots(I, eta_f)[3]
        dH_f = H(I, eta_f, q_f, phi_f) - H(I, eta_f, q4, 0)

        _, _, t_cross, _ = get_areas(ret)
        crossed_idxs = np.where(ret.t > t_cross)[0]
        if len(crossed_idxs) == 0:
            return dH_f > 0, -1
        return dH_f > 0, ret.y[3, crossed_idxs[0]]

    counts = 0
    n_eta = 50
    n_eps = 30
    res = []
    for eta0 in np.linspace(0.01, 0.02, n_eta):
        for eps in np.linspace(1e-3, 2e-3, n_eps):
            capture, eta_f = result_single(eps, tf, eta0)
            print('\t', q0, 'Finished', eta0, eps, capture, eta_f)
            res.append((eta_f, capture))
            if capture:
                counts += 1
    return res

def run_stats(I_deg):
    I = np.radians(I_deg)

    PKL_FN = '1dat%d.pkl' % I_deg
    if not os.path.exists(PKL_FN):
        p = Pool(4)
        res_arr = p.starmap(stats_runner, [
            (I, -np.pi / 2 + 0.11),
            (I, -np.pi / 2 + 0.115),
            (I, -np.pi / 2 + 0.12),
            (I, -np.pi / 2 + 0.125),
            (I, -np.pi / 2 + 0.13),
            (I, -np.pi / 2 + 0.135),
            (I, -np.pi / 2 + 0.14),
            (I, -np.pi / 2 + 0.145),
        ])
        with open(PKL_FN, 'wb') as f:
            pickle.dump(res_arr, f)
    else:
        with open(PKL_FN, 'rb') as f:
            res_arr = pickle.load(f)

    captures, escapes = [], []
    for res in res_arr:
        for etaf, capture in res:
            if capture:
                captures.append(etaf)
            else:
                escapes.append(etaf)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    n, bins, _ = ax2.hist([captures, escapes],
                           bins=40,
                           label=['Capture', 'Escape'],
                           stacked= True)
    ax2.set_xlabel(r'$\eta_\star$')
    ax2.legend()

    eta_vals = (bins[ :-1] + bins[1: ]) / 2
    # ax1.plot(eta_vals, n[0] / n[1])
    ax1.errorbar(eta_vals, n[0] / n[1], yerr = np.sqrt(n[0]) / n[1],
                 fmt='o', label='Data')
    mean_p = np.mean(n[0] / n[1])
    ax1.axhline(mean_p, c='r', label='Mean')
    fit = lambda eta: 48 * np.cos(I) * np.sqrt(eta * np.sin(I)) / (
        4 * np.pi * np.sin(I)
            + 24 * np.cos(I) * np.sqrt(eta * np.sin(I))
            + 4 * np.pi * eta * np.cos(I)**2)
    ax1.plot(eta_vals, fit(eta_vals), 'g:', linewidth=2, label='Fit')
    ax1.set_title(r'$I = %d^\circ, \langle \eta_\star \rangle = %.3f$'
                  % (I_deg, mean_p))
    ax1.set_ylabel('Capture Probability')
    ax1.legend()
    plt.savefig('1hist%d.png' % I_deg)
    plt.close(fig)

if __name__ == '__main__':
    # I = np.radians(20)
    # tf = 10000
    # eta0 = 0.01
    # capture/escape cases respectively
    # plot_single(I, 1e-3, tf, eta0, -np.pi / 2 + 0.14, '1testo1.png')
    # plot_single(I, 1.2e-3, tf, eta0, -np.pi / 2 + 0.14, '1testo2.png')

    run_stats(20)
    run_stats(10)
    run_stats(25)
