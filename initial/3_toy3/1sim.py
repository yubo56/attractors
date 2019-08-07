import os
import scipy.optimize as opt
import numpy as np
import multiprocessing as mp

import matplotlib
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

PLOT_DIR = '1plots'
from utils import to_cart, to_ang, solve_ic, get_areas, get_plot_coords,\
    roots, H, get_mu4

EPS = 3e-4

def my_fmt(I_deg, delta):
    return '%d_%s' % (I_deg, ('%.2f' % delta).replace('.', '_'))

def run_single(I, tf, eta0, q0, delta):
    q4 = roots(I, eta0)[3]

    y0 = [*to_cart(q0, 0.2), eta0]
    return solve_ic(I, EPS, delta, y0, tf)

def stats_runner(I, q0, delta):
    tf = 3000
    n_eta = 201

    def result_single(tf, eta0):
        ret = run_single(I, tf, eta0, q0, delta)

        q_f, phi_f = to_ang(ret.y[0][-1], ret.y[1][-1], ret.y[2][-1])
        eta_f = ret.y[3, -1]
        q4 = roots(I, eta_f)[3]
        dH_f = H(I, eta_f, q_f, phi_f) - H(I, eta_f, q4, 0)

        _, _, t_cross, _ = get_areas(ret)
        crossed_idxs = np.where(ret.t > t_cross)[0]
        if len(crossed_idxs) == 0:
            return dH_f > 0, eta0, -1
        eta_cross = ret.sol(t_cross)[3]
        return dH_f > 0, eta_cross, ret.y[2, -1]

    counts = 0
    res = []
    for eta0 in np.linspace(0.05, 0.2, n_eta):
        capture, eta_f, z_f = result_single(tf, eta0)
        print('\t', '%.3f' % q0, 'Finished', '%.3f' % (eta_f - eta0), capture,
              '%.3f' % z_f)
        res.append((eta_f, capture))
        if capture:
                counts += 1
    return res

def run_stats(I_deg, delta, p=None):
    I = np.radians(I_deg)
    n_q = 151

    PKL_FN = '1dat%s.pkl' % my_fmt(I_deg, delta)
    if not os.path.exists(PKL_FN):
        q_vals = -np.pi / 2 - np.linspace(0.1, 0.3, n_q)

        if p:
            res_arr = p.starmap(stats_runner, [(I, q0, delta) for q0 in q_vals])
        else:
            res_arr = [stats_runner(I, q0, delta) for q0 in q_vals]
        with open(PKL_FN, 'wb') as f:
            pickle.dump(res_arr, f)
    else:
        with open(PKL_FN, 'rb') as f:
            res_arr = pickle.load(f)

    captures, escapes = [], []
    for res in res_arr:
        for eta_cross, capture in res:
            if capture:
                captures.append(eta_cross)
            else:
                escapes.append(eta_cross)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    n, bins, _ = ax1.hist([captures, escapes],
                           bins=40,
                           label=['Capture', 'Escape'],
                           stacked= True)
    ax2.set_xlabel(r'$\eta_\star$')
    ax1.legend()

    eta_vals = (bins[ :-1] + bins[1: ]) / 2
    # ax2.plot(eta_vals, n[0] / n[1])
    nonzero_idxs = np.where(n[1] != 0)[0]
    ax2.errorbar(eta_vals[nonzero_idxs],
                 n[0][nonzero_idxs] / n[1][nonzero_idxs],
                 yerr = np.sqrt(n[0][nonzero_idxs]) / n[1][nonzero_idxs],
                 fmt='o', label='Data')

    # overplot fit
    def fit(eta):
        ''' P_hop(eta) analytical from bottom'''
        return (
            (32 * np.cos(I) * eta + 8 * delta) * np.sqrt(eta * np.sin(I))
        ) / (
            2 * np.pi * (1 - 2 * eta * np.sin(I) + delta * eta * np.cos(I))
                + (16 * np.cos(I) * eta + 4 * delta) * np.sqrt(eta * np.sin(I)))
    ax2.plot(eta_vals, fit(eta_vals), 'r', linewidth=2, label='Analytical')
    ax1.set_title(r'$I = %d^\circ$' % I_deg)
    ax2.set_ylabel(r'$P_c$')
    ax1.set_ylabel('Counts')
    ax2.legend()
    plt.savefig('1hist%s.png' % my_fmt(I_deg, delta),
                dpi=400)
    plt.close(fig)

if __name__ == '__main__':
    p = mp.Pool(4)
    for delta in [0, 0.3, 0.5, 0.7]:
        run_stats(20, delta, p)
