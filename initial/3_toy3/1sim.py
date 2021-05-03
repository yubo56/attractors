import os
import pickle
import scipy.optimize as opt
from scipy import integrate
from scipy.interpolate import interp1d
import numpy as np
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
PLOT_DIR = '1plots'
from utils import to_cart, to_ang, solve_ic, get_areas, get_plot_coords,\
    roots, H

EPS = 3e-4

def my_fmt(I_deg, delta):
    return '%d_%s' % (I_deg, ('%.2f' % delta).replace('.', '_'))

def run_single(I, tf, eta0, q0, delta):
    q4 = roots(I, eta0)[3]

    y0 = [*to_cart(q0, 0.2), eta0]
    return solve_ic(I, EPS, delta, y0, tf)

def stats_runner(I, q0, delta):
    tf = 3000
    n_eta = 61

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

def get_hop_anal(I, eta, delta):
    ''' P_hop(eta) analytical from bottom'''
    bot = (
        2 * np.pi * (1 - 2 * eta * np.sin(I) - delta * eta * np.cos(I))
            + (16 * np.cos(I) * eta + 4 * delta) * np.sqrt(eta * np.sin(I)))
    top = (
        2 * np.pi * (1 - 2 * eta * np.sin(I) - delta * eta * np.cos(I))
            - (16 * np.cos(I) * eta + 4 * delta) * np.sqrt(eta * np.sin(I)))
    return (bot - top) / bot

def get_hop_approx(I, eta, delta):
    ''' eta = scalar, numerically integrate over approximate separatrix '''
    def mu_up(phi):
        return eta * np.cos(I) + np.sqrt(2 * eta * np.sin(I) * (1 - np.cos(phi)))

    def mu_down(phi):
        return eta * np.cos(I) - np.sqrt(2 * eta * np.sin(I) * (1 - np.cos(phi)))

    def arg_top(phi):
        m = mu_up(phi)
        return (1 - m**2) + delta * (
            -eta * np.cos(I) - np.sqrt(eta * np.sin(I) * (1 - np.cos(phi)) / 2)
        )

    def arg_bot(phi):
        m = mu_down(phi)
        return (1 - m**2) + delta * (
            -eta * np.cos(I) + np.sqrt(eta * np.sin(I) * (1 - np.cos(phi)) / 2)
        )

    eps = 0.15 # integrand = 0/0 @ endpoints, omit
    top = integrate.quad(arg_top, eps, 2 * np.pi - eps)[0]
    bot = integrate.quad(arg_bot, eps, 2 * np.pi - eps)[0]
    return (bot - top) / bot

def get_hop_num(I, eta, delta):
    ''' eta = scalar, numerically integrate over exact separatrix '''
    q4 = roots(I, eta)[3]
    def mu_up(phi):
        def dH(q):
            return H(I, eta, q, phi) - H(I, eta, q4, 0)
        return np.cos(opt.brentq(dH, q4, 0))

    def mu_down(phi):
        def dH(q):
            return H(I, eta, q, phi) - H(I, eta, q4, 0)
        return np.cos(opt.brentq(dH, -np.pi, q4))

    def arg_top(phi):
        m = mu_up(phi)
        return (1 - m**2) + delta * eta * (
            np.sin(I) - eta * np.cos(I)**2
                + (m * np.cos(I) - np.sin(I) * np.sqrt(1 - m**2)
                   * np.cos(phi))) / (-m + eta * np.cos(I))

    def arg_bot(phi):
        m = mu_down(phi)
        return (1 - m**2) + delta * eta * (
            np.sin(I) - eta * np.cos(I)**2
                + (m * np.cos(I) - np.sin(I) * np.sqrt(1 - m**2)
                   * np.cos(phi))) / (-m + eta * np.cos(I))

    eps = 0.15 # integrand = 0/0 @ endpoints, omit
    top = integrate.quad(arg_top, eps, 2 * np.pi - eps)[0]
    bot = integrate.quad(arg_bot, eps, 2 * np.pi - eps)[0]
    return (bot - top) / bot

def get_hop_traj(I, eta_i, delta):
    ''' eta = scalar, numerically integrate over IC-integrated trajectory '''
    tf = 150
    # quick heuristic calculation of Delta_-, to IC for evolve around sep
    q4 = roots(I, eta_i)[3]
    def mu_down(phi):
        return eta_i * np.cos(I) - np.sqrt(
            2 * eta_i * np.sin(I) * (1 - np.cos(phi)))

    def arg_top(phi):
        m = mu_up(phi)
        return (1 - m**2) + delta * (
            -eta_i * np.cos(I) - np.sqrt(eta_i * np.sin(I) * (1 - np.cos(phi)) / 2)
        )

    def arg_bot(phi):
        m = mu_down(phi)
        return (1 - m**2) + delta * (
            -eta_i * np.cos(I) + np.sqrt(eta_i * np.sin(I) * (1 - np.cos(phi)) / 2)
        )

    bot = EPS * integrate.quad(arg_bot, 0, 2 * np.pi)[0]
    def dH(q):
        # return bot/5 below H4
        return H(I, eta_i, q, 0) - (H(I, eta_i, q4, 0) - bot / 5)
    q0 = opt.brentq(dH, -np.pi, q4)
    y0 = [*to_cart(q0, 0), eta_i]
    ret = solve_ic(I, EPS, delta, y0, tf)

    # figure out timestamps of end of bottom (max(phi)), end of top (min(phi))
    # examine over t in [0, t_events[0][3]], which is after 1 circle
    # mostly monotonic function, should converge easily if restrict domain well
    max_phi_val = lambda t: -to_ang(*ret.sol(t)[ :3])[1]
    ret1 = opt.minimize(max_phi_val, ret.t_events[0][3] / 2)
    end_bottom = ret1.x[0]

    min_phi_val = lambda t: to_ang(*ret.sol(t)[ :3])[1]
    ret2 = opt.minimize(min_phi_val, 2 * ret.t_events[0][3] / 2)
    end_top = ret2.x[0]

    # now actually integrate, using realistic values
    def arg_top(phi):
        t = np.linspace(end_bottom, end_top, 200)
        x, y, z, eta_sol = ret.sol(t)
        q, phi_sol = to_ang(x, y, z)
        m = interp1d(phi_sol, np.cos(q))
        eta = interp1d(phi_sol, eta_sol)
        try:
            return (1 - m(phi)**2) + delta * eta(phi) * (
                np.sin(I) - eta(phi) * np.cos(I)**2
                    + (m(phi) * np.cos(I) - np.sin(I) * np.sqrt(1 - m(phi)**2)
                       * np.cos(phi))) / (-m(phi) + eta(phi) * np.cos(I))
        except:
            print(phi)
            raise
    def arg_bot(phi):
        t = np.linspace(0, end_bottom, 200)
        x, y, z, eta_sol = ret.sol(t)
        q, phi_sol = to_ang(x, y, z)
        m = interp1d(phi_sol, np.cos(q))
        eta = interp1d(phi_sol, eta_sol)
        try:
            return (1 - m(phi)**2) + delta * eta(phi) * (
                np.sin(I) - eta(phi) * np.cos(I)**2
                    + (m(phi) * np.cos(I) - np.sin(I) * np.sqrt(1 - m(phi)**2)
                       * np.cos(phi))) / (-m(phi) + eta(phi) * np.cos(I))
        except:
            print(phi, phi_sol.min(), phi_sol.max())
            raise

    eps = 0.4 # integrand = 0/0 @ endpoints + traj doesn't go all the way
    phi_vals = np.linspace(eps, 2 * np.pi - eps, 201)
    top_vals = [arg_top(f) for f in phi_vals]
    bot_vals = [arg_bot(f) for f in phi_vals]
    top = integrate.simps(top_vals, x=phi_vals)
    bot = integrate.simps(bot_vals, x=phi_vals)
    return (bot - top) / bot

def run_stats(I_deg, delta, p=None, n_q=151, pkl_template='1dat%s.pkl'):
    I = np.radians(I_deg)

    PKL_FN = pkl_template % my_fmt(I_deg, delta)
    if not os.path.exists(PKL_FN):
        q_vals = -np.pi / 2 - np.linspace(0.3, 0.4, n_q)

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

    # overplot fits
    fit_anal = get_hop_anal(I, eta_vals, delta)
    ax2.plot(eta_vals, fit_anal, 'r', linewidth=2, label='Anal.')
    # fit_quad_approx = [get_hop_approx(I, eta, delta) for eta in eta_vals]
    # ax2.plot(eta_vals, fit_quad_approx, 'g:', linewidth=2, label='Num.')
    fit_quad_num = [get_hop_num(I, eta, delta) for eta in eta_vals]
    ax2.plot(eta_vals, fit_quad_num, 'k:', linewidth=2, label='Num.')
    # fit_quad_traj = [get_hop_traj(I, eta, delta) for eta in eta_vals]
    # ax2.plot(eta_vals, fit_quad_traj, 'm:', linewidth=2, label='Num int')

    ax1.set_title(r'$I = %d^\circ$' % I_deg)
    ax2.set_ylabel(r'$P_c$')
    ax1.set_ylabel('Counts')
    ax2.legend()
    plt.savefig('1hist%s.png' % my_fmt(I_deg, delta),
                dpi=400)
    plt.close(fig)

def plot_0(n_q=151):
    I = np.radians(20)
    PKL_FN = '1dat_many%s.pkl' % my_fmt(20, 0)
    with open(PKL_FN, 'rb') as f:
        res_arr = pickle.load(f)

    captures, escapes = [], []
    for res in res_arr:
        for eta_cross, capture in res:
            if capture:
                captures.append(eta_cross)
            else:
                escapes.append(eta_cross)

    fig = plt.figure(figsize=(6, 6))
    plt.xlabel(r'$\eta$')

    n, bins = np.histogram(captures, bins=40)
    n2, _ = np.histogram(escapes, bins=bins)
    ntot = n + n2

    eta_vals = (bins[ :-1] + bins[1: ]) / 2
    # ax2.plot(eta_vals, n[0] / n[1])
    nonzero_idxs = np.where(n != 0)[0]
    plt.plot(eta_vals[nonzero_idxs],
                 n[nonzero_idxs] / ntot[nonzero_idxs],
                 'ko', label='Sim')

    # overplot fits
    fit_anal = get_hop_anal(I, eta_vals, 0)
    plt.plot(eta_vals, fit_anal, 'r', linewidth=2, label='Theory')

    plt.ylabel(r'$P_{\rm III \to II}$')
    plt.legend(fontsize=14, loc='lower right')
    plt.tight_layout()
    plt.savefig('1hist_toy', dpi=300)
    plt.close()

if __name__ == '__main__':
    p = mp.Pool(8)
    # for delta in [0, 0.3, 0.5, 0.7]:
    #     run_stats(20, delta, p)
    run_stats(20, 0, p, pkl_template='1dat_many%s.pkl', n_q=1000)
    plot_0(n_q=1000)
