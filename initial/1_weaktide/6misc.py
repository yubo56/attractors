'''
misc little plots

python 6*.py, check the mu_finals
'''
import numpy as np
from multiprocessing import Pool
import os, pickle, lzma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
from scipy.optimize import brenth
from scipy.interpolate import interp1d
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

POOL_SIZE = 50
TRAJ_COLORS = ['darkorchid', 'red', 'fuchsia',
               'deepskyblue', 'blue', 'teal', 'k']

from utils import roots, s_c_str, get_mu_equil, solve_ic, to_cart, to_ang,\
    get_H4, H, get_mu4, get_ps_anal, get_anal_caps, get_num_caps, get_etac,\
    solve_with_events5

def get_cs_val(I, s_c, s):
    '''
    calculates CS1, CS2 locations for an array s
    returns -1 if no CS for that state
    '''
    cs1_qs = np.full_like(s, -1)
    cs2_qs = np.full_like(s, -1)
    for idx, s_val in enumerate(s):
        cs_qs = roots(I, s_c, s_val)
        if len(cs_qs) == 4:
            cs1_qs[idx] = cs_qs[0]
            cs2_qs[idx] = cs_qs[1]
        else:
            cs2_qs[idx] = cs_qs[0]
    return cs1_qs, cs2_qs

def equils_plot(I, s_c, eps, mu0, phi0, s0, tf, fn_str, idx, rtol=1e-9,
                time_idxbase=[100, 300, 600, 4600]):
    plotdir = '%s_sims' % fn_str
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    fn = '%s/%s_%d' % (plotdir, s_c_str(s_c), idx)
    pkl_fn = fn + '.pkl'

    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        solve_ret = solve_with_events5(I, s_c, eps, mu0, phi0, s0, tf, rtol=rtol)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump(solve_ret, f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            solve_ret = pickle.load(f)

    (mu_0, s_0, t_0, mu_pi, s_pi, t_pi), mu, phi, _s, ret_solveivp = solve_ret
    t = ret_solveivp.t
    [t_events] = ret_solveivp.t_events
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.axes([0.1, 0.3, 0.35, 0.45])
    ax2 = plt.axes([0.465, 0.58, 0.2, 0.4])
    ax3 = plt.axes([0.695, 0.58, 0.2, 0.4])
    ax4 = plt.axes([0.465, 0.15, 0.2, 0.4])
    ax5 = plt.axes([0.695, 0.15, 0.2, 0.4])
    axs = [ax1, ax2, ax3, ax4, ax5]
    svec = ret_solveivp.y[0:3, :]
    s = ret_solveivp.y[3, :]
    q, phi = to_ang(*svec)
    axs[0].plot(s, np.degrees(q), alpha=0.5, c=TRAJ_COLORS[idx], lw=1.0)
    last_idx = np.where(t > 0.98 * tf)[0]
    # axs[0].plot(s[last_idx], np.degrees(q)[last_idx], 'ko', markersize=2.0)
    axs[0].set_xlim(0, 3)
    axs[0].set_ylim(0, 135)
    axs[0].set_xlabel(r'$\Omega_{\rm s}$')
    axs[0].set_ylabel(r'$\theta$ (deg)')
    if s_c < 0.1:
        axs[0].set_title(r'$\eta_{\rm sync} = %.2f$' % s_c)
    else:
        axs[0].set_title(r'$\eta_{\rm sync} = %.1f$' % s_c)

    s_lt = np.linspace(s_c / 10, 1, 200) # max eta = 10
    s_gt = np.linspace(1, 3, 200) # other interesting part of the interval
    s_tot = np.concatenate((s_lt, s_gt))
    cs1_qs, cs2_qs = get_cs_val(I, s_c, s_tot)
    cs1_exist_idx = np.where(cs1_qs > -1)[0]
    mu_equil_lt = [np.degrees(np.arccos(get_mu_equil(s))) for s in s_lt]

    axs[0].plot(s_tot, np.degrees(cs2_qs), 'tab:green', label='CS2', lw=2.5)
    axs[0].plot(s_tot[cs1_exist_idx], np.degrees(-cs1_qs[cs1_exist_idx]),
               'darkorange', label='CS1', lw=2.5)
    axs[0].plot(s_lt, mu_equil_lt, 'k', lw=4)

    times = np.array(time_idxbase) * tf / 5000
    time_idxs = [np.where(t > _t)[0][0] for _t in times]
    ylim_bot = -1
    ylim_top = 1
    for time, t_idx, ax in zip(times, time_idxs, axs[1: ]):
        axs[0].axvline(s[t_idx], c='k', lw=0.7, ls='--')
        # should span 2 events total
        if time > np.max(t_events):
            # happens if convergence to tCE2 and libration width is smaller than
            # CS2 offset (i.e. doesn't cross phi = pi any more)
            tmin = time - 100
            tmax = time + 100
        else:
            t_lt = np.where(t_0 < time)[0]
            t_gt = np.where(t_0 > time)[0]
            if len(t_gt) == 0:
                t_lt = np.where(t_pi < time)[0]
                t_gt = np.where(t_pi > time)[0]
                tmin = t_pi[t_lt[-1]]
                tmax = t_pi[t_gt[1]]
            else:
                tmin = t_0[t_lt[-1]]
                tmax = t_0[t_gt[0]]
        plt_t = np.where(np.logical_and(
            t > tmin,
            t < tmax))[0]
        mu_plt = np.cos(q)[plt_t]
        num_t = len(plt_t)
        dense_t = np.linspace(0, num_t - 1, 10 * num_t)
        phi_dense = interp1d(np.arange(num_t), np.unwrap(phi[plt_t]))(dense_t)
        mu_dense = interp1d(np.arange(num_t), mu_plt)(dense_t)
        ax.plot(phi_dense, mu_dense, marker='o', ls='', c=TRAJ_COLORS[idx],
                ms=1.0)
        ax.set_xticks([0, np.pi, 2 * np.pi])

        # overplot separatrix if exists
        etac = get_etac(I)
        if s_c / s[t_idx] < etac:
            N = 100
            phi_grid, mu_grid = np.meshgrid(np.linspace(0, 2 * np.pi, N),
                                            np.linspace(-1, 1, N))
            H4 = get_H4(I, s_c, s[t_idx])
            H_grid = H(I, s_c, s[t_idx], mu_grid, phi_grid)
            ax.contour(phi_grid, mu_grid, H_grid, levels=[H4], colors='k')

        ax.text(0.1, -1 + 0.03, r'$\Omega_{\rm s} = %.1f$' % s[t_idx],
                ha='left', va='bottom', color='k', fontsize=14)

        if np.min(mu_plt) < -0.9:
            ylim_bot =-1.1
        if np.max(mu_plt) > 0.9:
            ylim_top = 1.1
    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_ylim(ylim_bot, ylim_top)
        ax.set_xlim(0, 2 * np.pi)
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([r'$0$', r'$180$', r'$360$'])
    ax5.set_xticklabels([r'$0$', r'$180$', r'$360$'])
    ax4.set_xlabel(r'$\phi$')
    ax5.set_xlabel(r'$\phi$')
    ax2.set_yticklabels([])
    ax4.set_yticklabels([])
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.set_ylabel(r'$\cos \theta$')
    ax5.yaxis.set_label_position("right")
    ax5.yaxis.tick_right()
    ax5.set_ylabel(r'$\cos \theta$')
    plt.savefig(fn, dpi=300)
    plt.close()

def plot_equils(I, s_cs, fn_str='6equils', tf=5000):
    ''' plot in (s, mu) space showing how the tCS arise '''

    s_dq = np.linspace(2, 3, 200)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # plot the first one separately
    s_c = s_cs[0]
    s_lt = np.linspace(s_c / 10, 1, 200) # max eta = 10
    s_gt = np.linspace(1, 3, 200) # other interesting part of the interval
    s_tot = np.concatenate((s_lt, s_gt))

    for idx, c in enumerate(TRAJ_COLORS[ :6]):
        pkl_fn = '%s_sims/%s_%d.pkl' % (fn_str, s_c_str(s_c), idx)
        with lzma.open(pkl_fn, 'rb') as f:
            solve_ret = pickle.load(f)
        (mu_0, s_0, t_0, mu_pi, s_pi, t_pi), mu, phi, _s, ret_solveivp = solve_ret
        svec = ret_solveivp.y[0:3, :]
        s = ret_solveivp.y[3, :]
        t = ret_solveivp.t
        q, phi = to_ang(*svec)
        _lw = 1.5 if (idx == 2 or idx == 3) else 0.8
        _al = 0.8 if (idx == 2 or idx == 3) else 0.4
        ax.plot(s, np.degrees(q), alpha=_al, c=c, lw=_lw)
        offset = 0.5 if idx == 3 else (-0.5 if idx == 2 else 0)
        ax.plot(s[0], np.degrees(q)[0] + offset, ms=10,
                c=c, marker='x')

    cs1_qs, cs2_qs = get_cs_val(I, s_c, s_tot)
    cs1_exist_idx = np.where(cs1_qs > -1)[0]
    mu_equil_lt = [np.degrees(np.arccos(get_mu_equil(s))) for s in s_lt]

    ax.set_ylabel(r'$\theta$ (deg)')
    ax.set_ylim([0, 130])

    dS_interp = interp1d(s_lt, mu_equil_lt)
    CS2_interp = interp1d(s_tot, np.degrees(cs2_qs))
    tce2_s = brenth(lambda s: dS_interp(s) - CS2_interp(s), np.min(s_tot), 1)

    ax.plot(s_tot, np.degrees(cs2_qs), 'tab:green', label='CS2', lw=2.5)
    dash_idx = np.where(s_tot[cs1_exist_idx] > 2)[0]
    solid_idx = np.where(s_tot[cs1_exist_idx] < 2)[0]
    ax.plot(s_tot[cs1_exist_idx][dash_idx],
             np.degrees(-cs1_qs[cs1_exist_idx][dash_idx]), 'darkorange',
             label='CS1', lw=2.5, ls='--')
    ax.plot(s_tot[cs1_exist_idx][solid_idx],
             np.degrees(-cs1_qs[cs1_exist_idx][solid_idx]), 'darkorange',
             label='CS1', lw=2.5)
    ax.text(s_tot[-1], np.degrees(cs2_qs)[-1], 'CS2', c='tab:green',
             fontsize=14, va='bottom', ha='right')
    ax.text(s_tot[-1], -np.degrees(cs1_qs)[-1], 'CS1', c='darkorange',
             fontsize=14, va='bottom', ha='right')
    ax.plot(s_lt, mu_equil_lt, 'k', lw=4)
    ax.plot(s_dq, np.degrees(np.arccos(2 / s_dq)), 'b', lw=4)

    # label text along boundaries
    ax.text(0.75, 55, r'$\dot{\Omega}_{\rm s} < 0$', c='k', fontsize=14,
             rotation=-53)
    ax.text(0.6, 48, r'$\dot{\Omega}_{\rm s} > 0$', c='k', fontsize=14,
             rotation=-53)

    ax.text(2.45, 40, r'$\dot{\theta}_{\rm tide} < 0$', c='b', fontsize=14,
             rotation=25)
    ax.text(2.55, 30, r'$\dot{\theta}_{\rm tide} > 0$', c='b', fontsize=14,
             rotation=25)

    ax.plot(tce2_s, dS_interp(tce2_s), mec='tab:green', mfc='none', marker='o',
             ms=15, mew=3)
    ax.text(tce2_s, dS_interp(tce2_s) + 10, 'tCE2', c='tab:green', va='bottom',
             ha='center')
    # ax.text(np.max(s_dq), 110, r'$\eta_{\rm sync} = %.2f$' % s_c,
    #         va='top', ha='right')
    ax.set_title(r'$\eta_{\rm sync} = %.2f$' % s_c)

    s1_search_min = np.min(s_tot[cs1_exist_idx])
    s1_search_max = min(np.max(s_tot[cs1_exist_idx]), 1)
    CS1_interp = interp1d(s_tot[cs1_exist_idx],
                          np.degrees(-cs1_qs[cs1_exist_idx]))
    tce1_s = brenth(lambda s: dS_interp(s) - CS1_interp(s), s1_search_min,
                    s1_search_max)
    ax.plot(tce1_s, dS_interp(tce1_s), mec='darkorange', mfc='none',
             marker='o', ms=15, mew=3)
    ax.text(tce1_s * 0.97, dS_interp(tce1_s) + 5, 'tCE1', c='darkorange',
             va='bottom', ha='right')
    ax.set_xlabel(r'$\Omega_{\rm s} / n$')

    plt.tight_layout()
    plt.savefig(fn_str + 'single', dpi=400)
    plt.clf()

    s_cs = s_cs[1: ]

    fig, axs = plt.subplots(
        len(s_cs), 1,
        figsize=(6, 2 + 3 * len(s_cs)),
        sharex=True)
    axs[-1].set_xlabel(r'$\Omega_{\rm s} / n$')

    for ax, s_c, in zip(axs, s_cs):
        s_lt = np.linspace(s_c / 10, 1, 200) # max eta = 10
        s_gt = np.linspace(1, 3, 200) # other interesting part of the interval
        s_tot = np.concatenate((s_lt, s_gt))

        for idx, c in enumerate(TRAJ_COLORS[ :6]):
            pkl_fn = '%s_sims/%s_%d.pkl' % (fn_str, s_c_str(s_c), idx)
            with lzma.open(pkl_fn, 'rb') as f:
                solve_ret = pickle.load(f)
            (mu_0, s_0, t_0, mu_pi, s_pi, t_pi), mu, phi, \
                _s, ret_solveivp = solve_ret
            svec = ret_solveivp.y[0:3, :]
            s = ret_solveivp.y[3, :]
            t = ret_solveivp.t
            q, phi = to_ang(*svec)
            _al = 0.4 if (idx != 0 or s_c != 0.7) else 0.8
            _lw = 0.8 if (idx != 0 or s_c != 0.7) else 1.5
            ax.plot(s, np.degrees(q), alpha=_al, c=c, lw=_lw)
            offset = 0.5 if (idx == 3 or idx == 5) else\
                (-0.5 if (idx == 2 or idx == 6) else 0)
            ax.plot(s[0], np.degrees(q)[0] + offset, ms=10,
                    c=c, marker='x',
                    )

        cs1_qs, cs2_qs = get_cs_val(I, s_c, s_tot)
        cs1_exist_idx = np.where(cs1_qs > -1)[0]
        mu_equil_lt = [np.degrees(np.arccos(get_mu_equil(s))) for s in s_lt]

        ax.set_ylabel(r'$\theta$ (deg)')
        ax.set_ylim([0, 130])

        dS_interp = interp1d(s_lt, mu_equil_lt)
        CS2_interp = interp1d(s_tot, np.degrees(cs2_qs))
        tce2_s = brenth(lambda s: dS_interp(s) - CS2_interp(s), np.min(s_tot), 1)

        ax.plot(s_tot, np.degrees(cs2_qs), 'tab:green', label='CS2', lw=2.5)
        dash_idx = np.where(s_tot[cs1_exist_idx] > 2)[0]
        solid_idx = np.where(s_tot[cs1_exist_idx] < 2)[0]
        ax.plot(s_tot[cs1_exist_idx][dash_idx],
                 np.degrees(-cs1_qs[cs1_exist_idx][dash_idx]), 'darkorange',
                 label='CS1', lw=2.5, ls='--')
        ax.plot(s_tot[cs1_exist_idx][solid_idx],
                 np.degrees(-cs1_qs[cs1_exist_idx][solid_idx]), 'darkorange',
                 label='CS1', lw=2.5)
        ax.text(s_tot[-1], np.degrees(cs2_qs)[-1], 'CS2', c='tab:green',
                 fontsize=14, va='bottom', ha='right')
        ax.text(s_tot[-1], -np.degrees(cs1_qs)[-1], 'CS1', c='darkorange',
                 fontsize=14, va='bottom', ha='right')
        ax.plot(s_lt, mu_equil_lt, 'k', lw=4)
        ax.plot(s_dq, np.degrees(np.arccos(2 / s_dq)), 'b', lw=4)

        # label text along boundaries
        if s_c < 0.4:
            ax.text(0.75, 50, r'$\dot{\Omega}_{\rm s} < 0$', c='k', fontsize=14,
                     rotation=-53)
            ax.text(0.6, 38, r'$\dot{\Omega}_{\rm s} > 0$', c='k', fontsize=14,
                     rotation=-53)
        else:
            ax.text(0.1, 63, r'$\dot{\Omega}_{\rm s} < 0$', c='k', fontsize=14,
                     rotation=-30)
            ax.text(0.2, 77, r'$\dot{\Omega}_{\rm s} > 0$', c='k', fontsize=14,
                     rotation=-30)

        ax.text(2.45, 40, r'$\dot{\theta}_{\rm tide} < 0$', c='b', fontsize=14,
                 rotation=25)
        ax.text(2.55, 26, r'$\dot{\theta}_{\rm tide} > 0$', c='b', fontsize=14,
                 rotation=25)

        ax.plot(tce2_s, dS_interp(tce2_s), mec='tab:green', mfc='none', marker='o',
                 ms=15, mew=3)
        ax.text(tce2_s, dS_interp(tce2_s) - 23, 'tCE2', c='tab:green',
                va='top', ha='right')
        # ax.text(np.max(s_dq), 110, r'$\eta_{\rm sync} = %.2f$' % s_c,
        #         va='top', ha='right')
        ax.set_title(r'$\eta_{\rm sync} = %.2f$' % s_c)

        etac = get_etac(I)
        if s_c < etac:
            s1_search_min = np.min(s_tot[cs1_exist_idx])
            s1_search_max = min(np.max(s_tot[cs1_exist_idx]), 1)
            CS1_interp = interp1d(s_tot[cs1_exist_idx],
                                  np.degrees(-cs1_qs[cs1_exist_idx]))
            tce1_s = brenth(lambda s: dS_interp(s) - CS1_interp(s), s1_search_min,
                            s1_search_max)
            ax.plot(tce1_s, dS_interp(tce1_s), mec='darkorange', mfc='none',
                     marker='o', ms=15, mew=3)
            ax.text(tce1_s * 1.1, dS_interp(tce1_s) + 3, 'tCE1', c='darkorange',
                     va='bottom', ha='left')

    plt.tight_layout()
    # fig.subplots_adjust(hspace=0.06)
    plt.savefig(fn_str, dpi=300)
    plt.clf()

def plot_phop(I, s_c):
    s = np.linspace(2 * s_c, 10, 100)
    tops, bots = get_ps_anal(I, s_c, s)
    pc32 = (tops + bots) / bots
    p_caps32 = np.minimum(pc32, np.ones_like(pc32))
    pc12 = (tops + bots) / tops
    p_caps12 = np.minimum(pc12, np.ones_like(pc12))
    plt.ylim(bottom=0)
    plt.xlabel(r'$s / \Omega_1$')
    plt.ylabel('Probability')
    plt.plot(s, p_caps32, label=r'$III \to II$')
    plt.plot(s, p_caps12, label=r'$I \to II$')
    plt.legend()
    plt.savefig('6pc%s' % s_c_str(s_c), dpi=400)

def get_cross_dat(I, s_c, s0, eps, tf, mu0, phi0):
    [mu4] = get_mu4(I, s_c, np.array([s0]))

    H4_0 = get_H4(I, s_c, s0)
    H_0 = H(I, s_c, s0, mu0, phi0)
    if H_0 > H4_0:
        print('Inside separatrix for', mu0, phi0)
        return [-1, 0]

    # stop sim when H = H4
    print('Running for', mu0, phi0)
    init = [*to_cart(np.arccos(mu0), phi0), s0]
    def event(t, y):
        x, y, z, s = y
        _, phi = to_ang(x, y, z)
        H4 = get_H4(I, s_c, s)
        H_curr = H(I, s_c, s, z, phi)
        dH = H_curr - H4
        return dH
    event.terminal = True
    _, _, s, ret = solve_ic(I, s_c, eps, init, tf,
                               rtol=1e-4,
                               dense_output=True,
                               events=[event])
    if ret.t_events[0].size > 0:
        return [s[-1], mu0 - mu4]
    else:
        return [-s[-1], 0]

def plot_equil_dist_anal(I, s_c, s0, eps, tf=8000):
    pkl_fn = '6pc_dist%s.pkl' % s_c_str(s_c)
    n_mu = 501
    n_phi = 50
    mu_vals =  np.linspace(-0.99, 0.99, n_mu)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    if not os.path.exists(pkl_fn):
        # store tuple (s_cross, mu0 - mu4)
        # s_cross convention: -1 = inside separatrix (pcap = 1),
        # negative = - s_final (no encounter; s_f currently unused)
        cross_dat = np.zeros((n_mu, n_phi, 2), dtype=np.float64)

        # build arguments array up from scratch
        args = []
        for idx, mu0 in enumerate(mu_vals):
            for idx2, phi0 in enumerate(phi_vals):
                args.append((I, s_c, s0, eps, tf, mu0, phi0))
                # cross_dat[idx, idx2] = get_cross_dat(I, s_c, s0, eps, tf, mu0, phi0)
        p = Pool(POOL_SIZE)
        res = p.starmap(get_cross_dat, args)
        cross_dat = np.reshape(np.array(res), (n_mu, n_phi, 2))
        with open(pkl_fn, 'wb') as f:
            pickle.dump(cross_dat, f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            cross_dat = pickle.load(f)
    try:
        p_caps_anal = get_anal_caps(I, s_c, cross_dat, mu_vals)
        p_caps = get_num_caps(I, s_c, cross_dat, mu_vals)
        tot_probs_anal = np.sum(p_caps_anal / n_phi, axis=1)
        tot_probs = np.sum(p_caps / n_phi, axis=1)
        plt.plot(mu_vals, tot_probs_anal, 'ro', ms=2, label='Anal')
        plt.plot(mu_vals, tot_probs, 'bo', ms=2, label='Num')
        plt.ylim([0, 1])
        plt.xticks([-1, -0.5, 0, 0.5, 1])
        plt.legend()
        plt.tight_layout()
        plt.savefig('6pc_dist%s' % s_c_str(s_c), dpi=300)
        plt.clf()
    except Exception as e: # on remote, can't plot, just return
        print(e)
        return

if __name__ == '__main__':
    eps = 1e-3
    I = np.radians(20)

    eps_equils = 1e-2
    equils_tf = 500
    # to display
    # equils_plot(I, 0.06, eps_equils, -0.5, 0, 2.5, equils_tf, '6equils', 2)
    # equils_plot(I, 0.06, eps_equils, -0.5, 5, 2.5, equils_tf, '6equils', 3)
    # equils_plot(I, 0.7, eps_equils, 0.985, 0, 2.5, equils_tf, '6equils', 0,
    #             time_idxbase=[100, 800, 1200, 4600])
    # equils_plot(I, 0.06, eps_equils, -0.4, 0, 0.1, equils_tf, '6equils', 6)
    # not to display
    # equils_plot(I, 0.06, eps_equils, 0.985, 0, 2.5, equils_tf, '6equils', 0)
    # equils_plot(I, 0.06, eps_equils, -0.05, np.pi, 2.5, equils_tf, '6equils', 1)
    # equils_plot(I, 0.06, eps_equils, 0.8, 0, 0.1, equils_tf, '6equils', 4)
    # equils_plot(I, 0.06, eps_equils, -0.2, 0, 0.1, equils_tf, '6equils', 5)
    # equils_plot(I, 0.5, eps_equils, 0.985, 0, 2.5, equils_tf, '6equils', 0)
    # equils_plot(I, 0.5, eps_equils, -0.05, np.pi, 2.5, equils_tf, '6equils', 1)
    # equils_plot(I, 0.5, eps_equils, -0.5, 0, 2.5, equils_tf, '6equils', 2)
    # equils_plot(I, 0.5, eps_equils, -0.5, 5, 2.5, equils_tf, '6equils', 3)
    # equils_plot(I, 0.5, eps_equils, 0.8, 0, 0.1, equils_tf, '6equils', 4)
    # equils_plot(I, 0.5, eps_equils, -0.2, 0, 0.1, equils_tf, '6equils', 5)
    # equils_plot(I, 0.7, eps_equils, -0.05, np.pi, 2.5, equils_tf, '6equils', 1)
    # equils_plot(I, 0.7, eps_equils, -0.5, 0, 2.5, equils_tf, '6equils', 2)
    # equils_plot(I, 0.7, eps_equils, -0.5, 5, 2.5, equils_tf, '6equils', 3)
    # equils_plot(I, 0.7, eps_equils, 0.8, 0, 0.1, equils_tf, '6equils', 4)
    # equils_plot(I, 0.7, eps_equils, -0.2, 0, 0.1, equils_tf, '6equils', 5)
    plot_equils(I, [0.06, 0.5, 0.7], tf=equils_tf)

    # equils_plot(I, 0.01, eps, -0.96, 2.56, 10, 8000, '6equils', 20, rtol=1e-4)
    # equils_plot(I, 0.01, eps, -0.96, 2.56, 10, 8000, '6equils', 21, rtol=1e-9)
    # equils_plot(I, 0.01, eps, -0.95, 0, 10, 8000, '6equils', 21)
    # equils_plot(I, 0.01, eps, -0.99, 0, 10, 8000, '6equils', 22)

    # plot_phop(I, 0.2)
    # plot_equil_dist_anal(I, 0.06, 10, eps)
    # plot_equil_dist_anal(I, 0.2, 10, eps)
    # plot_equil_dist_anal(I, 0.7, 10, eps)

    # test cases
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0.9, np.pi)) # no cross
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0, np.pi)) # in sep
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0, 0)) # fast cross

    # mu = -0.91
    # phi = np.pi / 2
    # mu_vals = np.array([mu])
    # cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, mu, phi)
    # print(cross_dat)
    # cross_dat_re = np.reshape(np.array(cross_dat), (1, 1, 2))
    # print(get_num_caps(I, 0.7, cross_dat_re, mu_vals))

    # for mu in [0.4, 0.45, 0.5]:
    # for mu in [0.5]:
    #     cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, mu, np.pi)
    #     print(cross_dat)
    #     print(get_num_caps(I, 0.7, np.reshape(np.array(cross_dat), (1, 1, 2))))
    # cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0.468, 1.0471975511965976)
    # print(cross_dat)
    # print(get_num_caps(I, 0.7, np.reshape(np.array(cross_dat), (1, 1, 2))))
