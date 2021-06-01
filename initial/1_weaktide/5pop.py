'''
plot distributions of s, q_f for the three populations Z1/Z2/Z3-cross/Z3-hop
'''
import os, pickle, lzma
import numpy as np
from multiprocessing import Pool
import scipy.optimize as opt
from scipy.interpolate import interp1d
from scipy import integrate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

from utils import solve_ic, to_ang, to_cart, get_etac, get_mu4, get_mu2,\
    stringify, H, roots, get_H4, s_c_str, get_mu_equil, get_anal_caps,\
    get_num_caps, get_crit_mus, get_areas_ward, solve_with_events5, TIMES, TF
# PKL_FILE = '5dat%s_%d.pkl'
PKL_FILE = '5dat_hightol%s_%d.pkl'
# N_PTS = 1 # TEST
N_PTS_TOTAL = 20000
N_THREADS = 64
N_PTS = N_PTS_TOTAL // N_THREADS

def get_sep_hop(t_0, s_0, mu_0, t_pi, s_pi, mu_pi):
    '''
    gets the sep hop/cross time, else -1
    '''
    # two ways to detect a separatrix crossing, either 1) t_0s stop appearing,
    # or 2) mu_pi - mu_0 changes signs
    if len(t_0) and len(t_pi) and t_0[-1] < t_pi[-2]:
        # ends at librating, not circulating solution

        # (nb) technically, this could also be a circulating solution after
        # bifurcation, ignore for now
        return t_0[-1], s_0[-1]
    else:
        # (nb) technically, mu_0, mu_pi are evaluated at different times, but
        # for circulating solutions they interlock, so they are evaluated at
        # similar enough times to get the sep crossing to reasonable precision
        len_min = min(len(mu_0), len(mu_pi))
        dmu_signs = np.sign(mu_0[ :len_min] - mu_pi[ :len_min])

        if len(dmu_signs) > 0 and dmu_signs[0] != dmu_signs[-1]:
            # can end in a circulating solution about CS1 that is librating
            # about mu=1! still plot sep crossing as normal though

            # criterion 2, circulating and sign flip, sep crossing
            t_cross_idx = np.where(dmu_signs == dmu_signs[-1])[0][0]
            return t_0[t_cross_idx], s_0[t_cross_idx]
        else:
            # no separatrix crossing, even distribution
            return -1, -1

def _run_sim_thread(I, eps, s_c, s0, num_threads, thread_idx):
    '''
    run N_PTS random sims
    '''
    H4 = get_H4(I, s_c, s0)
    trajs = [[], [], [], []]

    def get_outcome_for_init(mu0, phi0, thread_idx, idx):
        '''
        returns 0-3 describing early stage outcome
        '''
        print('(%d-%d/%d) Starting for %.2f, %.3f, %.3f' %
              (thread_idx, idx, N_PTS, s_c, mu0, phi0))
        args, mu, phi, s, _ = solve_with_events5(I, s_c, eps, mu0, phi0, s0, TF,
                                                 rtol=1e-9)

        t_cross, _ = get_sep_hop(*args)
        H_f = H(I, s_c, s[-1], mu[-1], phi[-1])
        if t_cross == -1: # no sep encounter, either above or below H4
            if H_f > H4:
                return 1, (mu, s, mu0, phi0)
            else:
                # assert z_f > 0, 'z_f is %f' % z_f
                return 0, (mu, s, mu0, phi0)
        else:
            if H_f > H4:
                return 3, (mu, s, mu0, phi0)
            else:
                # assert z_f > 0, 'z_f is %f' % z_f
                return 2, (mu, s, mu0, phi0)

    np.random.seed()
    mus = (np.random.rand(N_PTS) * 2) - 1
    phis = np.random.rand(N_PTS) * 2 * np.pi
    # TEST
    # mus = [np.cos(2.21700)]
    # phis = [2.70410]
    # mus = [0.99]
    # phis = [0]
    for mu0, phi0, idx in zip(mus, phis, range(N_PTS)):
        outcome, traj = get_outcome_for_init(mu0, phi0, thread_idx, idx)
        trajs[outcome].append(traj)
    return trajs

def run_sim(I, eps, s_c, s0=10, num_threads=1):
    '''
    returns list of trajs from sim, memoized

    for fixed s0, s_c: random (mu, phi), evolve forward in time. Outcomes:
    I - Go to CS1, no separatrix encounter
    II - Go to CS2, no separatrix encounter
    III - separatrix traversing, to CSI
    IV - separatrix hopping, to CS2
    V - Go to CS3, no separatrix encounter (unstable, no)

    VI - Above separatrix, bifurcation
    VII - Inside separatrix, bifurcation
    VIII - Below separatrix, bifurcation (not sure exists?)

    Really, I-IV (V is impossible) are the only initial outcomes, VI/VII are
    later outcomes since bifurcation is later than separatrix interactions. So
    we will focus on (s0, mu0, phi0) -> {I, IV} outcome probability
    '''
    pkl_fn = PKL_FILE % (s_c_str(s_c), np.degrees(I))

    if not os.path.exists(pkl_fn):
        print('Running sims, %s not found' % pkl_fn)
        assert num_threads > 0

        # TEST
        # _run_sim_thread(I, eps, s_c, s0, num_threads, 0)
        p = Pool(num_threads)
        traj_lst = p.starmap(_run_sim_thread, [
            (I, eps, s_c, s0, num_threads, thread_idx)
            for thread_idx in range(num_threads)
        ])

        # merge the list of [traj1, traj2, traj3, traj4] in traj_lst
        trajs = [[], [], [], []]
        for traj_thread in traj_lst:
            for sim_traj, target_traj in zip(traj_thread, trajs):
                target_traj.extend(sim_traj)

        with open(pkl_fn, 'wb') as f:
            pickle.dump(trajs, f)

    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            trajs = pickle.load(f)
    return trajs

def plot_final_dists(I, s_c, s0, trajs):
    '''
    plot distribution of (mu, s) over time
    '''
    fig, _axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = _axes.flat
    fig.subplots_adjust(hspace=0, wspace=0)

    stride = len(TIMES) // 7 # number of times to plot (mu, s)
    t_colors = TIMES[::stride]
    for ax, outcome_trajs in zip(axes, trajs):
        if not outcome_trajs:
            continue
        # mu, s, mu0, phi0
        for mu, s, _, _ in outcome_trajs[::20]:
            scat = ax.scatter(s[::stride], mu[::stride],
                              c=t_colors,
                              norm=matplotlib.colors.LogNorm(),
                              cmap='rainbow',
                              s=0.5**2)
            # overplot a big black dot on the last one
            ax.scatter(s[-1], mu[-1], c='k', s=2**2)

    # set up all labels/bounds
    axes[2].set_xlabel(r'$s_f$')
    axes[2].set_xlim([0, s0])
    axes[2].set_ylabel(r'$\cos \theta_f$')
    axes[2].set_ylim([-1, 1])
    cbar = fig.colorbar(scat, ax=_axes.ravel().tolist())
    cbar.minorticks_off()
    cbar.set_ticks(t_colors)
    cbar.set_ticklabels([round(t) for t in t_colors])
    plt.suptitle(r'$I = %d^\circ, s_c = %.2f, T_f = %d$ (NH-1, NH-2, X1, X2)' %
                 (np.degrees(I), s_c, TF))
    plt.savefig('5outcomes%s_%d.png' % (s_c_str(s_c), np.degrees(I)), dpi=400)
    plt.close()

def plot_eq_dists(I, s_c, s0, IC_eq1, IC_eq2):
    '''
    plot scatter + theta-binned hist
    '''
    # set up axes
    left, width = 0.22, 0.53
    bottom, height = 0.12, 0.76
    spacing = 0.05
    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom,
                  0.98 - (left + width + spacing), height]
    plt.figure(figsize=(6, 6))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_hist = plt.axes(rect_histy)
    ax_hist.tick_params(direction='in', labelleft=False)

    # plot scatter
    ms = 3
    # this is a bit of a stupid unpacking, but avoids having to case on IC*
    # being zero-length
    mu1 = [ic[0] for ic in IC_eq1]
    mu2 = [ic[0] for ic in IC_eq2]
    phi1 = [ic[1] for ic in IC_eq1]
    phi2 = [ic[1] for ic in IC_eq2]
    ax_scatter.scatter(phi1, mu1, c='darkorange', s=ms)
    ax_scatter.scatter(phi2, mu2, c='tab:green', s=ms)
    ax_scatter.set_xlabel(r'$\phi_{\rm i}$ (deg)')
    ax_scatter.set_ylabel(r'$\cos \theta_{\rm i}$')
    ax_scatter.set_ylim(-1, 1)
    ax_scatter.set_xlim(0, 2 * np.pi)
    ylim = ax_scatter.get_ylim()
    ax_scatter.scatter(3, -10, c='darkorange',
                       label='tCE1', s=20)
    ax_scatter.scatter(3, -10, c='tab:green', label='tCE2', s=20)
    ax_scatter.legend(loc='lower center', fontsize=14,
                      bbox_to_anchor=(left + (width + spacing) / 2,
                                      1),
                      ncol=2)
    ax_scatter.set_ylim(ylim)
    # ax_scatter.set_title(r'$s_c = %.2f, I = %d^\circ$' % (s_c, np.degrees(I)))
    # overplot separatrix
    lw = 2
    n_pts = 50
    phi_sep = np.linspace(0, 2 * np.pi, n_pts)
    mu_sep_top, mu_sep_bot = np.zeros_like(phi_sep), np.zeros_like(phi_sep)
    mu4 = get_mu4(I, s_c, np.array([s0]))[0]
    for idx, phi in enumerate(phi_sep):
        if idx == 0 or idx == phi:
            mu_sep_bot[idx] = mu4
            mu_sep_top[idx] = mu4
            continue
        def dH(mu):
            return H(I, s_c, s0, mu, phi) - H(I, s_c, s0, mu4, 0)
        mu_sep_bot[idx] = opt.brentq(dH, -1, mu4)
        mu_sep_top[idx] = opt.brentq(dH, mu4, 1)
    ax_scatter.plot(phi_sep, mu_sep_bot, 'k', lw=lw)
    ax_scatter.plot(phi_sep, mu_sep_top, 'k', lw=lw)
    ax_scatter.set_xticks([0, np.pi, 2 * np.pi])
    ax_scatter.set_xticklabels([r'$0$', r'$180$', r'$360$'])

    ax_scatter.text(np.pi, mu4, 'II', backgroundcolor=(1, 1, 1, 0.9),
                    fontsize=14, ha='center', va='center')
    ax_scatter.text(np.pi, np.max(mu_sep_top) + 0.1, 'I',
                    backgroundcolor=(1, 1, 1, 0.9), fontsize=14, ha='center',
                    va='center')
    ax_scatter.text(np.pi, np.min(mu_sep_bot) - 0.1, 'III',
                    backgroundcolor=(1, 1, 1, 0.9), fontsize=14, ha='center',
                    va='center')

    # plot hist vs mu0 (significant blending, okay)
    n, bins, _ = ax_hist.hist(
        [mu2, mu1], bins=60, color=['tab:green', 'darkorange'],
        orientation='horizontal', stacked=True)
    ax_hist.set_ylim(ax_scatter.get_ylim())

    plt.savefig('5Hhists%s_%d.png' % (s_c_str(s_c), np.degrees(I)), dpi=300)
    plt.close()

    # try to overplot the semi-analytical simulations I ran
    pkl_fn = '6pc_disthtol%s.pkl' % s_c_str(s_c)
    if np.degrees(I) == 20 and os.path.exists(pkl_fn):
        n_phi = 50
        with open(pkl_fn, 'rb') as f:
            cross_dat = pickle.load(f)
        n_mu = len(cross_dat)
        mu_vals =  np.linspace(-0.99, 0.99, n_mu)
        p_caps = get_num_caps(I, s_c, cross_dat, mu_vals)
        tot_probs = np.sum(p_caps / n_phi, axis=1)
        if s_c < 0.1:
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(6, 6),
                gridspec_kw={'height_ratios': [2, 1]},
                sharex=True)
        else:
            fig = plt.figure(figsize=(6, 4))
            ax1 = plt.gca()
        ax1.plot(mu_vals, tot_probs, 'b', alpha=0.8, lw=3.5)

        # try anal prob
        p_caps_al = get_anal_caps(I, s_c, cross_dat, mu_vals)
        s_crosses = cross_dat[:, :, 0]
        tot_probs_al = np.sum(p_caps_al / n_phi, axis=1)

        bin_cents = (bins[ :-1] + bins[1: ]) / 2
        bin_probs = np.array(n[0]) / np.array(n[1])
        ax1.plot(bin_cents, bin_probs, 'ro', ms=3)
        ax1.set_ylabel(r'tCE2 Probability')
        s_c_text = '%.1f' % s_c if s_c > 0.1 else '%.2f' % s_c
        ax1.text(1, 1, r'$\eta_{\rm sync} = %s$' % s_c_text,
                 ha='right', va='top', fontsize=16)

        if s_c < 0.1:
            ax1.plot(mu_vals, tot_probs_al, 'g--', alpha=0.8, lw=1.0)
            for mu, crosses in zip(mu_vals, s_crosses):
                cross_idx = np.where(crosses > 0)[0]
                all_crosses = crosses[cross_idx]
                ax2.plot(np.full_like(crosses[cross_idx], mu),
                         s_c / all_crosses,
                         'bo', alpha=0.8, markersize=0.8)

            ax2.set_ylabel(r'$\eta_{\rm cross}$')
            ax2.set_xlabel(r'$\cos \theta_{\rm i}$')
            ax2.axhline(s_c, c='k', ls='--', lw=1.5)
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.03)
        else:
            ax1.set_xlabel(r'$\cos \theta_{\rm i}$')
            plt.tight_layout()
        plt.savefig('5pc_fits%s_%d.png' % (s_c_str(s_c), np.degrees(I)), dpi=400)
        plt.close()

def plot_cum_probs(I, s_c_vals, s0, counts):
    '''
    plot probabilities of ending up in tCS2 and the obliquities of the two
    '''
    s_c_crit = get_etac(I) # etac = s_c_crit / (s = 1)
    fig, (ax2, ax1) = plt.subplots(
        2, 1, figsize=(7, 6),
        gridspec_kw={'height_ratios': [2, 3]},
        sharex=True)
    # fig.subplots_adjust(hspace=0)
    probs_dat = np.array(counts) / (N_THREADS * N_PTS)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel(r'$P_{\rm tCE2}$')
    s_c_cont = np.concatenate((
        np.linspace(min(s_c_vals) / 10, s_c_crit * 0.8, 30),
        np.linspace(s_c_crit * 0.8, s_c_crit * 0.99, 30),
        np.linspace(s_c_crit * 1.01, max(s_c_vals), 30)
    ))
    A2s = np.array([get_areas_ward(I, s_c, s0)[1] for s_c in s_c_cont])
    A3s = np.array([get_areas_ward(I, s_c, s0)[2] for s_c in s_c_cont])
    A2_interp = interp1d(s_c_cont, A2s / (4 * np.pi))
    A3_interp = interp1d(s_c_cont, A3s / (4 * np.pi))
    idxs = np.argsort(s_c_vals)
    A2frac = A2_interp(np.array(s_c_vals))
    A3frac = A3_interp(np.array(s_c_vals))
    ax1.fill_between(np.array(s_c_vals)[idxs],
                     A2frac[idxs],
                     facecolor='tab:green',
                     alpha=0.2)
    ax1.fill_between(np.array(s_c_vals)[idxs],
                     A2frac[idxs],
                     np.minimum(probs_dat, A2frac + A3frac)[idxs],
                     facecolor='tab:blue',
                     alpha=0.2)
    ax1.fill_between(np.array(s_c_vals)[idxs],
                     np.minimum(probs_dat, A2frac + A3frac)[idxs],
                     probs_dat[idxs],
                     facecolor='darkorange',
                     alpha=0.2)
    ax1.text(2.0, 0.05, 'Inititially zone II', fontsize=14, ha='right')
    ax1.text(2.0, 0.5, 'Initially zone III', fontsize=14, ha='right')
    ax1.text(2.0, 0.9, 'Initially zone I', fontsize=14, ha='right')

    # calculate locations of equilibria in ~continuous way
    ax1.axvline(s_c_crit, c='k', lw=1.0, ls='--')
    ax2.axvline(s_c_crit, c='k', lw=1.0, ls='--')
    cs1_equil_mu = -np.ones_like(s_c_cont) # -1 = does not exist
    cs2_equil_mu = np.zeros_like(s_c_cont)
    for idx, s_c in enumerate(s_c_cont):
        cs1_crit_mu, cs2_crit_mu = get_crit_mus(I, s_c)
        cs2_equil_mu[idx] = cs2_crit_mu
        if cs1_crit_mu is not None:
            cs1_equil_mu[idx] = cs1_crit_mu

    cs1_idxs = np.where(cs1_equil_mu > -1)[0]
    ax2.plot(np.array(s_c_cont)[cs1_idxs],
             np.degrees(np.arccos(cs1_equil_mu[cs1_idxs])),
             'darkorange', label='tCE1', lw=2)
    ax2.plot(s_c_cont, np.degrees(np.arccos(cs2_equil_mu)),
             'tab:green', label='tCE2', lw=2)
    ax2.set_ylabel(r'$\theta$')

    # eta_arr = (s_c_cont * np.sin(I) + np.sqrt(
    #         s_c_cont**2 * np.sin(I)**2
    #         + 8 * np.cos(I) * s_c_cont)) / (4 * np.cos(I))
    # tce2_anal = np.degrees(np.arccos(
    #     eta_arr * np.cos(I) / (1 + eta_arr * np.sin(I))
    #     ))
    eta_arr = np.sqrt(s_c_cont / (2 * np.cos(I)))
    tce2_anal = np.degrees(np.arccos(
        np.sqrt(s_c_cont * np.cos(I) / 2)
        ))
    idxs = np.where(s_c_cont < s_c_crit)[0]
    ax2.plot(s_c_cont[idxs], tce2_anal[idxs], 'b--', lw=1.5)
    ax2.set_ylim(0, 90)
    ax2.set_yticks([0, 45, 90])
    ax2.set_yticklabels(['0', '45', '90'])
    ax1.set_xlim(left=-0.1)

    idx = np.where(s_c_cont < 0.4)[0]
    # (1/10)^(1/2) + 3 / (2 * (1 + (1/10)^(1/2)))
    # 1.4558
    ax1.plot(s_c_cont[idx],
             4 * 1.46 / np.pi * np.sqrt(s_c_cont * np.sin(I))[idx],
             'r--',
             lw=1.5)

    ax1.set_xlabel(r'$\eta_{\rm sync}$')
    ax1.scatter(s_c_vals, probs_dat, c='tab:red')
    ax2.legend(ncol=2, fontsize=14, loc='upper right')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    plt.savefig('5probs_%d' % np.degrees(I), dpi=300)
    plt.close()

def run():
    s_c_vals_20 = [
        # 0.7,
        0.2,
        0.06,
        # 2.0,
        # 1.2,
        # 1.0,
        # 0.65,
        # 0.6,
        # 0.55,
        0.5,
        # 0.45,
        # 0.4,
        # 0.35,
        # 0.3,
        # 0.25,
        # 0.1,
        # 0.03,
        # 0.01,
    ]
    s_c_vals_5 = [
        # 0.7,
        # 0.2,
        # 0.06,
        # 2.0,
        # 1.2,
        # 1.0,
        # 0.85,
        # 0.8,
        # 0.75,
        # 0.65,
        # 0.6,
        # 0.55,
        # 0.5,
        # 0.45,
        # 0.4,
        # 0.35,
        # 0.3,
        # 0.25,
        # 0.1,
        # 0.03,
        # 0.01,
    ]
    eps = 1e-3
    s0 = 10

    for I, s_c_vals in [
            [np.radians(5), s_c_vals_5],
            [np.radians(20), s_c_vals_20],
    ]:
        counts = []
        for s_c in s_c_vals:
            mu_cs_cache = []
            IC_eq1 = []
            IC_eq2 = []
            trajs = run_sim(I, eps, s_c, s0=s0, num_threads=N_THREADS)
            count = 0
            for outcome_trajs in trajs:
                for mu, s, mu0, phi0 in outcome_trajs:
                    mu_f = mu[-1]
                    for s_key, mu_cs_cand in mu_cs_cache:
                        if abs(s_key - s[-1]) < 1e-3:
                            mu_cs = mu_cs_cand
                            break
                    else:
                        mu_cs = np.cos(roots(I, s_c, s[-1]))
                        mu_cs_cache.append((s[-1], mu_cs))
                    if len(mu_cs) == 4 and abs(mu_f - mu_cs[0]) < 1e-3:
                        IC_eq1.append((mu0, phi0))
                    elif len(mu_cs) == 2 or\
                        (len(mu_cs) == 4 and abs(mu_f - mu_cs[1]) < 1e-3):
                        IC_eq2.append((mu0, phi0))
                    else:
                        print('Unable to classify (s_f, mu_f):', s[-1], mu_f)
            plot_final_dists(I, s_c, s0, trajs)
            plot_eq_dists(I, s_c, s0, np.array(IC_eq1), np.array(IC_eq2))

def plot_all_cumprobs():
    s_c_vals_20 = [
        0.7,
        0.2,
        0.06,
        2.0,
        1.2,
        1.0,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.1,
        0.03,
        # 0.01,
    ]
    s_c_vals_5 = [
        0.7,
        0.2,
        0.06,
        2.0,
        1.2,
        1.0,
        0.85,
        0.8,
        0.75,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.1,
        0.03,
        0.01,
    ]
    eps = 1e-3
    s0 = 10

    for Id, s_c_vals in [
            # [5, s_c_vals_5],
            [20, s_c_vals_20],
    ]:
        I = np.radians(Id)
        pkl_fn = '5cum_probs_%d.pkl' % Id
        if not os.path.exists(pkl_fn):
            print('Running %s' % pkl_fn)
            counts = []
            for s_c in s_c_vals:
                mu_cs_cache = []
                IC_eq1 = []
                IC_eq2 = []
                trajs = run_sim(I, eps, s_c, s0=s0, num_threads=N_THREADS)
                count = 0
                for outcome_trajs in trajs:
                    for mu, s, mu0, phi0 in outcome_trajs:
                        mu_f = mu[-1]
                        for s_key, mu_cs_cand in mu_cs_cache:
                            if abs(s_key - s[-1]) < 1e-3:
                                mu_cs = mu_cs_cand
                                break
                        else:
                            mu_cs = np.cos(roots(I, s_c, s[-1]))
                            mu_cs_cache.append((s[-1], mu_cs))
                        if len(mu_cs) == 4 and abs(mu_f - mu_cs[0]) < 1e-3:
                            IC_eq1.append((mu0, phi0))
                        elif len(mu_cs) == 2 or\
                            (len(mu_cs) == 4 and abs(mu_f - mu_cs[1]) < 1e-3):
                            IC_eq2.append((mu0, phi0))
                        else:
                            print('Unable to classify (s_f, mu_f):', s[-1], mu_f)
                counts.append(len(IC_eq2))
            with lzma.open(pkl_fn, 'wb') as f:
                pickle.dump(counts, f)
        else:
            with lzma.open(pkl_fn, 'rb') as f:
                print('Loading %s' % pkl_fn)
                counts = pickle.load(f)
        plot_cum_probs(I, s_c_vals, s0, counts)

def plot_anal_cs_equils(I=np.radians(20), s_c=0.2):
    s_c_crit = get_etac(I)
    s_c_cont = np.geomspace(s_c_crit / 100, 3, 100)
    cs1_equil_mu = -np.ones_like(s_c_cont) # -1 = does not exist
    cs2_equil_mu = np.zeros_like(s_c_cont)
    for idx, s_c in enumerate(s_c_cont):
        cs1_crit_mu, cs2_crit_mu = get_crit_mus(I, s_c)
        cs2_equil_mu[idx] = cs2_crit_mu
        if cs1_crit_mu is not None:
            cs1_equil_mu[idx] = cs1_crit_mu
    eta_arr = (s_c_cont * np.sin(I) + np.sqrt(
            s_c_cont**2 * np.sin(I)**2
            + 8 * np.cos(I) * s_c_cont)) / (4 * np.cos(I))
    plt.semilogx(s_c_cont, np.degrees(np.arccos(
        eta_arr * np.cos(I) / (1 + eta_arr * np.sin(I))
        )), 'k', lw=3)
    cs1_idxs = np.where(cs1_equil_mu > -1)[0]
    plt.plot(np.array(s_c_cont)[cs1_idxs],
             np.degrees(np.arccos(cs1_equil_mu[cs1_idxs])),
             'darkorange', label='tCE1', lw=2)
    eta_num = s_c_cont * ((1 + cs2_equil_mu**2) / (2 * cs2_equil_mu))
    plt.plot(s_c_cont, np.degrees(np.arccos(cs2_equil_mu)),
             'tab:green', label='tCE2', lw=2)
    plt.savefig('5anal_cs_equils', dpi=300)
    plt.close()

if __name__ == '__main__':
    # run()
    plot_all_cumprobs()
    # plot_anal_cs_equils()

    # bunch of debugging cases...
    # seems to be the "top edge too close to 1 case", cannot integrate well
    # I = np.radians(20)
    # s_c = 0.01
    # eps = 1e-3
    # mu0 = -0.96
    # phi0 = 2.56
    # s0 = 10
    # args, mu, phi, s, _ = solve_with_events5(I, s_c, eps, mu0, phi0, s0, TF)
    # t_cross, _ = get_sep_hop(*args)
    # print(mu[-1])
