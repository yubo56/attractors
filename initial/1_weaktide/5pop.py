'''
plot distributions of s, q_f for the three populations Z1/Z2/Z3-cross/Z3-hop
'''
import os
import numpy as np
from multiprocessing import Pool
import scipy.optimize as opt
from scipy import integrate

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

from utils import solve_ic, to_ang, to_cart, get_etac, get_mu4, get_mu2,\
    stringify, H, roots, get_H4, s_c_str, get_mu_equil, get_anal_caps,\
    get_num_caps
PKL_FILE = '5dat%s_%d.pkl'
# N_PTS = 1 # TEST
N_PTS_TOTAL = 20000
N_THREADS = 50
N_PTS = N_PTS_TOTAL // N_THREADS
TF = 8000
TIMES = np.exp(np.linspace(0, np.log(TF), 100))

def solve_with_events(I, s_c, eps, mu0, phi0, s0):
    '''
    solves IVP then returns (mu, s, t) at phi=0, pi + others
    '''
    init = [*to_cart(np.arccos(mu0), phi0), s0]
    event = lambda t, y: y[1]
    t, _, _, ret = solve_ic(I, s_c, eps, init, TF,
                            rtol=1e-4,
                            events=[event], dense_output=True)
    sol_times = ret.sol(TIMES)
    q, phi = to_ang(*sol_times[0:3])
    s = sol_times[3]

    [t_events] = ret.t_events
    x_events, _, mu_events, s_events = ret.sol(t_events)
    # phi = 0 means x < 0
    idxs_0 = np.where(x_events < 0)
    idxs_pi = np.where(x_events > 0)
    mu_0, s_0, t_0 = mu_events[idxs_0], s_events[idxs_0], t_events[idxs_0]
    mu_pi, s_pi, t_pi = mu_events[idxs_pi], s_events[idxs_pi], t_events[idxs_pi]

    shat_f = np.sqrt(np.sum(ret.y[ :3, -1]**2))
    return (mu_0, s_0, t_0, mu_pi, s_pi, t_pi), np.cos(q), phi, s

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
        args, mu, phi, s = solve_with_events(I, s_c, eps, mu0, phi0, s0)

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
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.8
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    plt.figure(figsize=(8, 8))
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
    ax_scatter.scatter(phi1, mu1, c='r', label='EQ1', s=ms)
    ax_scatter.scatter(phi2, mu2, c='g', label='EQ2', s=ms)
    ax_scatter.set_xlabel(r'$\phi$')
    ax_scatter.set_ylabel(r'$\cos \theta$')
    ax_scatter.legend(loc='lower left')
    ax_scatter.set_title(r'$s_c = %.2f, I = %d^\circ$' % (s_c, np.degrees(I)))
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

    # plot hist vs mu0 (significant blending, okay)
    n, bins, _ = ax_hist.hist(
        [mu2, mu1], bins=60, color=['g', 'r'],
        orientation='horizontal', stacked=True)
    ax_hist.set_ylim(ax_scatter.get_ylim())

    plt.savefig('5Hhists%s_%d.png' % (s_c_str(s_c), np.degrees(I)), dpi=400)
    plt.close()

    # try to overplot the semi-analytical simulations I ran
    pkl_fn = '6pc_dist%s.pkl' % s_c_str(s_c)
    if os.path.exists(pkl_fn):
        n_mu = 101
        n_phi = 60
        mu_vals =  np.linspace(-0.99, 0.99, n_mu)
        with open(pkl_fn, 'rb') as f:
            cross_dat = pickle.load(f)
        # p_caps = get_anal_caps(I, s_c, cross_dat, mu_vals)
        p_caps = get_num_caps(I, s_c, cross_dat, mu_vals)
        tot_probs = np.sum(p_caps / n_phi, axis=1)
        plt.plot(mu_vals, tot_probs, 'k:')

        bin_cents = (bins[ :-1] + bins[1: ]) / 2
        bin_probs = np.array(n[0]) / np.array(n[1])
        plt.plot(bin_cents, bin_probs, 'bo', ms=2)
        plt.savefig('5pc_fits%s_%d.png' % (s_c_str(s_c), np.degrees(I)), dpi=400)
        plt.close()

def plot_cum_probs(I, s_c_vals, counts):
    '''
    plot probabilities of ending up in CS1 and the obliquities of the two
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0)
    ax1.plot(s_c_vals, np.array(counts) / (N_THREADS * N_PTS), 'bo')
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('CS1 Prob')

    # calculate locations of equilibria in ~continuous way
    s_c_crit = get_etac(I) # etac = s_c_crit / (s = 1)
    s_c_cont = np.concatenate((
        np.linspace(min(s_c_vals), s_c_crit * 0.8, 30),
        np.linspace(s_c_crit * 0.8, s_c_crit * 0.99, 30),
        np.linspace(s_c_crit * 1.01, max(s_c_vals), 30)
    ))
    cs1_equil_mu = -np.ones_like(s_c_cont) # -1 = does not exist
    cs2_equil_mu = np.zeros_like(s_c_cont)
    for idx, s_c in enumerate(s_c_cont):
        cs1_crit_mu, cs2_crit_mu = get_crit_mus
        cs2_equil_mu[idx] = cs2_crit_mu
        if cs1_crit_mu is not None:
            cs1_equil_mu = cs1_crit_mu

    cs1_idxs = np.where(cs1_equil_mu > -1)[0]
    ax2.plot(np.array(s_c_cont)[cs1_idxs],
             np.degrees(np.arccos(cs1_equil_mu[cs1_idxs])),
             'g', label='CS1 Eq')
    ax2.plot(s_c_cont, np.degrees(np.arccos(cs2_equil_mu)), 'b', label='CS2 Eq')
    ax2.set_ylabel(r'$\theta$')
    ax2.set_xlabel(r'$s_c / \Omega_1$')
    ax2.legend()
    plt.savefig('5probs_%d' % np.degrees(I), dpi=400)
    plt.close()

def run():
    eps = 1e-3
    s0 = 10

    s_c_vals_20 = [
        2.0,
        1.2,
        1.0,
        0.7,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.06,
    ]

    s_c_vals_5 = [
        # 2.0,
        # 1.2,
        # 1.0,
        # 0.85,
        # 0.8,
        # 0.75
        0.7,
        # 0.65,
        # 0.6,
        # 0.55,
        # 0.5,
        # 0.45,
        # 0.4,
        # 0.35,
        # 0.3,
        # 0.25,
        0.2,
        0.06,
    ]

    for I, s_c_vals in [
            [np.radians(5), s_c_vals_5],
            # [np.radians(20), s_c_vals_20],
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
            counts.append(len(IC_eq1))
            # plot_final_dists(I, s_c, s0, trajs)
            plot_eq_dists(I, s_c, s0, np.array(IC_eq1), np.array(IC_eq2))
        # plot_cum_probs(I, s_c_vals, counts)

if __name__ == '__main__':
    run()

    # seems to be the "top edge too close to 1 case", cannot integrate well
    # I = np.radians(5)
    # s_c = 0.7
    # eps = 1e-3
    # mu0 = -0.896953070778169
    # phi0 = 2.155276797870799
    # s0 = 10
    # args, mu, phi, s = solve_with_events(I, s_c, eps, mu0, phi0, s0)
    # t_cross, _ = get_sep_hop(*args)
    # print(mu[-1])
