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
    stringify, H, roots, get_H4, s_c_str
PKL_FILE = '5dat%s.pkl'
N_PTS = 1000
N_THREADS = 4
TF = 8000
TIMES = np.exp(np.linspace(0, np.log(TF), 100))

def solve_with_events(I, s_c, eps, mu0, phi0, s0):
    '''
    solves IVP then returns (mu, s, t) at phi=0, pi + others
    '''
    init_xy = np.sqrt(1 - mu0**2)
    init = [-init_xy * np.cos(phi0), -init_xy * np.sin(phi0), mu0, s0]

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
    if len(t_0) > 0 and t_0[-1] < t_pi[-2]:
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
    pkl_fn = PKL_FILE % s_c_str(s_c)

    if not os.path.exists(pkl_fn):
        print('Running sims, %s not found' % pkl_fn)
        assert num_threads > 0

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
        for mu, s, _, _ in outcome_trajs:
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
    plt.savefig('5outcomes%s.png' % s_c_str(s_c), dpi=400)
    plt.clf()

if __name__ == '__main__':
    I = np.radians(5)
    eps = 1e-3
    s0 = 10

    s_c_vals = [
        0.6,
        0.2,
        0.4,
        0.5,
        0.3,
        0.55,
        0.45,
        0.35,
        # 0.25,
        # 0.06,
    ]

    counts = []
    for s_c in s_c_vals:
        trajs = run_sim(I, eps, s_c, s0=s0, num_threads=N_THREADS)
        count = 0
        for outcome_trajs in trajs:
            for traj in outcome_trajs:
                mu = traj[0]
                if mu[-1] > 0.85:
                    count += 1
        counts.append(count)
        # plot_final_dists(I, s_c, s0, trajs)
    plt.plot(s_c_vals, np.array(counts) / (N_THREADS * N_PTS), 'bo')
    plt.xlabel(r'$s_c / \Omega_1$')
    plt.ylabel('CS1 Prob')
    plt.ylim([0, 1])
    plt.savefig('5probs', dpi=400)
    plt.clf()
