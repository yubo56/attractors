'''
study dissipating disk problem, where eta shrinks over time
'''
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
from utils import to_cart, to_ang, roots, H, solve_ic_base,\
    get_etac

def plot_traj_colors(I, ret, filename):
    ''' scatter plot of (phi, mu) w/ colored time '''
    fix, ax1 = plt.subplots(1, 1)
    mu_lim = 0.6
    first_idx = np.where(abs(np.cos(q)) < mu_lim)[0][0]
    scat = ax1.scatter(phi[first_idx: ], np.cos(q[first_idx: ]),
                       c=t[first_idx: ], s=0.3, cmap='Spectral')
    fig.colorbar(scat, ax=ax1)
    ax1.set_xlabel(r'$\phi$')
    ax1.set_xlim([0, 2 * np.pi])
    ax1.set_xticks([0, np.pi, 2 * np.pi])
    ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax1.set_ylabel(r'$\cos\theta$')

    eta_f = etas[-1]
    phi_arr = np.linspace(0, 2 * np.pi, 201)
    z4 = eta_f * np.cos(I) / (1 - eta_f * np.sin(I))
    sep_diff = np.sqrt(2 * eta_f * np.sin(I) * (1 - np.cos(phi_arr)))
    ax1.plot(phi_arr, z4 + sep_diff, 'k', label='Final Sep')
    ax1.plot(phi_arr, z4 - sep_diff, 'k')
    ax1.legend()
    plt.savefig(filename, dpi=400)
    print('Saved', filename)
    plt.clf()

def get_areas_2(ret, num_pts=201):
    '''
    not get_areas in utils since this that assumes start -> ejection from sep
    '''
    t_events = ret.t_events[-1]
    x_events, _, z_events, eta_events = ret.sol(t_events)
    # phi = 0 means x < 0
    idx_0 = np.where(x_events < 0)[0]
    idx_pi = np.where(x_events > 0)[0]
    t_0 = t_events[idx_0]
    t_pi = t_events[idx_pi]
    areas = []
    t_areas = []
    # to ease casework, if starts circulating, ensure first event is a t_0
    if t_0[0] < t_pi[1] and t_0[0] > t_pi[0]:
        t_pi = t_pi[1: ]

    # can start either circulating or librating, but always eventually
    # transitions to librating about CS2, and ends up circulating again...

    # if starts circulating, integrate (1 - mu) dphi for continuity... so many
    # cases
    for t_0_i, t_0_f, t_pi_f in zip(t_0, t_0[1: ], t_pi[1: ]):
        if t_0_f < t_pi_f: # circulating
            t_vals = np.linspace(t_0_i, t_0_f, num_pts)
            t_areas.append((t_0_f + t_pi_f) / 2)

            x, y, z, _ = ret.sol(t_vals)
            q, phi = to_ang(x, y, z)
            dphi = np.gradient(np.unwrap(phi))
            areas.append(np.sum((1 - np.cos(q)) * dphi))
    # truncate list of times during initial circulation
    if len(areas):
        t_end_circ = t_areas[-1]
        t_0 = t_0[np.where(t_0 > t_end_circ)[0][0]: ]
        t_pi = t_pi[np.where(t_pi > t_end_circ)[0][0]: ]
    # now solve librating period
    t_end_lib = max(t_pi) if len(t_0) == 0 else t_0[0]
    t_pi_lib = t_pi[np.where(t_pi < t_end_lib)[0]]
    for t_pi_i, t_pi_f in zip(t_pi_lib[2::2], t_pi_lib[ :-2:2]):
        t_vals = np.linspace(t_pi_i, t_pi_f, num_pts)
        t_areas.append((t_pi_i + t_pi_f) / 2)

        x, y, z, _ = ret.sol(t_vals)
        q, phi = to_ang(x, y, z)
        dphi = np.gradient(np.unwrap(phi))
        areas.append(np.sum(np.cos(q) * dphi))
    # final circ period; note that t_0[0] < t_pi_fin[0] again
    t_pi_fin = t_pi[np.where(t_pi > t_end_lib)[0]]
    for t_0_i, t_0_f, t_pi_f in zip(t_0, t_0[1: ], t_pi_fin[1: ]):
        t_vals = np.linspace(t_0_i, t_0_f, num_pts)
        t_areas.append((t_0_f + t_pi_f) / 2)

        x, y, z, _ = ret.sol(t_vals)
        q, phi = to_ang(x, y, z)
        dphi = np.abs(np.gradient(np.unwrap(phi)))
        areas.append(np.sum(np.cos(q) * dphi))
    return t_end_lib, np.array(t_areas), np.array(areas)

def plot_traj(I, ret, filename, dq):
    '''
    plot mu(t) and zoomed in for a trajectory
    overplot area conservation
    '''
    # get initial area
    t_end_lib, t_areas, areas = get_areas_2(ret)
    a_init_int = areas[0]
    a_init_dq = 2 * np.pi * (1 - np.cos(dq))
    print('Areas (integrated/estimated): ', a_init_int, a_init_dq)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                            sharex=True,
                            gridspec_kw={'height_ratios': [3, 2, 2]})
    fig.subplots_adjust(hspace=0)
    t = ret.t
    etas = ret.y[3]
    q, phi = to_ang(*ret.y[0:3])

    # calculate separatrix min/max mu @ phi = pi
    idx4s = np.where(etas < get_etac(I))[0]
    eta4s = etas[idx4s]
    q4s, q2s = np.zeros_like(eta4s), np.zeros_like(eta4s)
    for i, eta in enumerate(eta4s):
        _, q2, _, q4 = roots(I, eta)
        q2s[i] = q2
        q4s[i] = q4
    mu4s = np.cos(q4s)
    H4s = H(I, eta4s, q4s, 0)
    # H = -mu**2 / 2 + eta * (mu * cos(I) - sin(I) * sin(q) * cos(phi))
    mu_min, mu_max = np.zeros_like(eta4s), np.zeros_like(eta4s)
    for i in range(len(eta4s)):
        # suppress dividebyzero/sqrt
        with np.errstate(all='ignore'):
            try:
                f = lambda mu: -mu**2 / 2 + eta4s[i] * (
                    mu * np.cos(I) + np.sin(I) * np.sqrt(1 - mu**2))\
                    - H4s[i]
                # mu_min always exists
                mu_min[i] = opt.bisect(f, -1, mu4s[i])
                # mu_max may not always exist
                guess = mu4s[i] + np.sqrt(4 * eta4s[i] * np.sin(I))
                res = opt.newton(f, guess)
                if abs(res - mu_min[i]) > 1e-5: # not the same root
                    mu_max[i] = res
            except:
                pass

    # plot trajectory's mu, separatrix's min/max mu, CS2
    eta_c_anal = (a_init_dq / 16)**2 / np.sin(I)
    eta_c_num = ret.sol(t_end_lib)[3]
    print('eta @ cross', eta_c_anal, eta_c_num)
    eta_c = eta_c_anal
    a_crossed = 2 * np.pi * eta_c * np.cos(I) + 8 * np.sqrt(eta_c * np.sin(I))
    a_crossed2 = 2 * np.pi * eta_c * np.cos(I) - 8 * np.sqrt(eta_c * np.sin(I))
    mu_f = a_crossed / (2 * np.pi)
    mu_f2 = a_crossed2 / (2 * np.pi)
    for ax in [ax1, ax2]:
        ax.plot(t, np.cos(q), 'k', label='Sim')
        max_valid_idxs = np.where(mu_max > 0)[0]
        ax.plot(t[idx4s][max_valid_idxs], mu_max[max_valid_idxs],
                 'g:', label='Sep')
        ax.plot(t[idx4s], mu_min, 'g:')
        ax.plot(t[idx4s], np.cos(q2s), 'b', label='CS2')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\cos\theta$')
        ax.legend()

        # predicted final mu
        ax.axhline(mu_f, c='r')
        ax.axhline(mu_f2, c='r')
    ax2.set_ylim([-1.5 * mu_f, 1.5 * mu_f])
    ax1.set_title(r'$I = %d^\circ$' % np.degrees(I))

    # plot areas
    t_pre_cross = t_areas[np.where(t_areas < t_end_lib)[0]]
    t_crossed = t_areas[np.where(t_areas > t_end_lib)[0]]
    ax3.plot(t_pre_cross, np.full_like(t_pre_cross, a_init_dq), 'r:')
    ax3.plot(t_crossed, np.full_like(t_crossed, a_crossed), 'r:')
    ax3.plot(t_crossed, np.full_like(t_crossed, a_crossed2), 'r:')
    ax3.set_ylabel('Enclosed Area')
    ax3.plot(t_areas, areas, 'bo', markersize=0.3)

    plt.savefig(filename, dpi=400)
    print('Saved', filename)
    plt.clf()

def plot_single(I, eps, tf, eta0, q0, filename, dq=0.3):
    y0 = [*to_cart(q0 + dq, 0), eta0]
    ret = solve_ic_base(I, eps, y0, tf)
    plot_traj(I, ret, filename, dq)

def fetch_ring(I, eta, dq, n_pts):
    q2, _ = roots(I, eta)
    tf = 4 * np.pi / eta # 2pi / eta = precession about CS2
    y0_ring = [*to_cart(q2 + dq, 0), eta]
    event = lambda t, y: (to_ang(*y[0:3])[1] % 2 * np.pi) - np.pi
    event.direction = +1
    ret_ring = solve_ic_base(I, 0, y0_ring, tf, events=[event])
    # either initially librating, so 2 t_events, or circulating, in which case
    # we sample over two orbits, not the worst thing
    t_ring = np.linspace(0, ret_ring.t_events[-1][2], n_pts)
    return ret_ring.sol(t_ring)

def sim_for_dq(I, eps=-1e-3, eta0_mult=10, etaf=1e-3, n_pts=10, dq=0.3,
               plot=False):
    '''
    for a small dq (dtheta) displacement below CS2, simulate forwards in time
    for a ring of ICs, look @ final mu values (terminate @ eta = 1e-3)
    '''
    eta0 = eta0_mult * get_etac(I)
    print('Running for', dq)

    # start w/ a ring of constant J_init. calc via eps=0 sim
    y0s = fetch_ring(I, eta0, dq, n_pts)

    # run for each traj
    final_mus = []
    for y0 in y0s.T:
        term_event = lambda t, y: y[3] - 1e-5
        term_event.terminal = True
        ret = solve_ic_base(I, eps, y0, np.inf, events=[term_event])
        q, phi = to_ang(*ret.y[0:3])
        if plot:
            plt.plot(phi[-1], np.cos(q[-1]), 'go', markersize=0.5)
            q0, phi0 = to_ang(*y0[0:3])
            plt.plot(phi0, np.cos(q0), 'ro', markersize=0.5)
        final_mus.append(np.mean(np.cos(q[-20: ])))
    if plot:
        plt.savefig('3single_%d.png' % np.degrees(I))
        plt.clf()
        print('Final Mus', final_mus)
    return final_mus

def sim_for_many(I, eps=-1e-3, eta0_mult=10, etaf=1e-3, n_pts=21, n_dqs=51):
    '''
    variable number of points per ring,
    starting from n_pts/4 to n_pts linearly over dqs
    '''
    I_deg = round(np.degrees(I))
    eta0 = eta0_mult * get_etac(I)
    dqs = np.linspace(0.05, 0.99 * np.pi / 2, n_dqs)
    n_pts_float = np.linspace(n_pts // 4, n_pts, n_dqs)

    PKL_FN = '3dat%d.pkl' % I_deg
    filename = '3_ensemble_%d.png' % I_deg
    title = r'$I = %d^\circ$' % I_deg
    if not os.path.exists(PKL_FN):
        p = Pool(4)
        args = [(I, eps, eta0_mult, etaf, int(round(n_pt)), dq)
                for dq, n_pt in zip(dqs, n_pts_float)]
        res_arr = p.starmap(sim_for_dq, args)
        with open(PKL_FN, 'wb') as f:
            pickle.dump(res_arr, f)
    else:
        with open(PKL_FN, 'rb') as f:
            res_arr = pickle.load(f)
    for dq, final_mus in zip(dqs, res_arr):
        final_q_deg = np.degrees(np.arccos(final_mus))
        plt.scatter(np.full_like(final_mus, np.degrees(dq)),
                    final_q_deg, c='b', s=0.5)
    plt.xlabel(r'$\mathrm{d}\theta$')
    plt.ylabel(r'$\theta_f$')
    plt.ylim([120, 0])
    plt.title(title)
    plt.savefig(filename, dpi=400)
    plt.clf()

    # plot all ICs
    inits_fn = '3_ensemble_inits%d.png' % I_deg
    if not os.path.exists(inits_fn):
        color_base = np.linspace(0, 1, n_dqs)
        q2, _ = roots(I, eta0)
        for dq, npts, color_num in zip(dqs, n_pts_float, color_base):
            y0s = fetch_ring(I, eta0, dq, int(round(npts)))
            c = [[1 - color_num, 0, color_num]]
            q, phi = to_ang(*y0s[0:3])
            plt.scatter(phi, np.cos(q), c=c, s=0.5)
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim([0, 2 * np.pi])
        plt.ylim([-1, 1])
        plt.scatter(np.pi, np.cos(q2), c='g')
        plt.savefig(inits_fn, dpi=400)
        plt.clf()

if __name__ == '__main__':
    I = np.radians(5)

    tf = 50000
    eta0 = 10 * get_etac(I)
    q2, _ = roots(I, eta0)
    # first one goes down, second TODO
    # plot_single(I, -3e-4, tf, eta0, q2, '3testo.png', dq=0.3)
    plot_single(I, -3.1e-4, tf, eta0, q2, '3testo2.png', dq=0.3)

    # sim_for_dq(I, dq=np.pi / 2, plot=True)
    # sim_for_many(I, eps=-3e-4, n_pts=101, n_dqs=51)
    # sim_for_many(np.radians(10), eps=-3e-4, n_pts=101, n_dqs=51)
    # sim_for_many(np.radians(20), eps=-3e-4, n_pts=101, n_dqs=51)
