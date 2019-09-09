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

def get_inital_area(ret):
    '''
    gets initial area enclosed by ret ODE solution
    initial area usually encloses pole, so area enclosed is given by (1 - mu)
    dphi
    '''
    n_pts = 200

    t_i, t_f = ret.t_events[-1][0], ret.t_events[-1][2]
    t = np.linspace(t_i, t_f, n_pts)
    x, y, z, _ = ret.sol(t)
    q, phi = to_ang(x, y, z)
    phi = np.unwrap(phi)
    dphi = np.gradient(phi)
    return np.sum((1 - np.cos(q)) * dphi)

def plot_traj(I, ret, filename, dq):
    ''' plot a few mu(t) '''
    # get initial area
    a_init = get_inital_area(ret)
    a_cross = 2 * np.pi * (1 - np.cos(dq))
    print('Areas (integrated/estimated): ', a_init, a_cross)

    fig, axs = plt.subplots(2, 1,
                            sharex=True,
                            gridspec_kw={'height_ratios': [3, 2]})
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
    for ax in axs:
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
        eta_c = (a_cross / 16)**2 / np.sin(I)
        mu_f = (eta_c * np.cos(I) + 8 * np.sqrt(eta_c * np.sin(I))) / (2 * np.pi)
        ax.axhline(mu_f, c='r')
    axs[1].set_ylim([-1.5 * mu_f, 1.5 * mu_f])

    axs[0].set_title(r'$I = %d^\circ$' % np.degrees(I))
    plt.savefig(filename, dpi=400)
    print('Saved', filename)
    plt.clf()

def plot_single(I, eps, tf, eta0, q0, filename, dq=0.3):
    y0 = [*to_cart(q0 + dq, 0), eta0]
    ret = solve_ic_base(I, eps, y0, tf)
    plot_traj(I, ret, filename, dq)

def max_theta_a_crit(I):
    '''
    for given I, compute maximum initial mutual inclination such
    that will encounter separatrix at some point (i.e. 4pi - A3_min)
    '''
    area_crit_frac = 1 - (1 + np.tan(np.radians(I))**(2/3))**(-3/2)
    # area_crit_frac = area / 4pi = 2pi * (1 - cos(q_max)) / 4pi
    return np.arccos(1 - 2 * area_crit_frac)

def sim_for_dq(I, eps=-1e-3, eta0_mult=10, etaf=1e-3, n_pts=10, dq=0.3,
               plot=False):
    '''
    for a small dq (dtheta) displacement below CS2, simulate forwards in time
    for a ring of ICs, look @ final mu values (terminate @ eta = 1e-3)
    '''
    eta0 = eta0_mult * get_etac(I)
    q2, _ = roots(I, eta0)
    print('Running for', dq)

    # start w/ a ring of constant J_init. calc via eps=0 sim
    tf = 4 * np.pi / eta0 # 2pi / eta0 = precession about CS2
    y0_ring = [*to_cart(q2 + dq, 0), eta0]
    event = lambda t, y: (to_ang(*y[0:3])[1] % 2 * np.pi) - np.pi
    event.direction = +1
    ret_ring = solve_ic_base(I, 0, y0_ring, tf, events=[event])
    t_ring = np.linspace(0, ret_ring.t_events[-1][1], n_pts)
    y0s = ret_ring.sol(t_ring)

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

def sim_for_many(I, eps=-1e-3, eta0_mult=10, etaf=1e-3, n_pts=10, n_dqs=20):
    I_deg = round(np.degrees(I))
    # dqs = np.linspace(0.05, 0.99 * max_theta_a_crit(I), n_dqs)
    dqs = np.linspace(0.05, 0.99 * np.pi / 2, n_dqs)
    PKL_FN = '3dat%d.pkl' % I_deg
    filename = '3_ensemble_%d.png' % I_deg
    title = r'$I = %d^\circ$' % I_deg
    if not os.path.exists(PKL_FN):
        p = Pool(4)
        args = [(I, eps, eta0_mult, etaf, n_pts, dq) for dq in dqs]
        res_arr = p.starmap(sim_for_dq, args)
        with open(PKL_FN, 'wb') as f:
            pickle.dump(res_arr, f)
    else:
        with open(PKL_FN, 'rb') as f:
            res_arr = pickle.load(f)
    for dq, final_mus in zip(dqs, res_arr):
        plt.scatter(dq * np.ones_like(final_mus), final_mus)
    plt.xlabel(r'$\mathrm{d}\theta$')
    plt.ylabel(r'$\mu_f$')
    plt.title(title)
    plt.savefig(filename)

if __name__ == '__main__':
    I = np.radians(5)
    tf = 15000
    eta0 = 10 * get_etac(I)
    q2, _ = roots(I, eta0)

    # plot_single(I, -1e-3, tf, eta0, q2, '3testo.png', dq=0.3)
    # sim_for_dq(I, dq=max_theta_a_crit(I), plot=True)
    sim_for_many(I, n_pts=21, n_dqs=51)
