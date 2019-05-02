'''
randomly generates a bunch of points and sees which fixed point they evolve to

Results:
38.7% phase space below/above sep, 22.6% inside
26% go to 2, 74% go to 1. Below, 8% go to 2, 92% go to 1
'''
import os
import pickle
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from utils import roots, to_cart, to_ang, solve_ic, get_four_subplots,\
    plot_point, H, get_grids, get_phi, get_phis, is_below, get_sep_area
import matplotlib.lines as mlines

DAT_FN_TEMP = '3stats%s.pkl'
TAIL_LEN = 20 # number of points on the tail of the traj to determine sink
RETRIES = 50 # number of times to retry for convergence

def get_rand_init(N=1):
    '''
    randomly generate some initial x, y, z uniformly from sphere
    uniformly on sphere = uniform in np.cos(theta), phi
    '''
    _xs, _phis = np.random.rand(2, N)
    qs = np.arccos(2 * _xs - 1)
    phis = 2 * np.pi * _phis
    xs, ys, zs = to_cart(qs, phis)
    return np.array([qs, phis, xs, ys, zs]).T

def get_dists(I, eta, tail):
    eps = 1e-1 # squared distance to consider traj in sink
    qs, phis = roots(I, eta)
    cass_xs, cass_ys, cass_zs = to_cart(qs, phis)
    dists = np.array([
        np.mean((tail[0] - x)**2 + (tail[1] - y)**2 + (tail[2] - z)**2)
        for x, y, z in zip(cass_xs, cass_ys, cass_zs)
    ])
    sink_idxs = np.where(dists < eps)[0]
    return sink_idxs, dists

def get_sinks(I, eta, trajs, inits):
    ''' get which Cassini state each of trajs ended up at '''

    inits_by_cass = [[] for _ in range(4)] # one for each Cassini point

    zero_sinks = []
    mult_sinks = []
    for init, traj in zip(inits, trajs):
        tail = traj[ :, -TAIL_LEN: ]
        sink_idxs, dists = get_dists(I, eta, tail)

        if len(sink_idxs) == 0:
            zero_sinks.append((init, tail[ :, -1]))
        elif len(sink_idxs) > 1:
            mult_sinks.append((init, tail[ :, -1]))
        else:
            inits_by_cass[sink_idxs[0]].append(init)

    return inits_by_cass, zero_sinks, mult_sinks

def run_for_init(idx, s, params):
    ''' mapper function '''
    x, y, z = s
    I, tf, eta, tide = params
    if idx % 100 == 0:
        print(idx)

    _, _, y = solve_ic(I, eta, tide, [x, y, z], tf)
    for i in range(RETRIES):
        tail = y[:, -TAIL_LEN: ]
        sink_idxs, dists = get_dists(I, eta, tail)
        if len(sink_idxs) == 1:
            break
        _, _, y = solve_ic(I, eta, tide, tail[:, -1], tf)
    return y

def run_for_tide(tide=1e-3,
                 eta=0.1,
                 I=np.radians(20),
                 T_F=2000,
                 NUM_RUNS=20000):

    prefix = (str(np.round(-np.log10(tide), 1)) + '_' + str(eta))\
     .replace('.', '_')
    dat_fn = DAT_FN_TEMP % prefix

    if not os.path.exists(dat_fn):
        print('%s not found, running' % dat_fn)
        inits = get_rand_init(NUM_RUNS)

        p = Pool(4)
        params = (I, T_F, eta, tide)
        trajs = p.starmap(run_for_init,
                          [(i, s, params) for i, s in enumerate(inits[:, 2: ])])

        conv_data, zeros, mults = get_sinks(I, eta, trajs, inits[ :, 0:2])

        with open(dat_fn, 'wb') as dat_file:
            pickle.dump((conv_data, zeros, mults), dat_file)
    else:
        print('%s found, loading' % dat_fn)
        with open(dat_fn, 'rb') as dat_file:
            conv_data, zeros, mults = pickle.load(dat_file)

    f, axs = get_four_subplots()
    qs, phis = roots(I, eta)

    # points below the separatrix, where do they converge to?
    below_convs = []
    for q, phi, ax, conv_pts in zip(qs, phis, axs, conv_data):
        if conv_pts:
            q_plot, phi_plot = np.array(conv_pts).T
            below_convs.append(sum(is_below(I, eta, q_plot, phi_plot)))
            phis_plot = get_phis(q_plot, phi_plot)
            ax.plot(phis_plot,
                    np.cos(q_plot),
                    'ko',
                    markersize=0.5)
        else:
            below_convs.append(0)
        plot_point(ax, q, 'ro', markersize=8)

        x_grid, phi_grid = get_grids()
        H_grid = H(I, eta, x_grid, phi_grid)
        ax.contour(phi_grid,
                   x_grid,
                   H_grid,
                   levels=[H(I, eta, np.cos(qs[3]), phis[3])],
                   colors=['k'],
                   linewidths=1.6)

    if len(zeros):
        q_zeros, phi_zeros = np.array([tup[0] for tup in zeros]).T
        belows_zero = sum(is_below(I, eta, q_zeros, phi_zeros))
    else:
        belows_zero = 0

    tot_pts = sum([len(pts) for pts in conv_data]) + len(zeros)
    tot_belows = sum(below_convs) + belows_zero
    for q, phi, belows, conv_pts, ax in\
            zip(qs, phis, below_convs, conv_data, axs):
        ax.set_title(
            r'$(\phi_0, \theta_0, P_T, P_<) = (%.2f, %.2f) (%.3f, %.3f)$'
            % (get_phi(q), q, len(conv_pts) / tot_pts, belows / tot_belows),
            fontsize=8)

    for (q0, _phi0), (x, y, z) in zeros:
        phi0 = get_phi(q0, _phi0)
        axs[3].plot(phi0,
                    np.cos(q0),
                    'bo',
                    markersize=0.5)
        q, _phi = to_ang(x, y, z)
        phi = get_phi(q, _phi)
        axs[3].plot(phi,
                    np.cos(q),
                   'mo',
                    markersize=4)
        axs[3].set_title(
            r'$(\phi_0, \theta_0, P_T, P_<) = (%.2f, %.2f) (%.3f, %.3f)$'
            % (get_phi(qs[3]), qs[3],
               len(zeros) / tot_pts, belows_zero / tot_belows),
            fontsize=8)
        ln1 = mlines.Line2D([], [], color='b', marker='o', markersize=0.5,
                            linewidth=0, label='Init')
        ln2 = mlines.Line2D([], [], color='m', marker='o', markersize=4,
                            linewidth=0, label=r'Final')
        axs[3].legend(handles=[ln1, ln2], fontsize=6, loc='upper left')

    plt.suptitle((r'(I, $\eta$, $\epsilon$, N)=($%d^\circ$, %.1f, %.1e, %d)' +
                 r', A = %.3f')
                 % (np.degrees(I), eta, tide, NUM_RUNS, get_sep_area(eta, I)),
                 fontsize=10)
    plt.savefig('3stats%s.png' % prefix, dpi=400)

if __name__ == '__main__':
    run_for_tide(tide=3e-2)
    run_for_tide(tide=1e-2)
    run_for_tide(tide=3e-3)
    run_for_tide(tide=1e-3)
    run_for_tide(tide=3e-4, NUM_RUNS=10000)
    run_for_tide(tide=1e-4, NUM_RUNS=10000)
    run_for_tide(tide=3e-5, NUM_RUNS=10000)
    run_for_tide(tide=3e-4, eta=0.2, NUM_RUNS=10000)
    run_for_tide(tide=3e-4, eta=0.05, NUM_RUNS=10000)
    run_for_tide(tide=3e-4, eta=0.025, NUM_RUNS=10000)
