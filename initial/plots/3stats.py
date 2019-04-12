'''
randomly generates a bunch of points and sees which fixed point they evolve to
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
    plot_point, H, get_grids

DAT_FN_TEMP = '%s3data.pkl'
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
            zero_sinks.append(tail[ :, -1])
        elif len(sink_idxs) > 1:
            mult_sinks.append(tail[ :, -1])
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
                 NUM_RUNS=5000):

    prefix = str(np.round(-np.log10(tide), 1)).replace('.', '_')
    if prefix == '3_0':
        prefix = ''
    DAT_FN = DAT_FN_TEMP % prefix

    if not os.path.exists(DAT_FN):
        print('%s not found, running' % DAT_FN)
        inits = get_rand_init(NUM_RUNS)

        p = Pool(4)
        params = (I, T_F, eta, tide)
        trajs = p.starmap(run_for_init,
                          [(i, s, params) for i, s in enumerate(inits[:, 2: ])])

        conv_data, zeros, mults = get_sinks(I, eta, trajs, inits[ :, 0:2])

        with open(DAT_FN, 'wb') as dat_file:
            pickle.dump((conv_data, zeros, mults), dat_file)
    else:
        print('%s found, loading' % DAT_FN)
        with open(DAT_FN, 'rb') as dat_file:
            conv_data, zeros, mults = pickle.load(dat_file)

    f, axs = get_four_subplots()
    qs, phis = roots(I, eta)
    for q, phi, ax, conv_pts in zip(qs, phis, axs, conv_data):
        ax.set_title(r'$(\phi_0, \theta_0) = (%.3f, %.3f)$'
                     % (phi, q), fontsize=8)
        if conv_pts:
            q_plot, phi_plot = np.array(conv_pts).T
            ax.plot((phi_plot + 2 * np.pi) % (2 * np.pi),
                    np.cos(q_plot),
                    'ko',
                    markersize=0.5)
        plot_point(ax, q, 'ro', markersize=8)

        x_grid, phi_grid = get_grids()
        H_grid = H(I, eta, x_grid, phi_grid)
        ax.contour(phi_grid,
                   x_grid,
                   H_grid,
                   levels=[H(I, eta, np.cos(qs[3]), phis[3])],
                   colors=['k'],
                   linewidths=1.6)
    for arr, marker in zip([zeros, mults], ['go', 'co']):
        for x, y, z in arr:
            q, phi = to_ang(x, y, z)
            axs[3].plot((phi + 2 * np.pi) % (2 * np.pi),
                     np.cos(q),
                     marker,
                     markersize=0.5)

    plt.savefig('%s3stats.png' % prefix, dpi=400)

if __name__ == '__main__':
    run_for_tide()
    run_for_tide(tide=1e-2)
    run_for_tide(tide=3e-2)
    run_for_tide(tide=3e-3)
