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
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
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

    fig = plt.figure(figsize=(6, 5))
    qs, phis = roots(I, eta)

    # points below the separatrix, where do they converge to?
    below_convs = []
    # HACK HACK everything but CS1/2 are green as well
    colors = ['orange', 'tab:green', 'orange', 'tab:purple']
    for q, phi, conv_pts, c in zip(qs, phis, conv_data, colors):
        if conv_pts:
            q_plot, phi_plot = np.array(conv_pts).T
            below_convs.append(sum(is_below(I, eta, q_plot, phi_plot)))
            phis_plot = get_phis(q_plot, phi_plot)
            plt.plot(phis_plot,
                     np.cos(q_plot),
                     mfc=c, mec=c, marker='o', ls='',
                     markersize=2)
        else:
            below_convs.append(0)
        # plot_point(plt.gca(), q, '%so' % c, markersize=8)
    for idx, c in enumerate(colors[ :2]):
        plt.plot(0, -10, c=c, marker='o', ls='',
                 markersize=5, label='CS%d' % (idx + 1))
    plt.ylim([-1, 1])
    plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel(r'$\cos \theta_0$')
    plt.xlabel(r'$\phi_0$ (deg)')
    plt.xticks([0, np.pi, 2 * np.pi],
               ['0', r'$180$', r'$360$'])
    plt.text(np.pi, np.cos(qs[1]), 'II', backgroundcolor=(1, 1, 1, 0.9),
             ha='center', va='center')
    sepwidth = 2 * np.sqrt(eta * np.sin(I))
    plt.text(np.pi, np.cos(qs[1]) + 1.2 * sepwidth, 'I',
             backgroundcolor=(1, 1, 1, 0.9), ha='center', va='center')
    plt.text(np.pi, np.cos(qs[1]) - 1.2 * sepwidth, 'III',
             backgroundcolor=(1, 1, 1, 0.9), ha='center', va='center')

    x_grid, phi_grid = get_grids()
    H_grid = H(I, eta, x_grid, phi_grid)
    plt.contour(phi_grid,
                x_grid,
                H_grid,
                levels=[H(I, eta, np.cos(qs[3]), phis[3])],
                colors=['k'],
                linewidths=3,
                linestyles='solid')

    plt.tight_layout()
    plt.savefig('3stats%s.png' % prefix, dpi=400)

if __name__ == '__main__':
    # run_for_tide(tide=3e-2)
    # run_for_tide(tide=1e-2)
    # run_for_tide(tide=3e-3)
    # run_for_tide(tide=1e-3)
    # run_for_tide(tide=3e-4, NUM_RUNS=10000)
    # run_for_tide(tide=1e-4, NUM_RUNS=10000)
    # run_for_tide(tide=3e-5, NUM_RUNS=10000)
    run_for_tide(tide=3e-4, eta=0.2, NUM_RUNS=10000)
    # run_for_tide(tide=3e-4, eta=0.05, NUM_RUNS=10000)
    # run_for_tide(tide=3e-4, eta=0.025, NUM_RUNS=10000)
