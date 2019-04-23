'''
Take poincare map of separatrix-hopping trajectories, checking cos(theta) values
when phi = 0
'''

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from utils import to_cart, get_dydt, to_ang, is_below, get_phis, get_phi,\
    roots, is_inside
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def get_poincare(t, sol, interp):
    '''
    cos(theta, phi = 0) is equivalent to z(y = 0), x = -1
    '''
    x, y, z = sol

    cross_zs = []
    for idx in range(len(y) - 1):
        if y[idx] < 0 and y[idx + 1] > 0 and x[idx] > 0:
            y_interp = lambda t: interp(t)[1]
            t0 = brentq(y_interp, t[idx], t[idx + 1])
            z_exact = interp(t0)[2]
            cross_zs.append(z_exact)
    return cross_zs

def plot_map(eta, nbins, n_thresh=10):
    suffix = str(eta).replace('.', '_')
    filename = '3stats3_5_%s.pkl' % suffix
    out_file = '5poincare_%s.pkl' % suffix
    I = np.radians(20)
    tide = 3e-4
    tf = 5000
    max_sim = 20
    q, _ = roots(I, eta)

    if not os.path.exists(out_file):
        with open(filename, 'rb') as dat_file:
            conv_data, zeros, _ = pickle.load(dat_file)

        q2, phi2 = np.array(conv_data[1]).T
        hop_idxs = np.where(is_below(I, eta, q2, phi2))[0]

        q_zeros = []
        phi_zeros = []
        for (q0, _phi0), (x, y, z) in zeros:
            phi0 = get_phi(q0, _phi0)
            qf, _phif = to_ang(x, y, z)
            if is_inside(I, eta, [qf], [_phif]) and\
                    is_below(I, eta, [q0], [_phi0]):
                q_zeros.append(q0)
                phi_zeros.append(phi0)

        # assemble inits from both converged-to-2 and zeros inside
        inits = zip(
            list(q2[hop_idxs]) + q_zeros,
            list(phi2[hop_idxs]) + phi_zeros)

        q_poincare = []
        dydt = get_dydt(I, eta, tide)
        for sim_num, angs in enumerate(inits):
            s = to_cart(*angs)
            for i in range(max_sim):
                ret = solve_ivp(dydt, [0, tf], s, rtol=1e-6, dense_output=True)
                s = ret.y[:, -1]
                if s[2] > q[3]: # if z coord > CS4's z coord
                    break
            print('Finishing sim num %d/%d' %
                  (sim_num, len(hop_idxs) + len(q_zeros)))

            if s[2] > q[3]:
                q_poincare.append(get_poincare(ret.t, ret.y, ret.sol))
            else:
                print('(%.5f, %.5f, %.5f) did not converge' %
                      (s[0], s[1], s[2]))

        with open(out_file, mode='wb') as out:
            pickle.dump(q_poincare, out)

    else:
        with open(out_file, mode='rb') as in_file:
            q_poincare = pickle.load(in_file)

    # lf = -1
    lf = np.cos(q[3]) - 4 * 50 * tide
    q_arr = np.array([l[-1: ] for l in q_poincare
                      if l and l[-1] > lf])
    q2_arr = np.array([l[-2:-1] for l in q_poincare
                       if len(l) >= 2 and l[-1] > lf])

    # compute the smallest element in the first bin w/ a reasonable number
    def plot_and_get_minmax(q_arr, label):
        n, bins, _ = plt.hist(q_arr.flatten(), bins=nbins, label=label)
        first_idx = np.where(n > n_thresh)[0][0]
        return min([val for val in q_arr if val > bins[first_idx]]),\
            q_arr.max(), max(n), sum(n[first_idx: ])

    min2, max2, max_ct2, n_tot2 = plot_and_get_minmax(q2_arr, 'Penult.')
    min1, max1, max_ct1, n_tot1 = plot_and_get_minmax(q_arr, 'Last')
    xmin = min(min1, min2)
    plt.xlim([xmin - 0.2 * abs(xmin), 1.1 * (np.cos(q[3]) - xmin) + xmin])
    plt.ylim([0, 1.2 * max([max_ct1, max_ct2])])
    plt.text(xmin, 1.1 * max([max_ct1, max_ct2]),
             (r'Last $\in [%.4f, %.4f]\times 10^{-2}$' % (min1, max1)),
             fontsize=12)
    plt.text(xmin, 1.03 * max([max_ct1, max_ct2]),
             (r'Penult $\in [%.4f, %.4f]\times 10^{-2}$' % (min2, max2)),
             fontsize=12)
    plt.title(r'$(\eta, \epsilon, N) = (%s, %s, %d/%d/%d)$' %
              (str(eta), r'3 \times 10^{-4}', n_tot2, n_tot1, len(q_poincare)))
    plt.xlabel(r'$z$')
    plt.ylabel(r'$N$')
    plt.axvline(x=np.cos(q[3]), color='r', linewidth=4)
    plt.legend(fontsize=10, loc='upper right')
    plt.savefig('5poincare_%s.png' % suffix)
    plt.clf()

if __name__ == '__main__':
    plot_map(eta=0.2, nbins=40)
    plot_map(eta=0.1, nbins=25, n_thresh=5)
    plot_map(eta=0.05, nbins=25, n_thresh=3)
    plot_map(eta=0.025, nbins=10, n_thresh=3)
