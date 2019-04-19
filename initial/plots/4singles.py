'''
plot a few single plots, to understand separatrix hopping
fix tide=3e-4
'''

import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from utils import roots, to_cart, solve_ic, H, to_ang, is_below, get_phis

def plot_circs(ax, phi, q, color):
    ''' plot lines but do not plot across 2pi multiples '''
    idx = 0
    while idx < len(phi) - 1:
        idx_top = idx + 1
        while idx_top < len(phi) - 1 and\
                abs(phi[idx_top + 1] - phi[idx_top]) < 2:
            idx_top += 1

        phi_grad_s = np.sign(np.gradient(phi[idx: idx_top] % (2 * np.pi)))
        if idx_top - idx == 1 or all(phi_grad_s == phi_grad_s[0]):
            idx = idx_top + 1
            continue
        ax.plot(phi[idx: idx_top] % (2 * np.pi),
                np.cos(q[idx: idx_top]),
                color,
                linewidth=0.5)
        ax.plot(phi[idx], np.cos(q[idx]), '%so' % color, markersize=2)
        break

if __name__ == '__main__':
    filename = '3_53data.pkl'
    I = np.radians(20)
    eta = 0.1
    tide = 3e-4
    colors = ['b', 'r', 'g', 'k']
    tf = 500

    f, axs = plt.subplots(1, 2, sharey=True)
    f.subplots_adjust(wspace=0)
    with open(filename, 'rb') as dat_file:
        conv_data, _, mults = pickle.load(dat_file)

    # just for first two Cassini states
    for conv_pts, color, ax, idx in zip(conv_data[0: 2], colors, axs, range(2)):
        q_pts, phi_pts = np.array(conv_pts).T
        # pick out a point that starts at x in [-0.2, -0.1] and phi < 0.2, so
        # close to the separatrix
        first_idx = np.where(np.logical_and.reduce([
            np.cos(q_pts) < -0.1,
            np.cos(q_pts) > -0.2,
            get_phis(q_pts, phi_pts) < 0.2]))[0][3]
        s = to_cart(q_pts[first_idx], phi_pts[first_idx])
        _, _, sol = solve_ic(I, eta, tide, s, tf)

        x, y, z = sol
        q, phi = to_ang(x, y, z)
        phi = get_phis(q, phi)

        plot_circs(ax, phi, q, color)
        ax.set_title('Cassini State %d' % (idx + 1))

    axs[0].set_ylabel(r'$\cos \theta$')
    for ax in axs:
        ax.set_xlabel(r'$\phi$')
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.savefig('4singles.png')
