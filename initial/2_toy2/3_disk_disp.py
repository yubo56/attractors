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
from utils import to_cart, to_ang, roots, H, get_mu4, solve_ic_base,\
    get_etac

def plot_traj(I, ret, filename, dphi):
    ''' convention: -np.pi / 2 < q4 < 0 '''
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(6, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    q, phi = to_ang(*ret.y[0:3])
    mu_lim = 0.1
    first_idx = np.where(abs(np.cos(q)) < mu_lim)[0][0]
    scat = ax1.scatter(phi[first_idx: ], np.cos(q[first_idx: ]),
                       c=ret.t[first_idx: ], s=0.3, cmap='Spectral')
    fig.colorbar(scat, ax=ax1)
    ax1.set_xlabel(r'$\phi$')
    ax1.set_xlim([0, 2 * np.pi])
    ax1.set_ylim([-mu_lim, mu_lim])
    ax1.set_xticks([0, np.pi, 2 * np.pi])
    ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax1.set_ylabel(r'$\cos\theta$')

    eta_f = ret.y[3,-1]
    phi_arr = np.linspace(0, 2 * np.pi, 201)
    z4 = eta_f * np.cos(I) / (1 - eta_f * np.sin(I))
    sep_diff = np.sqrt(2 * eta_f * np.sin(I) * (1 - np.cos(phi_arr)))
    ax1.plot(phi_arr, z4 + sep_diff, 'k', label='Final Sep')
    ax1.plot(phi_arr, z4 - sep_diff, 'k')
    ax1.legend()

    # calculate where sep crossing should be
    j_init = 2 * np.pi * (1 - np.cos(dphi))
    eta_c_j = (j_init / 16)**2 / np.sin(I)
    print(eta_c_j)

    ax2.semilogy(ret.t, ret.y[3], 'k')
    ax2.axhline(eta_c_j, c='r')
    plt.savefig(filename, dpi=400)
    plt.clf()

def plot_single(I, eps, tf, eta0, q0, filename, solve=solve_ic_base, dphi=0.2):
    y0 = [*to_cart(q0, dphi), eta0]
    ret = solve_ic_base(I, eps, y0, tf)
    plot_traj(I, ret, filename, dphi)

if __name__ == '__main__':
    I = np.radians(5)
    tf = 20000
    eta0 = 2 * get_etac(I)
    q2, _ = roots(I, eta0)

    plot_single(I, -1e-3, tf, eta0, q2, '3testo2.png', dphi=0.68)
    plot_single(I, -1e-3, tf, eta0, q2, '3testo3.png', dphi=0.67)
