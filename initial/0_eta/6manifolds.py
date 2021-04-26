'''
plot stable/unstable manifolds
'''
from utils import solve_ic, roots, to_cart, to_ang, get_dydt, H, get_grids
from scipy.interpolate import interp1d
import os, pickle, lzma
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

from scipy.optimize import root
def plot_manifolds(eta):
    plt_fn = '6manifolds%s' % ('%.2f' % eta).replace('.', '_')

    fig = plt.figure(figsize=(6, 6))
    I = np.radians(20)
    tide = 1e-3
    q, _ = roots(I, eta)
    max_step = 0.05
    # improve CS4 location numerically
    _cs4 = np.array(to_cart(q[3], 0)) + \
        np.array([0, np.sin(tide / (eta * np.sin(I))), 0])
    dydt = lambda s: get_dydt(I, eta, tide)(0, s)
    cs4 = root(dydt, _cs4).x
    cs4_q, cs4_phi = to_ang(*cs4)

    def get_displaced(sign_q, sign_phi):
        # sign of sign_q is backwards from what is expected
        small = 0.001
        eigen = np.sqrt(eta * np.sin(I))
        return to_cart(cs4_q - small * sign_q,
                       cs4_phi + small * eigen * sign_phi)

    # backwards from CS4^0
    init = get_displaced(1, 1)
    pkl_fn = plt_fn + '.pkl'
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        _, t1, s1 = solve_ic(I, eta, tide, init,
                             -20000, rtol=1e-6, max_step=max_step)
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((t1, s1), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            t1, s1 = pickle.load(f)
    q1, _phi1 = to_ang(*s1)
    phi1 = np.unwrap(_phi1 + np.pi) - 2 * np.pi
    term1 = np.where(phi1 < 0)[0][0]
    plt.plot(phi1[ :term1], s1[2, :term1], 'b--',
             label=r'CS4$_{\rm L}^-$', linewidth=1.5, alpha=0.7)

    # forwards from CS4^0
    init = get_displaced(-1, 1)
    _, t2, s2 = solve_ic(I, eta, tide, init, 200, rtol=1e-6, max_step=max_step)
    q2, _phi2 = to_ang(*s2)
    phi2 = np.unwrap(_phi2 + np.pi) - 2 * np.pi
    phi2_grad = np.gradient(phi2) / np.gradient(t2)
    term2 = np.where(np.logical_and(
        abs(phi2_grad) < 5 * min(abs(phi2_grad)),
        phi2 < 1,
    ))[0][0]
    plt.plot(phi2[ :term2], s2[2, :term2], 'b',
             label=r'CS4$_{\rm L}^+$', linewidth=1.5, alpha=0.7)

    # backwards from CS4^1
    init = get_displaced(-1, -1)
    _, t3, s3 = solve_ic(I, eta, tide, init, -200, rtol=1e-6, max_step=max_step)
    q3, _phi3 = to_ang(*s3)
    phi3 = np.unwrap(_phi3 + np.pi)
    term3 = np.where(phi3 < 0)[0][0]
    plt.plot(phi3[ :term3], s3[2, :term3], 'k--',
             label=r'CS4$_{\rm R}^-$', linewidth=1.5, alpha=0.7)

    # forwards from CS4^1
    init = get_displaced(1, -1)
    _, t4, s4 = solve_ic(I, eta, tide, init, 200, rtol=1e-6, max_step=max_step)
    q4, _phi4 = to_ang(*s4)
    phi4 = np.unwrap(_phi4 + np.pi)
    term4 = np.where(phi4 < 0)[0][0]
    plt.plot(phi4[ :term4], s4[2, :term4], 'k',
             label=r'CS4$_{\rm R}^-$', linewidth=1.5, alpha=0.7)

    plt.xlim([0, 2 * np.pi])
    plt.xticks([0, np.pi, 2 * np.pi],
               ['0', r'$\pi$', r'$2\pi$'])

    ylim = plt.ylim()
    a1 = 0.4 # inner alpha
    a2 = 0.2 # Z3 alpha
    # fill Zone I with yellow
    plt.fill_between(phi4[ :term4], s4[2, :term4], np.ones_like(s4[2, :term4]),
                     facecolor='y', alpha=a1)
    top_interp = interp1d(phi4[ :term4], s4[2, :term4])
    # fill Zone III with yellow for now
    plt.fill_between(phi3[ :term3], s3[2, :term3], -np.ones_like(s3[2, :term3]),
                     color='y', alpha=a2)
    bot_interp = interp1d(phi3[ :term3], s3[2, :term3])
    # fill flow into zone I with yellow
    phi_inner = phi1[ :term1]
    mu_inner_outboundary = s1[2, :term1]
    rightmost = np.argmax(phi_inner)
    plt.fill_between(phi_inner[ :rightmost], mu_inner_outboundary[ :rightmost],
                     top_interp(phi_inner[ :rightmost]),
                     facecolor='y', alpha=a1)
    plt.fill_between(phi_inner[rightmost: ], mu_inner_outboundary[rightmost: ],
                     bot_interp(phi_inner[rightmost: ]),
                     facecolor='y', alpha=a1)
    phi_farright = np.linspace(phi_inner[rightmost], 2 * np.pi, 20)
    top_farright = top_interp(phi_farright)
    bot_farright = bot_interp(phi_farright)
    plt.fill_between(phi_farright, top_farright, bot_farright,
                     facecolor='y', alpha=a1)
    # fill Zone II with red
    inner_interp_above = interp1d(phi_inner[ :rightmost],
                                  mu_inner_outboundary[ :rightmost])
    inner_interp_below = interp1d(phi_inner[rightmost: ],
                                  mu_inner_outboundary[rightmost: ])
    phi_zone2 = phi_inner[ :rightmost]
    plt.fill_between(phi_zone2, inner_interp_above(phi_zone2),
                     inner_interp_below(phi_zone2),
                     facecolor='r', alpha=a1)
    # plot extension of probabilistic region
    # phi_shift = phi1 + 2 * np.pi
    # next_capture_line = np.where(np.logical_and(
    #     phi_shift > 0,
    #     phi_shift < 2 * np.pi))
    # plt.plot(phi_shift[next_capture_line], s1[2][next_capture_line],
    #          'b--', linewidth=1.5, alpha=0.7)
    # first, white it out so no double coloring
    # plt.fill_between(phi_shift[next_capture_line][1: -1],
    #                  s1[2][next_capture_line][1: -1],
    #                  bot_interp(phi_shift[next_capture_line][1: -1]),
    #                  facecolor='white')
    # plt.fill_between(phi_shift[next_capture_line][1: -1],
    #                  s1[2][next_capture_line][1: -1],
    #                  bot_interp(phi_shift[next_capture_line][1: -1]),
    #                  facecolor='r', alpha=a2)
    # just plot the line evolved backwards in time in Zone III, should be plenty
    # convincing
    for idx in range(1, 100):
        phi_shift = phi1 + 2 * idx * np.pi
        next_capture_line = np.where(np.logical_and(
            phi_shift > 0,
            phi_shift < 2 * np.pi))
        plt.plot(phi_shift[next_capture_line], s1[2][next_capture_line],
                 'r:', linewidth=1.5 * np.sqrt(2 / idx), alpha=a1)

    # plot separatrix
    x_grid, phi_grid = get_grids()
    H_grid = H(I, eta, x_grid, phi_grid)
    H_sep = H(I, eta, np.cos(cs4_q), cs4_phi - np.pi)
    plt.contour(phi_grid,
                x_grid,
                H_grid,
                levels=[H_sep],
                colors=['g'],
                linewidths=3,
                alpha=0.5,
                linestyles='solid')
    # overplot CS4 and location of H = H_sep - \Delta H_-
    plt.plot(0, np.cos(cs4_q), 'ko', markersize=10)
    plt.plot(2 * np.pi, np.cos(cs4_q), 'ko', markersize=10)
    plt.plot(phi1[term1], s1[2, term1], 'k*', markersize=15)
    plt.plot(phi1[term1] + 2 * np.pi, s1[2, term1], 'k*', markersize=15)
    plt.plot(phi3[term3], s3[2, term3], 'kx', markersize=15)
    plt.plot(phi3[term3] + 2 * np.pi, s3[2, term3], 'kx', markersize=15)

    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos \theta$')

    plt.ylim(ylim)
    plt.legend(fontsize=14, loc='center', ncol=2)
    plt.tight_layout()
    plt.savefig(plt_fn, dpi=300)
    plt.clf()


if __name__ == '__main__':
    # plot_manifolds(0.05)
    # plot_manifolds(0.1)
    plot_manifolds(0.2)
