import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
from utils import roots, get_four_subplots, plot_point, H, get_grids, get_etac,\
    get_sep_area

letters = ['a', 'b', 'c', 'd']
def plot_H_for_eta(f, ax, eta, I, idx):
    '''
    Hamiltonian is H = 1/2 cos^2(theta) + eta * sin(phi)
    canonical variables are (phi, x = cos(theta))
    '''
    eta_c = get_etac(I)
    x_grid, phi_grid = get_grids()
    H_grid = H(I, eta, x_grid, phi_grid)
    cset = ax.contour(phi_grid, x_grid, H_grid,
                      cmap='RdBu_r', linewidths=0.8, levels=5)

    thetas, phis = roots(I, eta)
    colors = ['y', 'r', 'm', 'c'] if len(thetas) == 4 else ['r', 'm']
    for color, theta, phi in zip(colors, thetas, phis):
        plot_point(ax, theta, '%so' % color, markersize=6)

    shade = '0.8' # shade interior of separatrix
    font_height = 0.15 # ~font height so y loc is center of text, not base
    if len(thetas) == 4:
        H4 = H(I, eta, np.cos(thetas[3]), phis[3])
        ax.contour(phi_grid,
                   x_grid,
                   H_grid,
                   levels=[H4],
                   colors=['k'],
                   linewidths=2.0,
                   linestyles='solid')
        ax.contourf(phi_grid,
                    x_grid,
                    H_grid,
                    levels=[H4, np.max(H_grid)],
                    colors=[shade])

        if eta < 0.8 * eta_c:
            # if Zone I is sufficiently large, place it inside zone
            ax.text(0.3, np.cos(thetas[3]) + 0.4 - font_height, 'I')
        else:
            # else draw an arrow
            y = (2 * np.cos(thetas[0]) + np.cos(thetas[3])) / 3
            x = 0.2
            dx = np.pi / 3
            ax.arrow(x + dx, y, -dx, 0,
                     width=0.006, head_width=0.056, head_length=0.08)
            ax.text(x + dx + 0.1, y - font_height, 'I')
        ax.text(2 * np.pi / 3, np.cos(thetas[3]) + 0.05 - font_height, 'II')
        ax.text(0.3, np.cos(thetas[3]) - 0.6, 'III')
    else:
        # estimate the location of zone II
        # fractional area (/4pi) enclosed by separatrix @ appearance
        area_frac = 1 - (1 + np.tan(I)**(2/3))**(-3/2)
        # estimate location of separatrix curve via 2pi(1 - cos(dq)) = A_crit
        dq = np.arccos(1 - 2 * area_frac)
        print(np.cos(thetas[0]), thetas[0], dq, area_frac)
        H_sep = H(I, eta, np.cos(thetas[0] + dq), np.pi)
        ax.contour(phi_grid,
                   x_grid,
                   H_grid,
                   levels=[H_sep],
                   colors=['k'],
                   linewidths=0.5,
                   linestyles='dashed')
        ax.contourf(phi_grid,
                    x_grid,
                    H_grid,
                    levels=[H_sep, np.max(H_grid)],
                    colors=[shade])
        ax.text(2 * np.pi / 3, np.cos(thetas[0]) - 0.1 - font_height, 'II')
        ax.text(0.3, np.cos(thetas[1]) + 0.6, 'III')

    ax.set_title(r'(%s) $\eta = %.2f$' % (letters[idx], eta), fontsize=14)

    plt.suptitle(r'$I = %d^\circ, \eta_c = %.3f$' %
                 (np.degrees(I), eta_c))

if __name__ == '__main__':
    I = np.radians(5)
    f, axs = get_four_subplots()
    # Figures 3b-3e of Kassandra's paper
    for idx, (ax, eta) in enumerate(zip(axs, [0.1, 0.5, 0.561, 2])):
        plot_H_for_eta(f, ax, eta, I, idx)

    plt.suptitle(r'$\eta_c = %.3f$' % get_etac(I))
    plt.savefig('1contours.png', dpi=400)
    plt.clf()

    f, axs = get_four_subplots()
    for idx, (ax, eta) in enumerate(zip(axs, [2, 0.73, 0.4, 0.1])):
        plot_H_for_eta(f, ax, eta, I, idx)

    plt.savefig('1contours_flip.png', dpi=400)
    plt.clf()

    # f, ax = plt.subplots(1, 1)
    # ax.set_ylabel(r'$\cos \theta$')
    # ax.set_xlabel(r'$\phi$')
    # ax.set_xticks([0, np.pi, 2 * np.pi])
    # ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    # ax.set_ylabel(r'$\cos \theta$')
    # ax.set_xlabel(r'$\phi$')
    # plot_H_for_eta(f, ax, 0.2, I)
    # plt.savefig('1contours_02.png', dpi=400)
