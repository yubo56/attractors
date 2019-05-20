import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from utils import roots, get_four_subplots, plot_point, H, get_grids, get_etac,\
    get_sep_area

def plot_H_for_eta(f, ax, eta, I):
    '''
    Hamiltonian is H = 1/2 cos^2(theta) + eta * sin(phi)
    canonical variables are (phi, x = cos(theta))
    '''
    x_grid, phi_grid = get_grids()
    H_grid = H(I, eta, x_grid, phi_grid)
    cset = ax.contour(phi_grid, x_grid, H_grid,
                      cmap='RdBu_r', linewidths=0.8, levels=20)

    thetas, phis = roots(I, eta)
    colors = ['r', 'm', 'g', 'c'] if len(thetas) == 4 else ['m', 'g']
    for color, theta, phi in zip(colors, thetas, phis):
        plot_point(ax, theta, '%so' % color, markersize=6)

    if len(thetas) == 4:
        ax.contour(phi_grid,
                   x_grid,
                   H_grid,
                   levels=[H(I, eta, np.cos(thetas[3]), phis[3])],
                   colors=['k'],
                   linewidths=1.6)

        title_str = r' $A_{in} = %.3f$' % get_sep_area(eta, I)
    else:
        title_str = ''

    ax.set_title(r'$(I, \eta)=(%d^\circ, %.3f)$%s'
                 % (np.degrees(I), eta, title_str), fontsize=8)

    cb = f.colorbar(cset, ax=ax, fraction=0.07, pad=0.05)
    cb.set_ticks([])
    cb.set_ticklabels([])

if __name__ == '__main__':
    f, axs = get_four_subplots()
    I = np.radians(20)
    # Figures 3b-3e of Kassandra's paper
    for ax, eta in zip(axs, [0.1, 0.5, 0.561, 2]):
        plot_H_for_eta(f, ax, eta, I)

    plt.suptitle(r'$\eta_c = %.3f$' % get_etac(I))
    plt.savefig('1contours.png', dpi=400)
    plt.clf()

    f, ax = plt.subplots(1, 1)
    ax.set_ylabel(r'$\cos \theta$')
    ax.set_xlabel(r'$\phi$')
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.set_ylabel(r'$\cos \theta$')
    ax.set_xlabel(r'$\phi$')
    plot_H_for_eta(f, ax, 0.1, I)
    plt.savefig('1contours_02.png', dpi=400)
