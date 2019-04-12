import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from utils import roots, get_four_subplots, plot_point, H, get_grids

def plot_H_for_eta(ax, eta, I=np.radians(20)):
    '''
    Hamiltonian is H = 1/2 cos^2(theta) + eta * sin(phi)
    canonical variables are (phi, x = cos(theta))
    '''
    x_grid, phi_grid = get_grids()
    H_grid = H(I, eta, x_grid, phi_grid)
    ax.contour(phi_grid, x_grid, H_grid, cmap='RdBu_r', linewidths=0.8)

    thetas, phis = roots(I, eta)
    colors = ['r', 'm', 'g', 'c'] if len(thetas) == 4 else ['m', 'g']
    for color, theta, phi in zip(colors, thetas, phis):
        plot_point(ax, theta, '%so' % color, markersize=6)

    ax.set_title(r'(I, $\eta$)=($%d^\circ$, %.3f)' % (np.degrees(I), eta),
                 fontsize=8)

    if len(thetas) == 4:
        ax.contour(phi_grid,
                   x_grid,
                   H_grid,
                   levels=[H(I, eta, np.cos(thetas[3]), phis[3])],
                   colors=['k'],
                   linewidths=1.6)

if __name__ == '__main__':
    f, axs = get_four_subplots()
    # Figures 3b-3e of Kassandra's paper
    for ax, eta in zip(axs, [0.1, 0.5, 0.561, 2]):
        plot_H_for_eta(ax, eta)

    plt.savefig('1contours.png', dpi=400)
