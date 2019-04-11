import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from utils import roots

def plot_H_for_eta(ax, eta, I=np.radians(20)):
    '''
    Hamiltonian is H = 1/2 cos^2(theta) + eta * sin(phi)
    canonical variables are (phi, x = cos(theta))
    '''
    def H(x, phi):
        return 0.5 * x**2 - eta * (
            x * np.cos(I) -
            np.sqrt(1 - x**2) * np.sin(I) * np.cos(phi))

    _phi = np.linspace(0, 2 * np.pi, 50)
    _x = np.linspace(-1, 1, 50)
    phi_grid = np.outer(_phi, np.ones_like(_x))
    x_grid = np.outer(np.ones_like(_phi), _x)
    H_grid = H(x_grid, phi_grid)
    ax.contour(phi_grid, x_grid, H_grid, cmap='RdBu_r', linewidths=0.8)

    thetas, phis = roots(eta, I)
    colors = ['b', 'g', 'r', 'y'] if len(thetas) == 4 else ['b', 'y']
    for color, theta, phi in zip(colors, thetas, phis):
        phi_arr = [phi] if theta > 0 else [phi, phi + 2 * np.pi]
        for phi_plot in phi_arr:
            ax.plot(phi_plot, np.cos(theta), '%so' % color, markersize=6)
    ax.set_title(r'(I, $\eta$)=($%d^\circ$, %.3f)' % (np.degrees(I), eta),
                 fontsize=8)
    ax.set_xticks([0, np.pi, 2 * np.pi])

    if len(thetas) == 4:
        # plot separatrix, which is H at q4
        ax.contour(phi_grid,
                   x_grid,
                   H_grid,
                   levels=[H(np.cos(thetas[3]), phis[3])],
                   colors=['k'],
                   linewidths=1.6)

if __name__ == '__main__':
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    f.subplots_adjust(wspace=0)
    # Figures 3b-3e of Kassandra's paper
    for ax, eta in zip([ax1, ax2, ax3, ax4], [0.1, 0.5, 0.561, 2]):
        plot_H_for_eta(ax, eta)
    ax1.set_ylabel(r'$\cos \theta$')
    ax3.set_xlabel(r'$\phi$')
    ax3.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax3.set_ylabel(r'$\cos \theta$')
    ax4.set_xlabel(r'$\phi$')
    plt.savefig('1contours.png', dpi=400)
    plt.clf()
