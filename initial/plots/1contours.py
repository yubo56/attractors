import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from utils import roots

def plot_H_for_eta(eta, I=np.radians(20)):
    '''
    Hamiltonian is H = 1/2 cos^2(theta) + eta * sin(phi)
    canonical variables are x = phi, y = cos(theta)
    '''
    _x = np.linspace(0, 2 * np.pi, 50)
    _y = np.linspace(-1, 1, 50)
    x = np.outer(_x, np.ones_like(_y))
    y = np.outer(np.ones_like(_x), _y)
    H = 0.5 * y**2 - eta * (
        y * np.cos(I) -
        np.sqrt(1 - y**2) * np.sin(I) * np.cos(x))
    plt.contour(x, y, H, cmap='RdBu_r')

    thetas, phis = roots(eta, I)
    colors = ['b', 'g', 'r', 'y'] if len(thetas) == 4 else ['b', 'y']
    for color, theta, phi in zip(colors, thetas, phis):
        phi_arr = [phi] if theta > 0 else [phi, phi + 2 * np.pi]
        for phi in phi_arr:
            plt.plot(phi, np.cos(theta), '%so' % color, markersize=16)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos \theta$')
    plt.savefig('1contour%s.png' % (('%.2f' % eta).replace('.', '_')))
    plt.clf()

if __name__ == '__main__':
    # Figures 3b-3e of Kassandra's paper
    for eta in [0.1, 0.5, 0.561, 2]:
        plot_H_for_eta(eta)
