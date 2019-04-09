import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
import scipy.optimize as opt

def roots(eta, I):
    ''' returns theta roots from EOM '''
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    try:
        roots = []
        # try to root find w/ 4 intervals
        args_4_arr = [
            (-np.pi, -np.pi / 2),
            (-np.pi/2, -np.pi / 4),
            (-np.pi / 4, 0),
            (0, np.pi),
        ]
        for args in args_4_arr:
            roots.append(opt.brentq(f, *args))
        return ['b', 'g', 'r', 'y'], np.array(roots)

    # throws ValueError when no zero in brackets
    except ValueError:
        roots = []
        # if not, fallback to only 2
        args_2_arr = [
            (-np.pi, 0),
            (0, np.pi),
        ]
        for args in args_2_arr:
            roots.append(opt.brentq(f, *args))
        return ['b', 'y'], np.array(roots)

def plot_H_for_eta(eta, I=math.radians(20)):
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

    colors, fixed_thetas = roots(eta, I)
    for color, theta in zip(colors, fixed_thetas):
        phis = [np.pi] if theta > 0 else [0, 2 * np.pi]
        for phi in phis:
            plt.plot(phi, np.cos(theta), '%so' % color, markersize=16)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos \theta$')
    plt.savefig('contour%s.png' % (('%.2f' % eta).replace('.', '_')))
    plt.clf()

if __name__ == '__main__':
    # Figures 3b-3e of Kassandra's paper
    for eta in [0.1, 0.5, 0.561, 2]:
        plot_H_for_eta(eta)
