import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from scipy.integrate import solve_ivp
from utils import roots

def solve_ic(eta, I, theta0, phi0, tide=0):
    '''
    solves ivp at given (eta, I, tide) for IC (phi0, theta0)
    '''
    def dydt(t, y):
        theta, phi = y
        return [
            -eta * np.sin(I) * np.sin(theta) * np.sin(phi),
            np.cos(theta) - eta * (
                np.sin(theta) * np.cos(I) +
                np.sin(I) * np.cos(theta) * np.cos(phi)) / np.sin(theta)
        ]
    tf = 150
    ret = solve_ivp(dydt, [0, tf], [theta0, phi0])
    return ret.t, ret.y

if __name__ == '__main__':
    eta = 0.1
    I = np.radians(20)
    thetas, phis = roots(eta, I)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

    for theta0, phi0, ax in zip(thetas, phis, [ax1, ax2, ax3, ax4]):
        t, y = solve_ic(eta, I, theta0, phi0)

        ax.plot(y[1, :] % (2 * np.pi), np.cos(y[0, :]), 'bo', markersize=4)
        ax.set_title('Init: (%.3f, %.3f)' % (phi0, np.cos(theta0)), fontsize=8)
    plt.savefig('2evolution.png', dpi=400)
