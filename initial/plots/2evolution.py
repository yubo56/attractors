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
    tf = 50
    ret = solve_ivp(dydt, [0, tf], [theta0, phi0])
    return ret.t, ret.y


if __name__ == '__main__':
    eta = 0.1
    I = np.radians(20)
    thetas, phis = roots(eta, I)

    for theta0, phi0, idx in zip(thetas, phis, range(len(thetas))):
        t, y = solve_ic(eta, I, theta0, phi0)

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0)
        ax1.plot(t, np.cos(y[0, :]))
        ax2.plot(t, y[1, :])
        ax2.set_xlabel('t')
        f.suptitle('Init: (%.3f, %.3f)' % (np.cos(theta0), phi0))
        plt.savefig('2testo_%d.png' % idx)
