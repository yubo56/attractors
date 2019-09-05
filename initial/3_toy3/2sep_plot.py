import scipy.optimize as opt
from scipy import integrate
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

from utils import H, roots, solve_ic, to_cart, to_ang

if __name__ == '__main__':
    phi = np.linspace(0, 2 * np.pi + 0, 1000)
    I = np.radians(20)
    eta = 0.2

    def mu_up(phi, eta=eta):
        q4 = roots(I, eta)[3]
        return (eta * np.cos(I) / (1 - eta * np.sin(I))
                + np.sqrt(2 * eta * np.sin(I) * (1 - np.cos(phi))))

    def mu_down(phi, eta=eta):
        q4 = roots(I, eta)[3]
        return (eta * np.cos(I) / (1 - eta * np.sin(I))
                - np.sqrt(2 * eta * np.sin(I) * (1 - np.cos(phi))))

    def n_mu_up(phi, eta=eta):
        q4 = roots(I, eta)[3]
        def dH(q):
            return H(I, eta, q, phi) - H(I, eta, q4, 0)
        return np.cos(opt.brentq(dH, q4, 0))

    def n_mu_down(phi, eta=eta):
        q4 = roots(I, eta)[3]
        def dH(q):
            return H(I, eta, q, phi) - H(I, eta, q4, 0)
        return np.cos(opt.brentq(dH, -np.pi, q4))

    # get a numerically-integrated separatrix crossing trajectory
    def int_traj():
        delta = 0.7
        EPS = 3e-4
        tf = 500
        q4 = roots(I, eta)[3]
        def arg_bot(phi):
            m = mu_down(phi)
            return delta * eta * (np.cos(I) + np.sin(I) * np.sqrt(
                (1 - np.cos(phi)) / (2 * eta * np.sin(I)))) + (1 - m**2)
        bot = EPS * integrate.quad(arg_bot, 0, 2 * np.pi)[0]
        def dH(q):
            # return bot/2 below H4
            return H(I, eta, q, 0) - (H(I, eta, q4, 0) - bot / 3)
        q0 = opt.brentq(dH, -np.pi, q4)
        y0 = [*to_cart(q0, 0), eta]
        ret = solve_ic(I, EPS, delta, y0, tf)
        t = np.linspace(0, ret.t_events[0][3], 200)
        x, y, z, _eta = ret.sol(t)
        q, phi = to_ang(x, y, z)
        plt.plot(phi, np.cos(q), 'b.')

    int_traj()
    plt.plot(phi, mu_up(phi), 'g')
    plt.plot(phi, mu_down(phi), 'g')
    plt.plot(phi, [n_mu_up(f) for f in phi], 'r')
    plt.plot(phi, [n_mu_down(f) for f in phi], 'r')
    plt.savefig('2sep_plot.png')
