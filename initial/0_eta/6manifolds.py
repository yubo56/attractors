'''
plot stable/unstable manifolds
'''
from utils import solve_ic, roots, to_cart, to_ang, get_dydt
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

from scipy.optimize import root


def plot_manifolds(eta):
    I = np.radians(20)
    tide = 1e-3
    q, _ = roots(I, eta)
    max_step = 0.05
    # improve CS4 location numerically
    _cs4 = np.array(to_cart(q[3], 0)) + \
        np.array([0, np.sin(tide / (eta * np.sin(I))), 0])
    dydt = lambda s: get_dydt(I, eta, tide)(0, s)
    cs4 = root(dydt, _cs4).x

    def get_displaced(sign_q, sign_phi):
        # sign of sign_q is backwards from what is expected
        cs4_q, cs4_phi = to_ang(*cs4)
        small = 0.001
        eigen = np.sqrt(eta * np.sin(I))
        return to_cart(cs4_q - small * sign_q,
                       cs4_phi + small * eigen * sign_phi)

    # backwards from CS4^0
    init = get_displaced(1, 1)
    _, t1, s1 = solve_ic(I, eta, tide, init, -200, rtol=1e-6, max_step=max_step)
    q1, _phi1 = to_ang(*s1)
    phi1 = np.unwrap(_phi1 + np.pi) - 2 * np.pi
    term1 = np.where(phi1 < 0)[0][0]
    plt.plot(phi1[ :term1], s1[2, :term1],
             label=r'$W_s^{(0)}(%.2f)$' % abs(t1[term1]))

    # forwards from CS4^0
    init = get_displaced(-1, 1)
    _, t2, s2 = solve_ic(I, eta, tide, init, 200, rtol=1e-6, max_step=max_step)
    q2, _phi2 = to_ang(*s2)
    phi2 = np.unwrap(_phi2 + np.pi) - 2 * np.pi
    phi2_grad = np.gradient(phi2) / np.gradient(t2)
    term2 = np.where(np.logical_and(
        abs(phi2_grad) < 2 * min(abs(phi2_grad)),
        phi2 < 1,
    ))[0][0]
    plt.plot(phi2[ :term2], s2[2, :term2],
             label=r'$W_u^{(0)}(%.2f)$' % abs(t2[term2]))

    # backwards from CS4^1
    init = get_displaced(-1, -1)
    _, t3, s3 = solve_ic(I, eta, tide, init, -200, rtol=1e-6, max_step=max_step)
    q3, _phi3 = to_ang(*s3)
    phi3 = np.unwrap(_phi3 + np.pi)
    term3 = np.where(phi3 < 0)[0][0]
    plt.plot(phi3[ :term3], s3[2, :term3],
             label=r'$W_s^{(1)}(%.2f)$' % abs(t3[term3]))

    # forwards from CS4^1
    init = get_displaced(1, -1)
    _, t4, s4 = solve_ic(I, eta, tide, init, 200, rtol=1e-6, max_step=max_step)
    q4, _phi4 = to_ang(*s4)
    phi4 = np.unwrap(_phi4 + np.pi)
    term4 = np.where(phi4 < 0)[0][0]
    plt.plot(phi4[ :term4], s4[2, :term4],
             label=r'$W_u^{(1)}(%.2f)$' % abs(t4[term4]))

    plt.xlim([0, 2 * np.pi])
    mu1, mu2, mu3 = min(s1[2, :term1]), min(s2[2, :term2]), min(s3[2, :term3])
    mu1max, mu4max = max(s1[2, :term1]), max(s4[2, :term4])

    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos \theta$')
    plt.title(r'$\eta = %.1f, \Delta \cos \theta = (%.3f, %.3f, %.3f)$' %
              (eta, (mu2 - mu1) / (mu2 - mu3), mu4max - mu1max, mu2 - mu3))

    plt.legend()
    plt.savefig('6manifolds%s.png' % ('%.2f' % eta).replace('.', '_'))
    plt.clf()


if __name__ == '__main__':
    plot_manifolds(0.05)
    plot_manifolds(0.1)
    plot_manifolds(0.2)
