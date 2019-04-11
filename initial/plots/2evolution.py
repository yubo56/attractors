import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from utils import roots, solve_ic

if __name__ == '__main__':
    eta = 0.1
    tide = 0.01
    I = np.radians(20)
    T_F = 10000

    qs, phis = roots(eta, I)

    pert = 0.08 # perturbation strength

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    f.subplots_adjust(wspace=0)

    for q0, phi0, ax in zip(qs, phis, [ax1, ax2, ax3, ax4]):
        q_i = q0 + pert
        phi_i = phi0 - pert

        s0 = [
            -np.sin(q_i) * np.cos(phi_i),
            -np.sin(q_i) * np.sin(phi_i),
            np.cos(q_i)]
        sim_time, t, sol = solve_ic(I, eta, tide, s0, T_F, method='LSODA')
        print('Sim time:', sim_time)

        x, y, z = sol
        r = np.sqrt(x**2 + y**2 + z**2)
        q = np.arccos(z / r) * np.sign(q_i)
        phi = (np.arctan2(-y / np.sin(q), -x / np.sin(q)) + 2 * np.pi)\
            % (2 * np.pi)

        ax.plot(phi % (2 * np.pi),
                np.cos(q),
                'bo',
                markersize=0.3)
        ax.plot(phi0 % (2 * np.pi),
                np.cos(q0),
                'ro',
                markersize=4)

        ax.set_title(r'Init: $(\phi_0, \theta_0) = (%.3f, %.3f)$'
                     % (phi0, q0), fontsize=8)
        ax.set_xticks([0, np.pi, 2 * np.pi])

    plt.suptitle(r'(I, $\eta$)=($%d^\circ$, %.3f)' % (np.degrees(I), eta),
                 fontsize=10)
    ax1.set_ylabel(r'$\cos \theta$')
    ax3.set_xlabel(r'$\phi$')
    ax3.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax3.set_ylabel(r'$\cos \theta$')
    ax4.set_xlabel(r'$\phi$')
    plt.savefig('2evolution.png', dpi=400)
