import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
from utils import roots, solve_ic, to_cart, to_ang, get_four_subplots,\
    plot_point, get_phis

if __name__ == '__main__':
    eta = 0.1
    tide = 0.001
    I = np.radians(20)
    T_F = 10000

    qs, phis = roots(I, eta)

    pert = 0.08 # perturbation strength

    f, axs = get_four_subplots()

    for q0, phi0, ax in zip(qs, phis, axs):
        q_i = q0 + pert
        phi_i = phi0 - pert

        s0 = to_cart(q_i, phi_i)
        sim_time, t, sol = solve_ic(I, eta, tide, s0, T_F)
        print('Sim time:', sim_time)

        x, y, z = sol
        q, phi = to_ang(x, y, z)
        phi = get_phis(q, phi)

        ax.plot(phi % (2 * np.pi),
                np.cos(q),
                'bo',
                markersize=0.3)
        plot_point(ax, q0, 'ro', markersize=4)

        ax.set_title(r'Init: $(\phi_0, \theta_0) = (%.3f, %.3f)$'
                     % (phi_i, q_i), fontsize=8)
        ax.set_xticks([0, np.pi, 2 * np.pi])

    plt.suptitle(r'(I, $\eta$, $\epsilon$)=($%d^\circ$, %.1f, %.1e)'
                 % (np.degrees(I), eta, tide), fontsize=10)
    plt.savefig('2evolution.png', dpi=400)
