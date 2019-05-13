import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import solve_ic, solve_ic_avg, to_ang, backwards_solve,\
    get_etac, get_upper_sc, get_mu4

def get_name(s_c, eps, mu0):
    epspow = -np.log10(eps)
    return ('%.1fx%.1fx%.1f' % (s_c, epspow, -mu0)).replace('.', '_')

def traj_for_sc(I, s_c, eps, mu0, s0, tf=2500):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    init = [-np.sqrt(1 - mu0**2), 0, mu0, s0]

    t, svec, s = solve_ic(I, s_c, eps, init, tf)
    t_avg, mu_avg, s_avg = solve_ic_avg(I, s_c, eps, [mu0, s0], tf)
    # mu, s, mu4 = backwards_solve(I, s_f)

    q, phi = to_ang(*svec)

    ax1.plot(t, svec[2, :], 'go', label=r'$\mu$', markersize=0.3)
    ax1.plot(t_avg, mu_avg, 'ro', label=r'$\mu_{avg}$', markersize=0.3)
    # plot mu4's up until CS disappears
    mu4 = get_mu4(I, s_c, s)
    mu4_avg = get_mu4(I, s_c, s_avg)
    base_slice = np.where(mu4 > 0)[0]
    avg_slice = np.where(mu4_avg > 0)[0]
    ax1.plot(t[base_slice], mu4[base_slice], 'g', label=r'$\mu_4$',
             linewidth=0.7)
    ax1.plot(t_avg[avg_slice], mu4_avg[avg_slice], 'r', label=r'$\mu_{4,avg}$',
             linewidth=0.7)

    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$\mu$')
    ax1.legend()

    ax2.plot(phi, np.cos(q), 'bo', markersize=0.3)
    ax2.plot(phi[-1], np.cos(q)[-1], 'ro', markersize=1.5)
    ax2.set_xlabel(r'$\phi$')
    ax2.set_ylabel(r'$\cos\theta$')

    plt.suptitle(r'$(s_c, s_0, \cos \theta_0) = (%.2f, %d, %.2f)$' %
                  (s_c, s0, mu0))

    plt.savefig('1sim_%s.png' % get_name(s_c, eps, mu0), dpi=400)
    plt.clf()

if __name__ == '__main__':
    I = np.radians(20)
    eps = 1e-3
    s0 = 10

    upper = get_upper_sc(I)
    lower = get_etac(I)

    traj_for_sc(I, 2, eps, -0.9, s0)
    traj_for_sc(I, 1, eps, 0.2, s0)
    traj_for_sc(I, 0.9 * upper + 0.1 * lower, eps, -0.7, s0)
    traj_for_sc(I, 0.9 * upper + 0.1 * lower, eps, -0.1, s0)
    traj_for_sc(I, 1, eps, 0.7, s0)
