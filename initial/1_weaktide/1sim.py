import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import solve_ic, to_ang,\
    get_etac, get_mu4, stringify

def get_name(s_c, eps, mu0):
    return stringify(s_c, mu0).replace('-', 'n')

def traj_for_sc(I, s_c, eps, mu0, s0, tf=2500):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    init = [-np.sqrt(1 - mu0**2), 0, mu0, s0]

    t, svec, s = solve_ic(I, s_c, eps, init, tf)

    q, phi = to_ang(*svec)

    # plot evolution of mu (at all times) and mu4
    ax1.plot(t, svec[2, :], 'go', label=r'$\mu$', markersize=0.3)
    mu4 = get_mu4(I, s_c, s)
    base_slice = np.where(mu4 > 0)[0]
    ax1.plot(t[base_slice], mu4[base_slice], 'r', label=r'$\mu_4$',
             linewidth=0.7)

    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$\mu$')
    ax1.legend()

    # scatter plot w/ color gradient + colorbar
    c = t
    ax2.plot(phi[-1], np.cos(q)[-1], 'ro', markersize=1.5)
    scat = ax2.scatter(phi, np.cos(q), c=c, s=0.3**2)
    ax2.set_xlabel(r'$\phi$')
    ax2.set_ylabel(r'$\cos\theta$')
    fig.colorbar(scat, ax=ax2)

    plt.suptitle(r'$(s_c, s_0, \cos \theta_0) = (%.2f, %d, %.2f)$' %
                  (s_c, s0, mu0))

    plt.savefig('1sim_%s.png' % get_name(s_c, eps, mu0), dpi=400)
    plt.clf()

if __name__ == '__main__':
    I = np.radians(20)
    eps = 1e-3
    s0 = 10
    s_c = 1.5

    for s_c in [0.7, 1.5]:
        for mu0 in [-0.9, -0.2, 0.5, 0.9]:
            traj_for_sc(I, s_c, eps, mu0, s0)
