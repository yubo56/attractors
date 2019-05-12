import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import solve_ic, to_ang, get_crits, get_etac

def get_name(s_c, eps, mu0):
    epspow = -np.log10(eps)
    return ('%.1fx%.1fx%.1f' % (s_c, epspow, -mu0)).replace('.', '_')

def traj_for_sc(I, s_c, eps, mu0, s0, tf=2500):
    init = [-np.sqrt(1 - mu0**2), 0, mu0, s0]

    t, svec, s = solve_ic(I, s_c, eps, init, tf)
    q, phi = to_ang(*svec)
    plt.plot(phi, np.cos(q), 'bo', markersize=0.3)
    plt.plot(phi[-1], np.cos(q)[-1], 'ro', markersize=1.5)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos\theta$')
    plt.title(r'$(s_c, s_0, \cos \theta_0) = (%.2f, %d, %.2f)$' %
              (s_c, s0, mu0))
    plt.savefig('1traj_%s.png' % get_name(s_c, eps, mu0))
    plt.clf()

    eta = s_c / s
    plt.plot(t, eta * np.cos(I) / (1 + eta * np.sin(I)), 'b', label=r'$\mu_4$')
    plt.plot(t, svec[2, :], 'go', label=r'$\mu$', markersize=0.3)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mu$')
    plt.title(r'$\mu(t), \mu_4(t)$')
    plt.legend()
    plt.savefig('1eta_%s.png' % get_name(s_c, eps, mu0))
    plt.clf()

if __name__ == '__main__':
    I = np.radians(20)
    eps = 1e-3
    s0 = 10

    s_c0 = get_etac(I) + 0.01
    mu, s, mu4 = get_crits(I, s_c0)
    plt.plot(s, mu, label=r'$\mu$')
    plt.plot(s, mu4, label=r'$\mu_4$')
    plt.xlabel('s')
    plt.xlim([0, 4])
    plt.ylabel(r'$\mu$')
    plt.legend()
    plt.savefig('1mu_c.png')

    # traj_for_sc(I, 2, eps, -0.9, s0)
    # traj_for_sc(I, 0.2, eps, -0.9, s0, tf=5000)
    # traj_for_sc(I, 0.03, eps, -0.5, s0, tf=3000)
    # traj_for_sc(I, 2, eps, -0.3, s0, tf=2000)
