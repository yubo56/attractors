import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import solve_ic, to_ang

if __name__ == '__main__':
    tf = 1000
    I = np.radians(20)
    s_c = 0.1
    eps = 1e-3

    _init = np.array([-1, 0, -0.9])
    mu0 = -0.9
    s0 = 10

    init = [-np.sqrt(1 - mu0**2), 0, mu0, s0]

    t, svec, s = solve_ic(I, s_c, eps, init, tf)
    q, phi = to_ang(*svec)
    plt.plot(phi, np.cos(q), 'bo', markersize=2)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos\theta$')
    plt.title(r'$(s_c, s_0, \cos \theta_0) = (%.2f, %d, %.2f)$' %
              (s_c, s0, mu0))
    plt.savefig('1traj.png')
    plt.clf()

    plt.plot(t, s_c / s)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\eta$')
    plt.title(r'$\eta(t)$')
    plt.savefig('1eta.png')
