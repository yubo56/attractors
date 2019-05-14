import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import get_inf_avg_sol, get_mu4, dmu_ds, stringify

N_PTS = 50

def plot_portrait(I=np.radians(20), s_c=1.5):
    '''
    plot phase portrait of cos(q), s in weak tide limit
    '''
    smax = 10
    _mu = np.linspace(-1, 1, N_PTS)
    _s = np.linspace(0.5, smax, N_PTS)
    s2 = np.linspace(2, smax, N_PTS)

    # plot where dmu changes signs
    plt.plot(s2, 2 / s2, 'r', label=r'$d\mu = 0$')

    # plot bounding (t -> -\infty, mu = 0, s = infty)
    mu, s, _ = get_inf_avg_sol(smax)
    plt.plot(s, mu, 'b', label=r'$(\mu_0, s_0) = (0, \infty)$')

    # plot mu_4 for a fiducial s_c
    mu4 = get_mu4(I, s_c, _s)
    idx4 = np.where(mu4 > 0)[0]
    plt.plot(_s[idx4], mu4[idx4], 'g', label=r'$\cos \theta_4$')

    # flow entire mu_4 curve backwards in time to find cross section
    sols = []
    bounds = []
    for idx in idx4:
        s_curr = _s[idx]
        ret = solve_ivp(dmu_ds, [s_curr, smax], [mu4[idx]], max_step=0.1,
                        dense_output=True)
        sols.append(ret.sol)
        sols_at_curr = [sol(s_curr) for sol in sols]
        bounds.append([np.max(sols_at_curr), np.min(sols_at_curr)])
    mu4_min, mu4_max = np.array(bounds).T
    plt.fill_between(_s[idx4], mu4_min, mu4_max, color='g', alpha=0.3)

    # fill out ~ mu4 +- sqrt(eta * cos(I))
    cass_width = np.sqrt(s_c / _s[idx4] * np.cos(I))
    plt.fill_between(_s[idx4], mu4[idx4] + cass_width, mu4[idx4] - cass_width,
                     color='m', alpha=0.3)

    plt.xlabel(r'$s$')
    plt.ylabel(r'$\cos\theta$')
    plt.xlim(0, smax)
    plt.ylim(-1.1, 1.1)
    plt.legend(loc='lower right')
    plt.title(r'$s_c=%.1f$' % s_c)
    plt.savefig('2phase_portrait%s.png' % stringify(s_c), dpi=400)
    plt.clf()

if __name__ == '__main__':
    plot_portrait()
    plot_portrait(s_c=0.7)
