import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import get_inf_avg_sol, stringify, get_mu4, dmu_ds_nocs, dydt_nocs,\
    get_dydt_num_avg, get_dydt_piecewise

N_PTS = 50

def plot_portrait(I=np.radians(20), s_c=1.5):
    '''
    plot phase portrait of cos(q), s in weak tide limit
    '''
    smax = 10
    eps = 1e-4
    _mu = np.linspace(-1, 1, N_PTS)
    _s = np.linspace(0.5, smax, N_PTS)
    s2 = np.linspace(2, smax, N_PTS)
    quiv_scale = 400

    def plot_shareds(s_arr=_s, plot_all=True):
        ''' few lines are shared on phase portrait and quiver'''

        if plot_all:
            # plot where dmu changes signs
            plt.plot(s2, 2 / s2, 'r', label=r'$d\mu = 0$')

            # plot bounding (t -> -\infty, mu = 0, s = infty)
            mu, s, _ = get_inf_avg_sol(smax)
            plt.plot(s, mu, 'b', label=r'$(\mu_0, s_0) = (0, \infty)$')

        # plot mu_4 for a fiducial s_c
        mu4 = get_mu4(I, s_c, s_arr)
        idx4 = np.where(mu4 > 0)[0]
        plt.plot(s_arr[idx4], mu4[idx4], 'g', label=r'$\mu_4$')
        return mu4, idx4

    s, mu = np.meshgrid(_s[::2], _mu[::2])

    # phase portrait no_cs approximation
    def phase_nocs():
        ds, dmu = dydt_nocs(s, mu)
        plt.quiver(s, mu, ds, dmu, scale=quiv_scale)

        plt.xlabel(r'$s$')
        plt.ylabel(r'$\mu$')
        plt.title(r'$s_c=%.1f$ ($\mu(\phi) = \mu(0)$)' % s_c)
        mu4, idx4 = plot_shareds()
        cass_width = np.sqrt(s_c / s2[idx4] * np.cos(I))
        plt.fill_between(_s[idx4],
                         mu4[idx4] + cass_width, mu4[idx4] - cass_width,
                         color='m', alpha=0.3)
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.legend(loc='lower left')
        plt.savefig('2quiver%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    # flow mu_4 back in time to find cross section in no_cs approx
    def cross_nocs():
        mu4, idx4 = plot_shareds()
        sols = []
        bounds = []
        for idx in idx4:
            s_curr = _s[idx]
            ret = solve_ivp(dmu_ds_nocs, [s_curr, smax], [mu4[idx]], max_step=0.1,
                            dense_output=True)
            sols.append(ret.sol)
            sols_at_curr = [sol(s_curr) for sol in sols]
            bounds.append([np.max(sols_at_curr), np.min(sols_at_curr)])
        mu4_min, mu4_max = np.array(bounds).T
        plt.fill_between(_s[idx4], mu4_min, mu4_max, color='g', alpha=0.3)

        plt.xlabel(r'$s$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.legend(loc='lower right')
        plt.title(r'$s_c=%.1f$' % s_c)
        plt.savefig('2phase_portrait%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    # phase portrait w/ under precession-integrated approximation
    _mu_int = np.linspace(-0.95, 1, N_PTS) # flow is really strong near (2, -1)
    s_int, mu_int = np.meshgrid(s2[::2], _mu_int[::2])
    def phase_int():
        dydt_num_avg = get_dydt_num_avg(I, s_c, eps)
        ds_num_avg, dmu_num_avg = dydt_num_avg(s_int, [mu_int])
        plt.quiver(s_int, mu_int, ds_num_avg, dmu_num_avg, scale=quiv_scale)
        mu4, idx4 = plot_shareds(s2, False) # just plot CS4 for clarity
        plt.xlabel(r'$s$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.title(r'$s_c=%.1f$ (Integrated)' % s_c)
        cass_width = np.sqrt(s_c / s2[idx4] * np.cos(I))
        plt.fill_between(s2[idx4],
                         mu4[idx4] + cass_width, mu4[idx4] - cass_width,
                         color='m', alpha=0.3)
        plt.savefig('2quiver_int%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    # phase portrait w/ under precession-integrated approximation
    def phase_pw():
        dydt_piecewise = get_dydt_piecewise(I, s_c)
        ds_piecewise, dmu_piecewise = dydt_piecewise(s_int, [mu_int])
        plt.quiver(s_int, mu_int, ds_piecewise, dmu_piecewise, scale=quiv_scale)
        mu4, idx4 = plot_shareds(s2, False) # just plot CS4 for clarity
        plt.xlabel(r'$s$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.title(r'$s_c=%.1f$ (Approximate)' % s_c)
        cass_width = np.sqrt(s_c / s2[idx4] * np.cos(I))
        plt.fill_between(s2[idx4],
                         mu4[idx4] + cass_width, mu4[idx4] - cass_width,
                         color='m', alpha=0.3)
        plt.savefig('2quiver_pw%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    phase_nocs()
    cross_nocs()
    phase_int()
    phase_pw()

if __name__ == '__main__':
    # plot_portrait()
    plot_portrait(s_c=0.7)
