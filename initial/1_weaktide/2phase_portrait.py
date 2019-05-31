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
    quiv_scale = 100

    # for integrated flow portraits, mu bound where mu4 disappears
    _mu_int = np.linspace(-0.95, 1, N_PTS) # flow is really strong near (2, -1)
    mu4_int = get_mu4(I, s_c, _s)
    idx4_int = np.where(mu4_int > 0)[0]
    s_int, mu_int = np.meshgrid(_s[idx4_int][::2], _mu_int[::2])

    def plot_shareds(s_arr=_s, plot_all=True):
        ''' few lines are shared on phase portrait and quiver'''

        if plot_all:
            # plot where dmu changes signs
            plt.plot(_s, 2 / _s, 'r', label=r'$d\mu = 0$')

            # plot bounding (t -> -\infty, mu = 0, s = infty)
            # mu, s, _ = get_inf_avg_sol(smax)
            # plt.plot(s, mu, 'b', label=r'$(\mu_0, s_0) = (0, \infty)$')

        # plot mu_4 for a fiducial s_c
        mu4 = get_mu4(I, s_c, s_arr)
        idx4 = np.where(mu4 > 0)[0]
        plt.plot(s_arr[idx4], mu4[idx4], 'g', label=r'$\mu_4$')
        return mu4, idx4

    s, mu = np.meshgrid(_s[::2], _mu[::2])

    # phase portrait no_cs approximation
    def phase_nocs():
        ds, dmu = dydt_nocs(s, mu)
        plt.quiver(s, mu, ds/s, dmu, scale=quiv_scale)

        plt.xlabel(r'$s$')
        plt.ylabel(r'$\mu$')
        plt.title(r'$s_c=%.1f$ ($\mu(\phi) = \mu(0)$)' % s_c)
        mu4, idx4 = plot_shareds()
        cass_width = np.sqrt(s_c / _s[idx4] * np.cos(I))
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.legend(loc='lower left')
        plt.savefig('2quiver%s.png' % stringify(s_c), dpi=400)
        # plt.fill_between(_s[idx4],
        #                  mu4[idx4] + cass_width, mu4[idx4] - cass_width,
        #                  color='m', alpha=0.3)
        # plt.savefig('2quiver_nocs%s.png' % stringify(s_c), dpi=400)
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
        plt.title(r'$s_c=%.1f$ ($\mu(\phi) = \mu(0)$)' % s_c)
        plt.savefig('2phase_portrait%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    # phase portrait w/ under precession-integrated approximation
    def phase_pw():
        dydt_piecewise = get_dydt_piecewise(I, s_c)
        ds_piecewise, dmu_piecewise = dydt_piecewise(s_int, [mu_int])
        plt.quiver(s_int, mu_int, ds_piecewise, dmu_piecewise, scale=quiv_scale)
        mu4, idx4 = plot_shareds(_s, False) # just plot CS4 for clarity
        plt.xlabel(r'$s$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.title(r'$s_c=%.1f$ (Approximate)' % s_c)
        cass_width = np.sqrt(s_c / _s[idx4] * np.cos(I))
        plt.fill_between(_s[idx4],
                         mu4[idx4] + cass_width, mu4[idx4] - cass_width,
                         color='m', alpha=0.3)
        plt.savefig('2quiver_pw%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    # cross section with an effective width guessed from piecewise
    def cross_pw():
        mu4, idx4 = plot_shareds(_s) # just plot CS4 for clarity
        cass_width = np.sqrt(s_c / _s * np.cos(I))

        sols_top = []
        sols_bot = []
        bounds = []
        for idx in idx4:
            s_curr = _s[idx]
            ret_top = solve_ivp(dmu_ds_nocs,
                                [s_curr, smax],
                                [mu4[idx] + cass_width[idx]],
                                max_step=0.1,
                                dense_output=True)
            sols_top.append(ret_top.sol)
            sols_top_curr = [sol(s_curr) for sol in sols_top]
            ret_bot = solve_ivp(dmu_ds_nocs,
                                [s_curr, smax],
                                [mu4[idx] - cass_width[idx]],
                                max_step=0.1,
                                dense_output=True)
            sols_bot.append(ret_bot.sol)
            sols_bot_curr = [sol(s_curr) for sol in sols_bot]

            bounds.append([np.max(sols_top_curr), np.min(sols_bot_curr)])
        mu4_min, mu4_max = np.array(bounds).T
        plt.fill_between(_s[idx4], mu4_min, mu4_max, color='g', alpha=0.3)

        plt.xlabel(r'$s$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.legend(loc='lower right')
        plt.title(r'$s_c=%.1f$ (Piecewise)' % s_c)
        plt.savefig('2phase_portrait_pw%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    # phase portrait w/ under precession-integrated approximation
    def phase_int():
        dydt_num_avg = get_dydt_num_avg(I, s_c, eps)
        ds_num_avg, dmu_num_avg = dydt_num_avg(s_int, [mu_int])
        plt.quiver(s_int, mu_int, ds_num_avg, dmu_num_avg, scale=quiv_scale)
        mu4, idx4 = plot_shareds(_s, False) # just plot CS4 for clarity
        plt.xlabel(r'$s$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.title(r'$s_c=%.1f$ (Integrated)' % s_c)
        cass_width = np.sqrt(s_c / _s[idx4] * np.cos(I))
        plt.fill_between(_s[idx4],
                         mu4[idx4] + cass_width, mu4[idx4] - cass_width,
                         color='m', alpha=0.3)
        plt.savefig('2quiver_int%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    # cross section by integrating mu4 backwards
    def cross_int():
        dydt_num_avg = get_dydt_num_avg(I, s_c, eps)
        def dmu_ds(s, y):
            print(s, y)
            mu = y[0]
            if abs(y[0]) > 1: # stop growing mu if numerically sick
                return np.array([0])
            ds, dmu = dydt_num_avg(s, y)
            return dmu/ds
        mu4, idx4 = plot_shareds(_s) # just plot CS4 for clarity
        cass_width = np.sqrt(s_c / _s * np.cos(I))

        sols_top = []
        sols_bot = []
        bounds = []
        for num_idx, idx in enumerate(idx4):
            s_curr = _s[idx]
            if num_idx < 5: # only integrate first 5, too expensive
                ret_top = solve_ivp(dmu_ds,
                                    [s_curr, smax],
                                    [mu4[idx] + 0.1],
                                    dense_output=True,
                                    method='BDF') # stiff near CS4
                sols_top.append(ret_top.sol)
                ret_bot = solve_ivp(dmu_ds,
                                    [s_curr, smax],
                                    [mu4[idx] - 0.1],
                                    dense_output=True,
                                    method='BDF')
                sols_bot.append(ret_bot.sol)
            sols_top_curr = [sol(s_curr) for sol in sols_top]
            sols_bot_curr = [sol(s_curr) for sol in sols_bot]

            bounds.append([np.max(sols_top_curr), np.min(sols_bot_curr)])
        mu4_min, mu4_max = np.array(bounds).T
        plt.fill_between(_s[idx4], mu4_min, mu4_max, color='g', alpha=0.3)

        plt.xlabel(r'$s$')
        plt.ylabel(r'$\cos\theta$')
        plt.xlim(0, smax)
        plt.ylim(-1.1, 1.1)
        plt.legend(loc='lower right')
        plt.title(r'$s_c=%.1f$ (Piecewise)' % s_c)
        plt.savefig('2phase_portrait_int%s.png' % stringify(s_c), dpi=400)
        plt.clf()

    phase_nocs()
    # cross_nocs()
    # phase_pw()
    # cross_pw()
    # phase_int()
    # cross_int doesn't seem to produce sensible results
    # cross_int()

if __name__ == '__main__':
    # plot_portrait(s_c=1.0)
    plot_portrait(s_c=0.7)
    # plot_portrait(s_c=0.3)
    # plot_portrait(s_c=0.1)
