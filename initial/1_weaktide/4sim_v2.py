import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import solve_ic, to_ang, to_cart, get_etac, get_mu4, get_mu2,\
    stringify, H, roots
PLOT_DIR = '4plots'

def get_name(s_c, eps, mu0, phi0):
    return stringify(s_c, mu0, phi0, strf='%.3f').replace('-', 'n')

def plot_traj(I, s_c, eps, mu0, phi0, s0, tf=2500):
    '''
    plots (s, mu_{+-}) trajectory over time (shaded) and a few snapshots of
    single orbits in (phi, mu) for each parameter set
    '''
    fig, ax = plt.subplots(1, 1)
    init_xy = np.sqrt(1 - mu0**2)
    init = [-init_xy * np.cos(phi0), -init_xy * np.sin(phi0), mu0, s0]

    # track all phi = 0, pi events, split them later
    event = lambda t, y: y[1]
    t, svec, s, ret = solve_ic(I, s_c, eps, init, tf,
                               events=[event], dense_output=True)
    q, phi = to_ang(*svec)

    # get top/bottom mus
    [t_events] = ret.t_events
    x_events, _, mu_events, s_events = ret.sol(t_events)
    # phi = 0 means x < 0
    idxs_0 = np.where(x_events < 0)
    idxs_pi = np.where(x_events > 0)
    mu_0, s_0, t_0 = mu_events[idxs_0], s_events[idxs_0], t_events[idxs_0]
    mu_pi, s_pi, t_pi = mu_events[idxs_pi], s_events[idxs_pi], t_events[idxs_pi]

    ax.scatter(s_0, mu_0, c='r', s=2**2, label=r'$\mu(\phi = 0)$')
    scat = ax.scatter(s_pi, mu_pi, c=t_pi,
                      marker='+', s=5**2, label=r'$\mu(\phi = \pi)$')
    fig.colorbar(scat, ax=ax)

    # overplot mu2, mu4
    mu2 = get_mu2(I, s_c, s)
    ax.plot(s, mu2, 'c-', label=r'$\mu_2$')
    mu4 = get_mu4(I, s_c, s)
    mu4_idx = np.where(mu4 > 0)
    ax.plot(s[mu4_idx], mu4[mu4_idx], 'g:', label=r'$\mu_4$')

    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$\mu$')
    ax.set_xlim([0, 1.1 * s0])
    ax.legend(loc='lower right')
    plt.suptitle(r'$(s_c; s_0, \mu_0, \phi_0) = (%.1f; %d, %.3f, %.2f)$' %
                 (s_c, s0, mu0, phi0))
    plt.savefig('%s/%s.png' % (PLOT_DIR, get_name(s_c, eps, mu0, phi0)),
                dpi=400)
    plt.close(fig)

    # 4 snapshots at beginning, just before/after separatrix crossing + end
    # for each snapshot, plot over two t_events (one circulation/libration)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.07)
    t_plot0 = t_events[0]
    t_plot3 = t_events[-3]
    # two ways to detect a separatrix crossing, either 1) t_0s stop appearing,
    # or 2) mu_pi - mu_0 changes signs
    if len(t_0) > 0 and t_0[-1] < t_pi[-2]:
        # ends at librating, not circulating solution

        # (nb) technically, this could also be a circulating solution after
        # bifurcation, ignore for now
        t_plot1 = t_0[-1] * 0.95
        t_plot2 = min(t_0[-1] * 1.05,
                      (t_plot1 + t_plot3) / 2) # in case 1.05 exceeds
        prefix = 'L'
    else:
        # (nb) technically, mu_0, mu_pi are evaluated at different times, but
        # for circulating solutions they interlock, so they are evaluated at
        # similar enough times to get the sep crossing to reasonable precision
        len_min = min(len(mu_0), len(mu_pi))
        dmu_signs = np.sign(mu_0[ :len_min] - mu_pi[ :len_min])

        if len(dmu_signs) > 0 and dmu_signs[0] != dmu_signs[-1]:
            # sanity check that we are in circulating solution; check in here in
            # case started at CS2 and never circulated
            assert len(mu_0 - len_min) < 2 and len(mu_pi - len_min) < 2,\
                'Lengths of two mus are %d, %d' % (len(mu_0), len(mu_pi))

            # criterion 2, circulating and sign flip, sep crossing
            t_cross_idx = np.where(dmu_signs == dmu_signs[-1])[0][0]
            t_plot1 = t_0[t_cross_idx] * 0.95
            t_plot2 = min(t_0[t_cross_idx] * 1.05,
                          (t_plot1 + t_plot3) / 2)
            prefix = 'H'
        else:
            # no separatrix crossing, even distribution
            t_plot1, t_plot2 = t_plot0 + (
                np.array([1, 2]) / 3 * (t_plot3 - t_plot0))
            prefix = 'N'
    # get last t_event before t_plots
    t_plot1_idx, t_plot2_idx = [np.where(t_events <= t)[0][-1]
                                for t in [t_plot1, t_plot2]]

    # plot0, plot3 have idxs 0, -3
    for ax, t_idx in zip([ax1, ax2, ax3, ax4],
                         [0, t_plot1_idx, t_plot2_idx, -3]):
        num_pt = 40
        N = 100

        # get solution over interval
        t_vals = np.linspace(t_events[t_idx], t_events[t_idx + 2], num_pt)[ :-1]
        x, y, z, s = ret.sol(t_vals)
        q, phi = to_ang(x, y, z)

        # first plot separatrix + CS's, so in background of trajectory
        phi_grid, mu_grid = np.meshgrid(np.linspace(0, 2 * np.pi, N),
                                        np.linspace(-1, 1, N))
        # evaluate H at average s, since it doesn't change much over one cycle
        s_avg = np.mean(s)
        H_grid = H(I, s_c, s_avg, mu_grid, phi_grid)
        [mu4] = get_mu4(I, s_c, np.array([s_avg]))
        ax.contour(phi_grid, mu_grid, H_grid,
                   levels=[H(I, s_c, s_avg, mu4, 0)], colors='k')

        cs_vals = np.cos(roots(I, s_c, s_avg))
        styles = ['ro', 'mo', 'go', 'co']
        ms = 5
        if len(cs_vals) == 2:
            # CS2 + 3
            ax.plot(np.pi, cs_vals[0], styles[1], markersize=ms)
            ax.plot(0, cs_vals[1], styles[2], markersize=ms)
            ax.plot(2 * np.pi, cs_vals[1], styles[2], markersize=ms)
        else:
            ax.plot(np.pi, cs_vals[1], styles[1], markersize=ms)
            for idx in [0, 2, 3]:
                ax.plot(0, cs_vals[idx], styles[idx], markersize=ms)
                ax.plot(2 * np.pi, cs_vals[idx], styles[idx], markersize=ms)

        # plot trajectory
        ax.scatter(phi, np.cos(q), s=2**2, c='b')
        ax.set_title(r'$t \in [%.1f, %.1f], \eta = %.3f$'
                     % (t_vals.min(), t_vals.max(), s_c / s_avg))
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([-1, 1])
    ax1.set_ylabel(r'$\cos \theta$')
    ax3.set_xlabel(r'$\phi$')
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax3.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax3.set_ylabel(r'$\cos \theta$')
    ax4.set_xlabel(r'$\phi$')

    plt.suptitle(r'$(s_c; s_0, \mu_0, \phi_0) = (%.1f; %d, %.3f, %.2f)$' %
                 (s_c, s0, mu0, phi0))
    plt.savefig('%s/%s_ind.png' % (PLOT_DIR, get_name(s_c, eps, mu0, phi0)),
                dpi=400)
    plt.close(fig)

if __name__ == '__main__':
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    I = np.radians(20)
    eps = 1e-3
    s0 = 10

    # s_c = 0.7, strongly attracting, plot above/inside/below respectively
    plot_traj(I, 0.7, eps, 0.8, 0, s0)
    plot_traj(I, 0.7, eps, 0.1, 2 * np.pi / 3, s0)
    plot_traj(I, 0.7, eps, -0.8, 0, s0)

    # s_c = 0.2, probabilistic, plot above/inside/below-enter/below-hop
    plot_traj(I, 0.2, eps, 0.3, 0, s0)
    plot_traj(I, 0.2, eps, 0.05, 2 * np.pi / 3, s0)
    plot_traj(I, 0.2, eps, -0.8, 0, s0)
    plot_traj(I, 0.2, eps, -0.8, 1, s0)
