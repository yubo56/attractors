'''
similar function to 1sim.py, but has separatrix detection capabilities now
'''
import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import solve_ic, to_ang, to_cart, get_etac, get_mu4, get_mu2,\
    stringify, H, roots, get_H4
PLOT_DIR = '4plots'
PKL_FILE = '4dat%s.pkl'

def get_name(s_c, eps, mu0, phi0):
    return stringify(s_c, mu0, phi0, strf='%.3f').replace('-', 'n')

def solve_with_events(I, s_c, eps, mu0, phi0, s0, tf):
    '''
    solves IVP then returns (mu, s, t) at phi=0, pi + others
    '''
    init_xy = np.sqrt(1 - mu0**2)
    init = [-init_xy * np.cos(phi0), -init_xy * np.sin(phi0), mu0, s0]

    event = lambda t, y: y[1]
    t, svec, s, ret = solve_ic(I, s_c, eps, init, tf,
                               events=[event], dense_output=True)
    q, phi = to_ang(*svec)

    [t_events] = ret.t_events
    x_events, _, mu_events, s_events = ret.sol(t_events)
    # phi = 0 means x < 0
    idxs_0 = np.where(x_events < 0)
    idxs_pi = np.where(x_events > 0)
    mu_0, s_0, t_0 = mu_events[idxs_0], s_events[idxs_0], t_events[idxs_0]
    mu_pi, s_pi, t_pi = mu_events[idxs_pi], s_events[idxs_pi], t_events[idxs_pi]

    return (mu_0, s_0, t_0), (mu_pi, s_pi, t_pi), t_events, s, ret

def get_sep_hop_time(t_0, t_pi, mu_0, mu_pi):
    '''
    gets the sep hop/cross time, else -1
    '''
    # two ways to detect a separatrix crossing, either 1) t_0s stop appearing,
    # or 2) mu_pi - mu_0 changes signs
    if len(t_0) > 0 and t_0[-1] < t_pi[-2]:
        # ends at librating, not circulating solution

        # (nb) technically, this could also be a circulating solution after
        # bifurcation, ignore for now
        return t_0[-1]
    else:
        # (nb) technically, mu_0, mu_pi are evaluated at different times, but
        # for circulating solutions they interlock, so they are evaluated at
        # similar enough times to get the sep crossing to reasonable precision
        len_min = min(len(mu_0), len(mu_pi))
        dmu_signs = np.sign(mu_0[ :len_min] - mu_pi[ :len_min])

        if len(dmu_signs) > 0 and dmu_signs[0] != dmu_signs[-1]:
            # can end in a circulating solution about CS1 that is librating
            # about mu=1! still plot sep crossing as normal though

            # criterion 2, circulating and sign flip, sep crossing
            t_cross_idx = np.where(dmu_signs == dmu_signs[-1])[0][0]
            return t_0[t_cross_idx]
        else:
            # no separatrix crossing, even distribution
            return -1

def plot_traj(I, eps, s_c, mu0, phi0, s0, tf=2500):
    '''
    plots (s, mu_{+-}) trajectory over time (shaded) and a few snapshots of
    single orbits in (phi, mu) for each parameter set
    '''
    filename_no_ext = '%s/%s' % (PLOT_DIR, get_name(s_c, eps, mu0, phi0))
    if os.path.exists('%s.png' % filename_no_ext):
        print(filename_no_ext, 'exists!')
        return
    else:
        print('Running', filename_no_ext)

    fig, ax = plt.subplots(1, 1)

    # get top/bottom mus
    (mu_0, s_0, t_0), (mu_pi, s_pi, t_pi),\
        t_events, s, ret = solve_with_events(I, s_c, eps, mu0, phi0, s0, tf)

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
    plt.savefig('%s.png' % filename_no_ext, dpi=400)
    plt.close(fig)

    # 4 snapshots at beginning, just before/after separatrix crossing + end
    # for each snapshot, plot over two t_events (one circulation/libration)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.07)
    t_plot0 = t_events[0]
    t_plot3 = t_events[-3]

    t_cross = get_sep_hop_time(t_0, t_pi, mu_0, mu_pi)
    if t_cross == -1:
        t_plot1, t_plot2 = t_plot0 + (
            np.array([1, 2]) / 3 * (t_plot3 - t_plot0))
    else:
        t_plot1 = t_cross * 0.95
        t_plot2 = min(t_cross * 1.05, (t_plot1 + t_plot3) / 2)

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

            # only plot separatrix if 4 CS
            H_grid = H(I, s_c, s_avg, mu_grid, phi_grid)
            H4 = get_H4(I, s_c, s_avg)
            ax.contour(phi_grid, mu_grid, H_grid, levels=[H4], colors='k')

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
    plt.savefig('%s_ind.png' % filename_no_ext, dpi=400)
    plt.close(fig)

def plot_individual(I, eps):
    '''
    in the terminology of statistics, the below correspond to cases (in order):
    VI (first attracts onto CS1, then kicked into circulating about CS2 at
        bifurcation)
    IV (above) -> VII
    II -> VII
    IV (below) -> VII

    I
    II
    IV (below)
    III
    III -> VI
    '''
    s0 = 10

    # s_c = 0.7, strongly attracting, plot above/inside/below respectively
    plot_traj(I, eps, 0.7, 0.99, 0, s0)
    plot_traj(I, eps, 0.7, 0.8, 0, s0)
    plot_traj(I, eps, 0.7, 0.1, 2 * np.pi / 3, s0)
    plot_traj(I, eps, 0.7, -0.8, 0, s0)

    # s_c = 0.2, probabilistic, plot above/inside/below-enter/below-through
    plot_traj(I, eps, 0.2, 0.3, 0, s0)
    plot_traj(I, eps, 0.2, 0.05, 2 * np.pi / 3, s0)
    plot_traj(I, eps, 0.2, -0.8, 0, s0)
    plot_traj(I, eps, 0.2, -0.82, 0, s0)
    plot_traj(I, eps, 0.2, -0.99, 0, s0)

def statistics(I, eps, s_c, s0=10, tf=2500):
    '''
    for fixed s0, s_c: random (mu, phi), evolve forward in time. Outcomes:
    I - Go to CS1, no separatrix encounter
    II - Go to CS2, no separatrix encounter
    III - separatrix traversing, to CSI
    IV - separatrix hopping, to CS2
    V - Go to CS3, no separatrix encounter (unstable, no)

    VI - Above separatrix, bifurcation
    VII - Inside separatrix, bifurcation
    VIII - Below separatrix, bifurcation (not sure exists?)

    Really, I-IV (V is impossible) are the only initial outcomes, VI/VII are
    later outcomes since bifurcation is later than separatrix interactions. So
    we will focus on (s0, mu0, phi0) -> {I, IV} outcome probability
    '''
    s_c_str = ('%.1f' % s_c).replace('.', '_')
    pkl_fn = PKL_FILE % s_c_str
    def get_outcome_for_init(mu0, phi0):
        '''
        returns 0-3 describing early stage outcome
        '''
        (mu_0, _, t_0), (mu_pi, _, t_pi),\
            _, _, ret = solve_with_events(I, s_c, eps, mu0, phi0, s0, tf)

        t_cross = get_sep_hop_time(t_0, t_pi, mu_0, mu_pi)
        x_f, y_f, z_f, s_f = ret.y[:, -1]
        q_f, phi_f = to_ang(x_f, y_f, z_f)
        H_f = H(I, s_c, s_f, np.cos(q_f), phi_f)
        H4 = get_H4(I, s_c, s_f)
        if t_cross == -1: # no sep encounter, either above or below H4
            if H_f > H4:
                return 1, (mu_0, t_0, mu_pi, t_pi, ret.y, mu0, phi0)
            else:
                assert z_f > 0, 'z_f is %f' % z_f
                return 0, (mu_0, t_0, mu_pi, t_pi, ret.y, mu0, phi0)
        else:
            if H_f > H4:
                return 3, (mu_0, t_0, mu_pi, t_pi, ret.y, mu0, phi0)
            else:
                assert z_f > 0, 'z_f is %f' % z_f
                return 2, (mu_0, t_0, mu_pi, t_pi, ret.y, mu0, phi0)

    if not os.path.exists(pkl_fn):
        print('Running sims, %s not found' % pkl_fn)
        trajs = [[], [], [], []]
        mus = np.linspace(-0.99, 0.99, 40)
        phis = np.linspace(0.1, 2 * np.pi - 0.1, 40)

        for mu0 in mus:
            for phi0 in phis:
                print('Running',  mu0, phi0)
                outcome, traj = get_outcome_for_init(mu0, phi0)
                trajs[outcome].append(traj)
        with open(pkl_fn, 'wb') as f:
            pickle.dump(trajs, f)

    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            trajs = pickle.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.07)
    titles = ['No hop, CS1', 'No hop, CS2', 'Cross to CS1', 'Hop to CS2']
    for ax, outcome_trajs, title in zip([ax1, ax2, ax3, ax4], trajs, titles):
        for _, _, _, _, _, mu0, phi0 in outcome_trajs:
            ax.plot(phi0, mu0, 'bo')
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([-1, 1])
        ax.set_title(title)
    ax1.set_ylabel(r'$\cos \theta$')
    ax3.set_xlabel(r'$\phi$')
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax3.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax3.set_ylabel(r'$\cos \theta$')
    ax4.set_xlabel(r'$\phi$')

    plt.suptitle(r'$(s_c, s_0) = %.1f, %.1f$' % (s_c, s0))
    plt.savefig('4_stats%s.png' % s_c_str)

if __name__ == '__main__':
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    I = np.radians(20)
    eps = 1e-3

    # plot_individual(I, eps)
    # statistics(I, eps, 0.2)
    # statistics(I, eps, 0.4)
    statistics(I, eps, 0.7)
