'''
study dissipating disk problem, where eta shrinks over time

really could have used splitting into multiple files, but no pre-planning sorry
'''
import os
import scipy.optimize as opt
import numpy as np
from multiprocessing import Pool
from matplotlib import cm
from scipy.interpolate import interp1d

import matplotlib
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

PLOT_DIR = '1plots'
from utils import to_cart, to_ang, roots, H, solve_ic_base,\
    get_etac
dpi = 600

def sep_areas_exact(I, eta):
    mu4 = eta * np.cos(I) / (1 - eta * np.sin(I))
    q4 = -np.arccos(mu4)

    # WH2004 eq 11-13
    z0 = eta * np.cos(I)
    chi = np.sqrt(-np.tan(q4)**3 / np.tan(I) - 1)
    rho = chi * np.sin(q4)**2 * np.cos(q4) / (
        chi**2 * np.cos(q4)**2 + 1)
    T = 2 * chi * np.cos(q4) / (
        chi**2 * np.cos(q4)**2 - 1)
    A2 = 8 * rho + 4 * np.arctan(T) - 8 * z0 * np.arctan(1 / chi)
    A1 = 2 * np.pi * (1 - z0) - A2 / 2
    A3 = 2 * np.pi * (1 + z0) - A2 / 2
    return A1, A2, A3

def d_sep_areas(I, eta):
    ''' numerically compute dA_i/deta '''
    deta = 1e-5
    As_i = sep_areas_exact(I, eta - deta)
    As_f = sep_areas_exact(I, eta + deta)
    return (np.array(As_f) - np.array(As_i)) / (2 * deta)

def plot_traj_colors(I, ret, filename):
    ''' scatter plot of (phi, mu) w/ colored time '''
    fig, ax1 = plt.subplots(1, 1)
    t = ret.t
    etas = ret.y[3]
    q, phi = to_ang(*ret.y[ :3])
    # mu_lim = 0.6
    # first_idx = np.where(abs(np.cos(q)) < mu_lim)[0][0]
    first_idx = 0
    scat = ax1.scatter(phi[first_idx: ], np.cos(q[first_idx: ]),
                       c=t[first_idx: ], s=0.3, cmap='Spectral')
    ylims = ax1.get_ylim()
    fig.colorbar(scat, ax=ax1)
    ax1.set_xlabel(r'$\phi$')
    ax1.set_xlim([0, 2 * np.pi])
    ax1.set_xticks([0, np.pi, 2 * np.pi])
    ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax1.set_ylabel(r'$\cos\theta$')

    eta_f = etas[-1]
    phi_arr = np.linspace(0, 2 * np.pi, 201)
    z4 = eta_f * np.cos(I) / (1 - eta_f * np.sin(I))
    sep_diff = np.sqrt(2 * eta_f * np.sin(I) * (1 - np.cos(phi_arr)))
    ax1.plot(phi_arr, z4 + sep_diff, 'k', label='Final Sep')
    ax1.plot(phi_arr, z4 - sep_diff, 'k')
    ax1.legend()
    ax1.set_ylim(ylims)
    plt.savefig(filename, dpi=dpi)
    print('Saved', filename)
    plt.clf()

def get_t_vals(t_i, t_f, int_n_pts=401):
    ''' uneven t spacing, concentrate @ ends + middle '''
    t_m = (t_f + t_i) / 2
    unit = np.linspace(0, 1, int_n_pts // 2 + 1)
    t1 = np.sin(unit * np.pi / 2)**2 * (t_m - t_i) + t_i
    t2 = np.sin(unit * np.pi / 2)**2 * (t_f - t_m) + t_m
    # first/last elements overlap
    return np.concatenate((t1, t2[1: ]))

def get_areas_2(ret):
    '''
    not get_areas in utils since this that assumes start -> ejection from sep,
    but really assumes a very full featured trajectory
    '''
    t_events = ret.t_events[-1]
    x_events, _, z_events, eta_events = ret.sol(t_events)
    # phi = 0 means x < 0
    idx_0 = np.where(x_events < 0)[0]
    idx_pi = np.where(x_events > 0)[0]
    t_0 = t_events[idx_0]
    t_pi = t_events[idx_pi]
    areas = []
    t_areas = []
    t_areas_f = []
    # to ease casework, if starts circulating, ensure first event is a t_0
    if t_0[0] < t_pi[1] and t_0[0] > t_pi[0]:
        t_pi = t_pi[1: ]

    # can start either circulating or librating, but always eventually
    # transitions to librating about CS2, and ends up circulating again...

    # if starts circulating, integrate (1 - mu) dphi for continuity... so many
    # cases
    for t_0_i, t_0_f, t_pi_f in zip(t_0, t_0[1: ], t_pi[1: ]):
        if t_0_f < t_pi_f: # circulating
            t_vals = get_t_vals(t_0_i, t_0_f)
            t_areas.append(t_0_i)
            t_areas_f.append(t_0_f)

            x, y, z, _ = ret.sol(t_vals)
            q, phi = to_ang(x, y, z)
            dphi = np.gradient(np.unwrap(phi))
            areas.append(np.sum((1 - np.cos(q)) * dphi))
        else:
            break
    # truncate list of times during initial circulation
    t_end_circ = t_pi_f
    if t_end_circ >= t_0[-1]: # pure circulating traj?
        return np.inf, np.inf, np.array(t_areas), np.array(t_areas_f), np.array(areas)
    if len(areas):
        t_0 = t_0[np.where(t_0 >= t_end_circ)[0][0]: ]
        t_pi = t_pi[np.where(t_pi >= t_end_circ)[0][0]: ]
    # now solve librating period
    t_end_lib = max(t_pi) if len(t_0) == 0 else t_0[0]
    t_pi_lib = t_pi[np.where(t_pi < t_end_lib)[0]]
    for t_pi_i, t_pi_f in zip(t_pi_lib[ :-2:2], t_pi_lib[2::2]):
        t_vals = get_t_vals(t_pi_i, t_pi_f)
        t_areas.append(t_pi_i)
        t_areas_f.append(t_pi_f)

        x, y, z, _ = ret.sol(t_vals)
        q, phi = to_ang(x, y, z)
        dphi = np.gradient(np.unwrap(phi))
        areas.append(np.sum((1 - np.cos(q)) * dphi))
    # final circ period; note that t_0[0] < t_pi_fin[0] again
    t_pi_fin = t_pi[np.where(t_pi > t_end_lib)[0]]
    for t_0_i, t_0_f, t_pi_f in zip(t_0, t_0[1: ], t_pi_fin[1: ]):
        t_vals = get_t_vals(t_0_i, t_0_f)
        t_areas.append(t_0_i)
        t_areas_f.append(t_0_f)

        x, y, z, _ = ret.sol(t_vals)
        q, phi = to_ang(x, y, z)
        dphi = np.gradient(np.unwrap(phi))
        areas.append(np.sum((1 - np.cos(q)) * dphi))
    return t_end_lib, t_end_circ,\
        np.array(t_areas), np.array(t_areas_f), np.array(areas)

def plot_single(I, eps, tf, eta0, q0, filename, dq=0.3,
                num_snapshots=1, plot_type='', events=[]):
    '''
    plot mu(t) and zoomed in for a trajectory
    num_snapshots controls how many points to use for snapshots
    plot_type controls which of the final area predictions to overlay (default
        none)
    '''
    eta_c = get_etac(I)
    y0 = [*to_cart(q0 + dq, 0), eta0]
    PKL_FN = '%s.pkl' % filename
    if not os.path.exists(PKL_FN):
        ret = solve_ic_base(I, eps, y0, tf, events=events)
        t_end_lib, t_end_circ, t_areas, t_areas_f, areas = get_areas_2(ret)
        with open(PKL_FN, 'wb') as f:
            pickle.dump((ret, t_end_lib, t_end_circ,
                         t_areas, t_areas_f, areas), f)
    else:
        with open(PKL_FN, 'rb') as f:
            ret, t_end_lib, t_end_circ,\
                t_areas, t_areas_f, areas = pickle.load(f)
    # get initial area
    eta_areas = ret.sol(t_areas)[3]
    a_init_int = areas[0]
    a_init_dq = 2 * np.pi * (1 - np.cos(dq))
    print('Areas (integrated/estimated): ', a_init_int, a_init_dq)

    # calculate times to plot for snapshots
    dAreas = abs((areas[1: ] - areas[ :-1]) / areas[1: ])
    peak_idxs = np.argsort(dAreas)
    # plot @ sep appearance
    cross_idx = np.where(eta_areas < eta_c)[0][0]
    sep_idxs = peak_idxs[-num_snapshots: ]
    # 2 plots for every sep_idx
    plot_idxs = sorted([1, cross_idx, *sep_idxs,
                        *([i + 1 for i in sep_idxs])]) + [-2]

    fig, (ax1, ax3) = plt.subplots(2, 1,
                            sharex=True,
                            figsize=(6, 5),
                            gridspec_kw={'height_ratios': [3, 2]})
    t = ret.t
    etas = ret.y[3]
    q, phi = to_ang(*ret.y[0:3])

    # calculate separatrix min/max mu @ phi = pi
    idx4s = np.where(etas < get_etac(I))[0]
    eta4s = etas[idx4s]
    q4s, q2s = np.zeros_like(eta4s), np.zeros_like(eta4s)
    for i, eta in enumerate(eta4s):
        _, q2, _, q4 = roots(I, eta)
        q2s[i] = q2
        q4s[i] = q4
    mu4s = np.cos(q4s)
    H4s = H(I, eta4s, q4s, 0)
    # H = -mu**2 / 2 + eta * (mu * cos(I) - sin(I) * sin(q) * cos(phi))
    mu_min, mu_max = np.zeros_like(eta4s), np.zeros_like(eta4s)
    for i in range(len(eta4s)):
        # suppress dividebyzero/sqrt
        with np.errstate(all='ignore'):
            try:
                f = lambda mu: -mu**2 / 2 + eta4s[i] * (
                    mu * np.cos(I) + np.sin(I) * np.sqrt(1 - mu**2))\
                    - H4s[i]
                # mu_min will always be < mu4, mu_max > mu4
                mu_min[i] = opt.bisect(f, -1, mu4s[i])
                # note mu_max may not exist, so push -1 if not
                try:
                    mu_max[i] = opt.bisect(f, mu4s[i], 1)
                except ValueError:
                    mu_max[i] = -1
            except:
                pass

    # plot trajectory's mu, separatrix's min/max mu, CS2
    A_sep_crit = 4 * np.pi * (1 - (1 + np.tan(I)**(2/3))**(-3/2))
    if a_init_int < A_sep_crit:
        opt_func = lambda eta: sep_areas_exact(I, eta)[1] - a_init_int
        eta_guess = (a_init_int / 16)**2 / np.sin(I)
        eta_cross = opt.newton(opt_func, x0=eta_guess)
        a_crossed21 = -sep_areas_exact(I, eta_cross)[0]
        a_crossed23 = sum(sep_areas_exact(I, eta_cross)[ :2])
    # else is either 3 -> 1 or 3 -> 2 -> 1, which are pretty different, so we
    # will handle them in the plot_type branches below
    mu_dat = np.cos(q)

    ax1.semilogx(etas, mu_dat, c='y', label='Sim')
    max_valid_idxs = np.where(mu_max > 0)[0]
    ax1.semilogx(etas[idx4s][max_valid_idxs], mu_max[max_valid_idxs],
             'k', label='Sep')
    ax1.semilogx(etas[idx4s], mu_min, 'k')
    ax1.semilogx(etas[idx4s], np.cos(q2s), 'm', label='CS2')
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$\cos\theta$')

    ax1.set_title(r'$I = %d^\circ$' % np.degrees(I))
    ax1.legend(loc='right')

    # plot areas
    etas_pre_cross = eta_areas[np.where(t_areas < t_end_lib)[0]]
    etas_crossed = eta_areas[np.where(t_areas >= t_end_lib)[0]]
    # theory
    lw = 2
    if plot_type == '23':
        ax3.semilogx(etas_pre_cross, np.full_like(etas_pre_cross, a_init_int),
                     'r', label='Th', lw=lw)
        ax3.semilogx(etas_crossed, np.full_like(etas_crossed, a_crossed23), 'r',
                     lw=lw)
    elif plot_type == '21':
        ax3.semilogx(etas_pre_cross, np.full_like(etas_pre_cross, a_init_int),
                     'r', label='Th', lw=lw)
        ax3.semilogx(etas_crossed, np.full_like(etas_crossed, a_crossed21), 'r',
                     lw=lw)
    elif plot_type == '31':
        ax3.semilogx(etas_pre_cross, np.full_like(etas_pre_cross, a_init_int),
                     'r', label='Th', lw=lw)
        opt_func = lambda eta: sum(sep_areas_exact(I, eta)[ :2]) - a_init_int
        eta_guess = 0.1 # whatever, easy to find
        eta_cross = opt.newton(opt_func, x0=eta_guess)
        a_crossed31 = -sep_areas_exact(I, eta_cross)[0]
        ax3.semilogx(etas_crossed, np.full_like(etas_crossed, a_crossed31), 'r',
                     lw=lw)
    elif plot_type == '321':
        # It's a bit trickier to find both crossings, so we will seed
        # the search using plot_idxs, which gives the two jumps in areas
        opt_func = lambda eta: sum(sep_areas_exact(I, eta)[ :2]) - a_init_int
        eta_guess = eta_areas[plot_idxs[1]]
        eta32 = opt.newton(opt_func, x0=eta_guess)

        opt_func2 = lambda eta: sep_areas_exact(I, eta)[1] - \
                sep_areas_exact(I, eta32)[1]
        try: # try to find a second, smaller value of eta with A2 = A2(eta32)
            eta21 = opt.bisect(opt_func2, 0, eta32 * 0.99)
        except: # already have the smaller value
            eta21 = eta32
            eta32 = opt.bisect(opt_func2, eta21 * 1.01, eta_guess)
        print('etas', eta_guess, eta21, eta32)
        a_crossed32 = sep_areas_exact(I, eta32)[1]
        a_crossed21 = -sep_areas_exact(I, eta21)[0]

        etas_pre_cross = eta_areas[np.where(t_areas < t_end_circ)[0]]
        etas_32 = eta_areas[np.where(np.logical_and(
            t_areas < t_end_lib,
            t_areas >= t_end_circ))[0]]
        # etas21 = etas_crossed
        ax3.semilogx(etas_pre_cross, np.full_like(etas_pre_cross, a_init_int),
                     'r', label='Th', lw=lw)
        ax3.semilogx(etas_32, np.full_like(etas_32, a_crossed32), 'r', lw=lw)
        ax3.semilogx(etas_crossed, np.full_like(etas_crossed, a_crossed21), 'r',
                     lw=lw)

    # data
    ax3.semilogx(eta_areas, areas, 'bo', markersize=0.3, label='Dat')
    ax3.set_xlabel(r'$\eta$')
    ax3.set_ylabel(r'$A$')
    # figure out where to place the legend
    if areas[-1] > areas[0]:
        ax3.legend(loc='lower right')
    else:
        ax3.legend(loc='upper right')

    for ax in [ax1, ax3]:
        for plt_idx in plot_idxs:
            ax.axvline(eta_areas[plt_idx], c='g', lw=0.5)

    # flip axes
    xlims = ax3.set_xlim()
    xlims2 = ax3.set_xlim(xlims[::-1])

    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)

    # separately:
    # plot snapshots @ every jump in areas
    n_rows = (len(plot_idxs) + 1) // 2
    fig, axs_nest = plt.subplots(n_rows, 2, sharex=True, sharey='row',
                                 figsize=(6, 4))
    axs = [ax for sublist in axs_nest for ax in sublist]
    if len(plot_idxs) % 2 == 1:
        fig.delaxes(axs[-1])
        del axs[-1]
    # cache some values to reuse once ylims are all set
    vals_arr = []
    for plot_idx, ax in zip(plot_idxs, axs):
        t_vals = np.linspace(t_areas[plot_idx], t_areas_f[plot_idx], 201)
        q_vals, phi_vals = to_ang(*ret.sol(t_vals)[ :3])
        ms = 1.0 if len(t_vals) < 50 else 0.2

        # plot CS2 as well
        curr_roots = roots(I, eta_areas[plot_idx])
        q2 = curr_roots[0] if len(curr_roots) == 2 else curr_roots[1]
        ax.plot(np.pi, np.cos(q2), 'mo', ms=2)

        # plot data + overplot an arrow to show direction
        ax.plot(phi_vals, np.cos(q_vals), ls='', marker='o', c='y', markersize=ms)
        arrow_idx = len(t_vals) // 2
        width_base = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.arrow(phi_vals[arrow_idx], np.cos(q_vals[arrow_idx]),
                 phi_vals[arrow_idx + 1] - phi_vals[arrow_idx],
                 np.cos(q_vals[arrow_idx + 1]) - np.cos(q_vals[arrow_idx]),
                 color='y', width=0,
                 head_width=0.056 * width_base, head_length=0.12)
        vals_arr.append((t_vals, q_vals, phi_vals))

    for plot_idx, ax, vals in zip(plot_idxs, axs, vals_arr):
        t_vals, q_vals, phi_vals = vals

        # shade the enclosed phase space area
        ylims = ax.get_ylim()
        ax.set_ylim(ylims[0], min(ylims[1], 1)) # max ylim is 1
        _phi_uw = np.unwrap(phi_vals)
        # make sure its median is in [0, 2 * np.pi]
        phi_uw = _phi_uw - np.median(_phi_uw)\
            + ((np.median(_phi_uw) + 2 * np.pi) % (2 * np.pi))
        if abs(phi_uw[-1] - phi_uw[0]) > np.pi:
            # circulating, if phi increasing, shade grey, else red
            fill_bound = ylims[1]
            if np.mean(np.gradient(phi_uw)) > 0:
                c = '0.5'
                a = 0.5
            else:
                c = 'r'
                a = 0.25
            ax.fill_between(phi_uw, fill_bound, np.cos(q_vals),
                            color=c, alpha=a)
        else:
            # librating; starts at top of circle, goes to bottom. easier to
            # handle if starts on LHS, so shift by 1/4
            shift_idx = np.argmin(phi_uw)
            phi_rolled = np.roll(phi_uw, -shift_idx) # brings argmin to idx 0
            q_rolled = np.roll(q_vals, -shift_idx)
            idx_turnaround = np.argmax(phi_rolled)
            # interpolate over max/min, but drop endpoints when evaluating to
            # avoid out of range errors

            # contains neither max nor min value
            phi_half2 = phi_rolled[idx_turnaround + 1: ]
            phi_half1 = phi_rolled[ :idx_turnaround + 1]
            mu_half2 = np.cos(q_rolled[idx_turnaround + 1: ])
            # interpolate containing both max and min val
            mu_half1_interp = interp1d(phi_half1,
                                       np.cos(q_rolled[ :idx_turnaround + 1]))
            mu_half1 = mu_half1_interp(phi_half2)
            ax.fill_between(phi_half2, mu_half1, mu_half2,
                            color='0.5', alpha=0.5)

    # now use the ylims to compute only the viewable separatrix
    for plot_idx, ax in zip(plot_idxs, axs):
        ylims = ax.get_ylim()
        # overplot separatrix + CS2
        eta_plots = [eta_areas[plot_idx], eta_areas[plot_idx + 1]]
        for eta_plot, style in zip(eta_plots, ['solid', 'dashed']):
            curr_roots = roots(I, eta_plot)
            if len(curr_roots) == 4:
                q4 = curr_roots[3]
                H4 = H(I, eta_plot, q4, 0)
                phi_grid, q_grid = np.meshgrid(
                    np.linspace(0, 2 * np.pi, 201),
                    np.arccos(np.linspace(ylims[0], ylims[1], 801)))
                H_grid = H(I, eta_plot, q_grid, phi_grid)
                ax.contour(phi_grid, np.cos(q_grid), H_grid, levels=[H4],
                           colors='k', linewidths=1.0, linestyles=style)
    axs[0].set_ylabel(r'$\cos \theta$')
    axs[-2].set_xlabel(r'$\phi$')
    for ax in axs[-2: ]:
        ax.set_xlim([0, 2 * np.pi])
        ax.set_xticks([np.pi])
        ax.set_xticklabels([r'$\pi$'])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename + '_subplots', dpi=dpi)
    plt.close(fig)
    print('Saved', filename, 'eta_c =', eta_c)

def sim_ring(I, eta, dq):
    q2, _ = roots(I, eta)
    tf = 4 * np.pi / eta # 2pi / eta = precession about CS2
    y0_ring = [*to_cart(q2 + dq, 0), eta]
    event = lambda t, y: (to_ang(*y[0:3])[1] % 2 * np.pi) - np.pi
    event.direction = +1
    return solve_ic_base(I, 0, y0_ring, tf, events=[event])

def fetch_ring(I, eta, dq, n_pts):
    ret_ring = sim_ring(I, eta, dq)
    # either initially librating, so 2 t_events, or circulating, in which case
    # we sample over two orbits, not the worst thing
    t_ring = np.linspace(0, ret_ring.t_events[-1][2], n_pts)
    return ret_ring.sol(t_ring)

def get_ring_area(ret):
    '''
    get the enclosed area for a ring of ICs, 2pi * (1 - cos q) isn't accurate
    enough For testing purposes, I = 20deg, dq = 80deg, anal area is 5.19212,
    integrated is 3.61797

    A third area-getter since I really didn't bother building out good
    abstractions whoops
    '''
    events = ret.t_events[-1]
    x_events, _, _, _ = ret.sol(events)
    t_0 = events[np.where(x_events < 0)[0]]
    t_pi = events[np.where(x_events > 0)[0]]
    if len(t_0) > 0:
        # circulating case
        t_vals = get_t_vals(t_0[0], t_0[1])
        x, y, z, _ = ret.sol(t_vals)
        q, phi = to_ang(x, y, z)
        dphi = np.gradient(np.unwrap(phi))
        return np.sum((1 - np.cos(q)) * dphi)
    else:
        # librating case
        t_vals = get_t_vals(t_pi[0], t_pi[2])
        x, y, z, _ = ret.sol(t_vals)
        q, phi = to_ang(x, y, z)
        dphi = np.gradient(np.unwrap(phi))
        return np.sum(np.cos(q) * dphi)

def sim_for_dq(I, eps=-1e-3, eta0_mult=10, etaf=1e-3, n_pts=10, dq=0.3,
               plot=False):
    '''
    for a small dq (dtheta) displacement below CS2, simulate forwards in time
    for a ring of ICs, look @ final mu values (terminate @ eta = 1e-3)
    '''
    eta0 = eta0_mult * get_etac(I)
    print('Running for', dq)

    # start w/ a ring of constant J_init. calc via eps=0 sim
    y0s = fetch_ring(I, eta0, dq, n_pts)

    # run for each traj
    final_mus = []
    for y0 in y0s.T:
        term_event = lambda t, y: y[3] - 1e-5
        term_event.terminal = True
        ret = solve_ic_base(I, eps, y0, np.inf, events=[term_event])
        q, phi = to_ang(*ret.y[0:3])
        if plot:
            plt.plot(phi[-1], np.cos(q[-1]), 'go', markersize=0.5)
            q0, phi0 = to_ang(*y0[0:3])
            plt.plot(phi0, np.cos(q0), 'ro', markersize=0.5)
        final_mus.append(np.mean(np.cos(q[-20: ])))
    if plot:
        plt.savefig('3single_%02d' % np.degrees(I))
        plt.clf()
        print('Final Mus', final_mus)
    return final_mus

def plot_anal_qfs(I, dqs, res_arr, eta0, axs, two_panel=True):
    '''
    for a given initial mutual inclination array dqs, return a list of tuples to
    plot for analytical location of final qs
    '''
    if two_panel:
        ax1, ax2 = axs
    else:
        [ax1] = axs
    A_sep_crit = 4 * np.pi * (1 - (1 + np.tan(I)**(2/3))**(-3/2))
    get_a2 = lambda eta: -sep_areas_exact(I, eta)[1]
    # if eta_3 < max_A2_eta, will experience double crossing
    max_A2_eta = opt.minimize_scalar(get_a2, 0.99 * get_etac(I),
                                     bounds=(0, get_etac(I)),
                                     method='bounded').x
    get_a3 = lambda eta: sep_areas_exact(I, eta)[2]
    min_A3_eta = opt.minimize_scalar(get_a3, 0,
                                     bounds=(0, get_etac(I)),
                                     method='bounded').x
    min_A3 = get_a3(min_A3_eta)

    # get initial areas more exactly
    areas_init = []
    for dq in dqs:
        ret_ring = sim_ring(I, eta0, dq)
        areas_init.append(abs(get_ring_area(ret_ring)))

    eta_2 = [] # etas that start in A2 when sep forms
    eta_3 = [] # start in A3
    idx_2 = []
    idx_3 = []
    idx_33 = [] # never leave A3
    for idx, j_init in enumerate(areas_init):
        if j_init > A_sep_crit:
            j_init_comp = 4 * np.pi - j_init
            # root find for eta when A3 = 4 * np.pi - j_init
            opt_func = lambda eta: sep_areas_exact(I, eta)[2] - j_init_comp
            eta_guess = ((
                4 * np.sqrt(np.sin(I)) + np.sqrt(
                    16 * np.sin(I) + 2 * np.pi * np.cos(I) * j_init)
            ) / (2 * np.pi * np.cos(I)))**2
            # do not guess higher than eta_c, will diverge area
            eta_guess = min(eta_guess, 0.999 * get_etac(I))
            try:
                root = opt.bisect(opt_func, 0, eta_guess)
                idx_3.append(idx)
                # ensure caught highest root
                try:
                    higher_root = opt.bisect(opt_func, 1.001 * root, eta_guess)
                    eta_3.append(higher_root)
                except:
                    eta_3.append(root)
            except:
                print('Could not find crossing eta', j_init, I, min_A3)
                idx_33.append(idx)
                continue
        else:
            # root find for eta when A2 = j_init
            opt_func = lambda eta: sep_areas_exact(I, eta)[1] - j_init
            eta_guess = (j_init / 16)**2 / np.sin(I)
            eta_2.append(opt.newton(opt_func, x0=eta_guess))
            idx_2.append(idx)
    eta_2 = np.array(eta_2)
    eta_3 = np.array(eta_3)

    # A2 -> A3, cutoff when probability < 0
    idx_23 = np.where(2 * np.sqrt(eta_2 * np.sin(I)) >
                            np.pi * eta_2 * np.cos(I))[0]
    eta_23 = eta_2[idx_23]
    q_f_23 = [np.arccos(sep_areas_exact(I, eta)[2] / (2 * np.pi) - 1)
              for eta in eta_2[idx_23]]
    ax1.plot(np.degrees(dqs[idx_23]), np.degrees(q_f_23), 'r',
             label=r'$2\to3$')

    # A2 -> A1, all eta_2s
    q_f_21 = [np.arccos(sum(sep_areas_exact(I, eta)[1: ]) / (2 * np.pi) - 1)
              for eta in eta_2]
    ax1.plot(np.degrees(dqs[idx_2]), np.degrees(q_f_21), 'c',
             label=r'$2\to1$')

    # A3 -> A1, J_f = A2_star + A3_star
    q_f_31 = [np.arccos(sum(sep_areas_exact(I, eta)[1: ]) / (2 * np.pi) - 1)
              for eta in eta_3]
    ax1.plot(np.degrees(dqs[idx_3]), np.degrees(q_f_31), 'g',
             label=r'$3\to1$')

    # A3 -> A2 -> A1
    eta_second_cross = []
    idx_second_cross = []
    for idx, eta in enumerate(eta_3):
        if eta < max_A2_eta:
            continue
        # find second value of eta w/ same A2 as current eta < max_A2_eta
        equals_first_a = lambda eta_2: sep_areas_exact(I, eta_2)[1]\
            - sep_areas_exact(I, eta)[1]
        eta_second_cross.append(opt.bisect(equals_first_a, 0, max_A2_eta))
        idx_second_cross.append(idx)
    q_f_321 = [np.arccos(sum(sep_areas_exact(I, eta)[1: ]) / (2 * np.pi) - 1)
               for eta in eta_second_cross]
    ax1.plot(np.degrees(dqs[idx_3][idx_second_cross]), np.degrees(q_f_321),
             'm', label=r'$3\to2\to1$')

    # A3 -> A3 (probably not interesting)
    # ax1.plot(np.degrees(dqs[idx_33]), np.degrees(dqs[idx_33]), 'c')

    ax1.legend()
    if not two_panel:
        return

    # overplot the simplest analytical estimate
    # this is the square root motivated one, second simplest
    # naive_areas_init = 2 * np.pi * (1 - np.cos(dqs))
    # naive_eta_cross = (naive_areas_init / 16)**2 / np.sin(I)
    # mu_f_1 = naive_eta_cross * np.cos(I) + 4 * np.sqrt(
    #     naive_eta_cross * np.sin(I)) / np.pi
    # mu_f_2 = naive_eta_cross * np.cos(I) - 4 * np.sqrt(
    #     naive_eta_cross * np.sin(I)) / np.pi
    # this is the power laws one, absolutely simplest
    mu_f_1 = (np.pi * dqs**2 / 16)**2 / np.tan(I) + dqs**2 / 4
    mu_f_2 = (np.pi * dqs**2 / 16)**2 / np.tan(I) - dqs**2 / 4
    ax1.plot(np.degrees(dqs), np.degrees(np.arccos(mu_f_1)),
             'k:', linewidth=0.6)
    ax1.plot(np.degrees(dqs), np.degrees(np.arccos(mu_f_2)),
             'k:', linewidth=0.6)

    # plot probabilities
    # Probability 2 -> 3
    _, dA2_23, dA3_23 = d_sep_areas(I, eta_23)
    P_23 = -dA3_23 / dA2_23
    ax2.plot(np.degrees(dqs[idx_23]), np.minimum(P_23, np.ones_like(P_23)), 'r')
    # Probability 2 -> 1
    dA1_21, dA2_21, _ = d_sep_areas(I, eta_2)
    P_21 = -dA1_21 / dA2_21
    ax2.plot(np.degrees(dqs[idx_2]), np.minimum(P_21, np.ones_like(P_21)), 'c')
    # Prob 3 -> 2 -> 1 = 3 -> 2
    eta_321 = eta_3[np.where(eta_3 > max_A2_eta)[0]]
    _, dA2_32, dA3_32 = d_sep_areas(I, eta_321)
    P_32 = -dA2_32 / dA3_32
    ax2.plot(np.degrees(dqs[idx_3][idx_second_cross]),
             np.minimum(P_32, np.ones_like(P_32)), 'm')
    # Prob 3 -> 1
    dA1_31, _, dA3_31 = d_sep_areas(I, eta_3)
    P_31 = -dA1_31 / dA3_31
    ax2.plot(np.degrees(dqs[idx_3]),
             np.minimum(P_31, np.ones_like(P_31)), 'g')

    # plot data-generated points for probabilities
    for eta, q23, q21, dq, final_mus in\
            zip(eta_23, q_f_23, q_f_21, dqs, res_arr):
        # figure out which of q23, q21 we are closer to
        count_21 = 0
        for final_mu in final_mus:
            if abs(final_mu - np.cos(q21)) < abs(final_mu - np.cos(q23)):
                count_21 += 1
        ax2.scatter(np.degrees(dq), count_21 / len(final_mus), c='b', s=0.8)
        ax2.scatter(np.degrees(dq), 1 - (count_21 / len(final_mus)),
                    c='b', s=0.8)
    idxs_321_total = np.arange(len(dqs))[idx_3][idx_second_cross]
    for eta, q321, q31, idx in\
            zip(eta_321, q_f_321, q_f_31, idxs_321_total):
        # figure out which of q23, q21 we are closer to
        count_321 = 0
        for final_mu in res_arr[idx]:
            if abs(final_mu - np.cos(q321)) < abs(final_mu - np.cos(q31)):
                count_321 += 1
        ax2.scatter(np.degrees(dqs[idx]), count_321 / len(final_mus), c='b', s=0.8)
        ax2.scatter(np.degrees(dqs[idx]), 1 - (count_321 / len(final_mus)),
                    c='b', s=0.8)

def plot_ICs(I, eta0, dqs, n_pts):
    q2, _ = roots(I, eta0)
    norm = cm.colors.Normalize(vmin=min(np.degrees(dqs)),
                               vmax=max(np.degrees(dqs)))
    cmap = cm.RdBu
    for dq in dqs:
        y0s = fetch_ring(I, eta0, dq, n_pts)
        q, _phi = to_ang(*y0s[0:3])
        phi = np.unwrap(_phi)
        gt_idx = np.where(phi > 2 * np.pi)[0]
        lt_idx = np.where(phi < 2 * np.pi)[0]
        if len(gt_idx):
            q = np.concatenate((q[gt_idx], q[lt_idx]))
            phi = np.concatenate((phi[gt_idx] - 2 * np.pi, phi[lt_idx]))
        plt.plot(phi, np.cos(q),
                 c=cmap(norm(np.degrees(dq))),
                 marker='o', markersize=1,
                 linewidth=0.5)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\cos\theta$')
    plt.scatter(np.pi, np.cos(q2), c='g')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))

def sim_for_many(I, eps=-1e-3, eta0_mult=10, etaf=1e-3, n_pts=21, n_dqs=51,
                 adiabatic=True, two_panel=True,
                 dqmin=0.05, dqmax = 0.99 * np.pi / 2):
    '''
    variable number of points per ring,
    starting from n_pts/4 to n_pts linearly over dqs

    plot sets whether to make a probability plot & overplot all predictions
    '''
    I_deg = round(np.degrees(I))
    eps_log = 10 * (-np.log10(-eps))
    eta0 = eta0_mult * get_etac(I)
    dqs = np.linspace(dqmin, dqmax, n_dqs)

    PKL_FN = '3dat%02d_%02d.pkl' % (I_deg, eps_log)
    filename = '3_ensemble_%02d_%02d' % (I_deg, eps_log)
    title = r'$I = %d^\circ, \epsilon = 10^{-%.1f}$' % (I_deg, eps_log / 10)

    # run/plot sim
    if not os.path.exists(PKL_FN):
        p = Pool(4)
        args = [(I, eps, eta0_mult, etaf, n_pts, dq) for dq in dqs]
        res_arr = p.starmap(sim_for_dq, args)
        with open(PKL_FN, 'wb') as f:
            pickle.dump(res_arr, f)
    else:
        with open(PKL_FN, 'rb') as f:
            res_arr = pickle.load(f)

    if adiabatic:
        # first, plot some analytical curves (data overlaid on top)
        # then plot data
        if two_panel:
            fig, (ax1, ax2) = plt.subplots(2, 1,
                                    sharex=True,
                                    figsize=(6, 8),
                                    gridspec_kw={'height_ratios': [3, 2]})
            fig.subplots_adjust(hspace=0)
            plot_anal_qfs(I, dqs, res_arr, eta0, [ax1, ax2], two_panel)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
            plot_anal_qfs(I, dqs, res_arr, eta0, [ax1], two_panel)

        # now plot data
        for dq, final_mus in zip(dqs, res_arr):
            final_q_deg = np.degrees(np.arccos(final_mus))
            ax1.scatter(np.full_like(final_mus, np.degrees(dq)),
                        final_q_deg, c='b', s=0.8)

        ax1.set_yticks([0, 45, 90])
        ax1.set_yticklabels([r'$0$', r'$45$', r'$90$'])
        ax1.set_ylabel(r'$\theta_{f}$ (deg)')
        ax1.set_title(title)

        if two_panel:
            ax2.set_xticks([0, 30, 60, 90], [r'$0$', r'$30$', r'$60$', r'$90$'])
            ax2.set_yticks([0, 0.5, 1], [r'$0$', r'$0.5$', r'$1$'])

            ax2.set_xlabel(r'$\theta_{sd,i}$')
            ax2.set_ylabel('Prob')
        else:
            ax1.set_xticks([0, 30, 60, 90])
            ax1.set_xticklabels([r'$0$', r'$30$', r'$60$', r'$90$'])

            ax1.set_xlabel(r'$\theta_{sd,i}$')

    else:
        # just plot data
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_xlabel(r'$\theta_{sd,i}$')
        ax1.set_ylabel(r'$\theta_{f}$ (deg)')
        ax1.set_xticks([0, 30, 60, 90], [r'$0$', r'$30$', r'$60$', r'$90$'])
        ax1.set_yticks([0, 45, 90], [r'$0$', r'$45$', r'$90$'])
        ylims = plt.ylim()

        # fit = dong's est +- linear
        dong_est_deg = np.degrees(np.sqrt(2 * np.pi / (-eps)) * np.tan(I))
        dqs_d = np.degrees(dqs)
        ax1.plot(dqs_d, dong_est_deg + (dqs_d - min(dqs_d)), 'r:')
        ax1.plot(dqs_d, dong_est_deg - (dqs_d - min(dqs_d)), 'r:')

        ax1.set_title(title)
        plt.ylim(ylims)
    for dq, final_mus in zip(dqs, res_arr):
        final_q_deg = np.degrees(np.arccos(final_mus))
        ax1.scatter(np.full_like(final_mus, np.degrees(dq)),
                    final_q_deg, c='b', s=0.8)
    if not two_panel or not adiabatic:
        plt.tight_layout() # not sure why this is necessary, xlabel cut off
    plt.savefig(filename, dpi=dpi)
    plt.clf()

    # plot all ICs
    # inits_fn = '3_ensemble_inits%02d_%02d' % (I_deg, eps_log)
    # plot_ICs(I, eta0, dqs, n_pts)
    # plt.savefig(inits_fn, dpi=dpi)
    # plt.clf()

def eps_scan(I, filename='3scan', dq=0.01, n_pts=151, n_pts_ring=21,
             eps_min=1e-2, eps_max=0.5):
    '''
    scan for theta_f for small dq @ various epsilons
    '''
    eta0 = 10 * get_etac(I)
    q2, _ = roots(I, eta0)
    y0s = fetch_ring(I, eta0, dq, n_pts_ring)
    pts = []
    eps_vals = np.exp(np.linspace(np.log(eps_max), np.log(eps_min), n_pts))
    PKL_FN = '%s.pkl' % filename
    if not os.path.exists(PKL_FN):
        qdeg_finals = []
        for y0 in y0s.T:
            qdeg_per = []
            for eps in eps_vals:
                term_event = lambda t, y: y[3] - 1e-5
                term_event.terminal = True
                ret = solve_ic_base(I, -eps, y0, np.inf, events=[term_event])
                q, phi = to_ang(*ret.y[ :3, -10: ])
                qdeg_per.append(np.degrees(np.mean(q)))
            qdeg_finals.append(qdeg_per)
        with open(PKL_FN, 'wb') as f:
            pickle.dump(qdeg_finals, f)
    else:
        with open(PKL_FN, 'rb') as f:
            qdeg_finals = pickle.load(f)
    for qdeg_per in qdeg_finals:
        plt.semilogx(eps_vals, qdeg_per, 'bo', ms=0.5)

    s_final = np.sqrt(2 * np.pi / eps_vals) * np.tan(I)
    q_final = np.maximum(np.degrees(s_final * np.cos(I)), np.degrees(q2))
    ylims = plt.ylim()
    plt.semilogx(eps_vals, q_final, 'r:', label='Analytical')
    plt.axvline(np.sin(I), c='k')
    plt.fill_betweenx(ylims,
                      np.sin(I), eps_min, color='0.5', alpha=0.5)
    plt.xlim([eps_min, eps_max])
    plt.ylim(ylims)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\theta_{ f}$')
    plt.title(r'$I = %d^\circ$' % np.degrees(I))
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename, dpi=dpi)
    plt.clf()

def I_scan(eps, filename='3Iscan', dq=0.01, n_pts=151, n_pts_ring=21,
             I_min=np.radians(1), I_max=np.radians(20)):
    '''
    scan for theta_f for small dq @ various epsilons
    '''
    pts = []
    I_vals = np.linspace(I_min, I_max, n_pts)
    I_degs = np.degrees(I_vals)
    eta_vals = 10 * get_etac(I_vals)
    q2s = []
    for I, eta0 in zip(I_vals, eta_vals):
        q2, _ = roots(I, eta0)
        q2s.append(q2)
        y0s = fetch_ring(I, eta0, dq, n_pts_ring)
        qdeg_finals = []
        for y0 in y0s.T:
            term_event = lambda t, y: y[3] - 1e-5
            term_event.terminal = True
            ret = solve_ic_base(I, -eps, y0, np.inf, events=[term_event])
            q, phi = to_ang(*ret.y[ :3, -10: ])
            qdeg_finals.append(np.degrees(np.mean(q)))
        plt.plot(np.full_like(qdeg_finals, np.degrees(I)), qdeg_finals,
                     'ko', ms=0.5)

    s_final = np.sqrt(2 * np.pi / eps) * np.tan(I_vals)
    plt.plot(I_degs, np.degrees(s_final), 'r', label='Analytical')
    q_final = np.maximum(np.degrees(s_final * np.cos(I)), np.degrees(q2s))
    plt.plot(I_degs, q_final, 'b', label='Analytical improved')
    old_lims = plt.ylim()
    plt.ylim([old_lims[-1], 0])
    plt.xlabel(r'$I$')
    plt.ylabel(r'$\theta_{f}$')
    plt.legend()
    plt.savefig(filename, dpi=dpi)
    print('Saved', filename)
    plt.clf()

def plot_singles(I):
    tf = np.inf
    eta0 = 10 * get_etac(I)
    q2, _ = roots(I, eta0)

    eta_f = 3e-4
    term_event = lambda t, y: y[3] - eta_f
    term_event.terminal = True
    events = [term_event]

    plot_single(I, -0.1, tf, eta0, q2, '3testo_nonad', dq=0.3,
                events=events)
    plot_single(I, -3e-4, tf, eta0, q2, '3testo23', plot_type='23', dq=0.3,
                events=events)
    plot_single(I, -3.01e-4, tf, eta0, q2, '3testo21', plot_type='21', dq=0.3,
                events=events)
    plot_single(I, -3.14e-4, tf, eta0, q2, '3testo321', plot_type='321',
                dq=np.radians(60), num_snapshots=2, events=events)
    plot_single(I, -3.01e-4, tf, eta0, q2, '3testo31', plot_type='31',
                dq=0.99 * np.pi / 2, events=events)

def plot_manys(I):
    sim_for_many(I, eps=-3e-4, n_pts=101, n_dqs=51)
    sim_for_many(np.radians(10), eps=-3e-4, n_pts=101, n_dqs=51,
                 two_panel=False)
    sim_for_many(np.radians(20), eps=-3e-4, n_pts=101, n_dqs=51,
                 two_panel=False)
    sim_for_many(I, eps=-1e-3, n_pts=101, n_dqs=51, two_panel=False)
    sim_for_many(I, eps=-3e-3, n_pts=101, n_dqs=51, two_panel=False)
    sim_for_many(I, eps=-1e-2, n_pts=101, n_dqs=101, dqmin=0.01,
                 two_panel=False)
    sim_for_many(I, eps=-3e-2, n_pts=101, n_dqs=101,
                 two_panel=False, dqmin=0.01)

    sim_for_many(I, eps=-3e-1, n_pts=101, n_dqs=101,
                 adiabatic=False, dqmin=0.01)
    sim_for_many(I, eps=-2e-1, n_pts=101, n_dqs=101,
                 adiabatic=False, dqmin=0.01)
    sim_for_many(I, eps=-1e-1, n_pts=101, n_dqs=101,
                 adiabatic=False, dqmin=0.01)

if __name__ == '__main__':
    I = np.radians(5)
    # plot_singles(I)
    plot_manys(I)

    # testing high epsilon
    # eta_c = get_etac(I)
    # y0 = [*to_cart(q2 + 0.01, 0), eta0]
    # term_event = lambda t, y: y[3] - 1e-5
    # term_event.terminal = True
    # ret = solve_ic_base(I, -5, y0, np.inf, events=[term_event])
    # plot_traj_colors(I, ret, '3testo_inf')

    # eps_scan(I, eps_max=2, eps_min=1e-2, n_pts=301, filename='3scan')
    # eps_scan(np.radians(20), eps_max=10, eps_min=5e-2,
    #          n_pts=301, filename='3scan_20')
    # I_scan(5, filename='3Iscan_test', n_pts=31, n_pts_ring=2)
    # I_scan(1, filename='3Iscan_test_eps1', n_pts=31, n_pts_ring=2)
    # I_scan(20, filename='3Iscan_test_eps20', n_pts=31, n_pts_ring=2)
