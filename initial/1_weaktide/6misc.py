'''
misc little plots
'''
import numpy as np
from multiprocessing import Pool
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
from scipy.optimize import brenth
from scipy.interpolate import interp1d

POOL_SIZE = 50

from utils import roots, s_c_str, get_mu_equil, solve_ic, to_cart, to_ang,\
    get_H4, H, get_mu4, get_ps_anal, get_anal_caps, get_num_caps, get_etac

def get_cs_val(I, s_c, s):
    '''
    calculates CS1, CS2 locations for an array s
    returns -1 if no CS for that state
    '''
    cs1_qs = np.full_like(s, -1)
    cs2_qs = np.full_like(s, -1)
    for idx, s_val in enumerate(s):
        cs_qs = roots(I, s_c, s_val)
        if len(cs_qs) == 4:
            cs1_qs[idx] = cs_qs[0]
            cs2_qs[idx] = cs_qs[1]
        else:
            cs2_qs[idx] = cs_qs[0]
    return cs1_qs, cs2_qs

def plot_equils(I, s_c, verbose=True, fn_str='6equils%s'):
    ''' plot in (s, mu) space showing how the tCS arise '''
    fig = plt.figure(figsize=(5, 5))
    s_lt = np.linspace(s_c / 10, 1, 200) # max eta = 10
    s_gt = np.linspace(1, 3, 200) # other interesting part of the interval
    s_tot = np.concatenate((s_lt, s_gt))

    cs1_qs, cs2_qs = get_cs_val(I, s_c, s_tot)
    cs1_exist_idx = np.where(cs1_qs > -1)[0]
    mu_equil_lt = [np.degrees(np.arccos(get_mu_equil(s))) for s in s_lt]

    s_dq = np.linspace(2, 3, 200)

    plt.xlabel(r'$\Omega_{\rm s} / n$')
    plt.ylabel(r'$\theta$ (deg)')
    plt.ylim([0, 180])

    s_search_min = np.min(s_tot[cs1_exist_idx])
    dS_interp = interp1d(s_lt, mu_equil_lt)
    CS2_interp = interp1d(s_tot, np.degrees(cs2_qs))
    CS1_interp = interp1d(s_tot[cs1_exist_idx],
                          np.degrees(-cs1_qs[cs1_exist_idx]))
    tce2_s = brenth(lambda s: dS_interp(s) - CS2_interp(s), s_search_min, 1)
    tce1_s = brenth(lambda s: dS_interp(s) - CS1_interp(s), s_search_min, 1)

    if verbose:
        plt.plot(s_tot, np.degrees(cs2_qs), 'tab:green', label='CS2', lw=2.5)
        dash_idx = np.where(s_tot[cs1_exist_idx] > 2)[0]
        solid_idx = np.where(s_tot[cs1_exist_idx] < 2)[0]
        plt.plot(s_tot[cs1_exist_idx][dash_idx],
                 np.degrees(-cs1_qs[cs1_exist_idx][dash_idx]), 'orange',
                 label='CS1', lw=2.5, ls='--')
        plt.plot(s_tot[cs1_exist_idx][solid_idx],
                 np.degrees(-cs1_qs[cs1_exist_idx][solid_idx]), 'orange',
                 label='CS1', lw=2.5)
        plt.text(s_tot[-1], np.degrees(cs2_qs)[-1], 'CS2', c='tab:green',
                 fontsize=14, va='bottom', ha='right')
        plt.text(s_tot[-1], -np.degrees(cs1_qs)[-1], 'CS1', c='orange',
                 fontsize=14, va='bottom', ha='right')
    plt.plot(s_lt, mu_equil_lt, 'k', lw=4)
    plt.plot(s_dq, np.degrees(np.arccos(2 / s_dq)), 'b', lw=4)
    # plt.annotate('', xy=(1.5, 90), xytext=(2.5, 120),
    #              arrowprops={'fc': 'g', 'width': 10, 'headwidth': 25})
    # plt.annotate('', xy=(0.8, 10), xytext=(0.3, 30),
    #              arrowprops={'fc': 'g', 'width': 10, 'headwidth': 25})
    # plt.annotate('', xy=(2.5, 20), xytext=(3.0, 10),
    #              arrowprops={'fc': 'g', 'width': 10, 'headwidth': 25})
    # plt.legend(loc='upper left', bbox_to_anchor=(0.15, 1), fontsize=14)

    # label text along boundaries
    if s_c == 0.2:
        plt.text(0.80, 47, r'$\dot{\Omega}_{\rm s} < 0$', c='k', fontsize=14,
                 rotation=-60)
        plt.text(0.56, 37, r'$\dot{\Omega}_{\rm s} > 0$', c='k', fontsize=14,
                 rotation=-60)

        plt.text(2.45, 43, r'$\dot{\theta}_{\rm tide} < 0$', c='b', fontsize=14,
                 rotation=20)
        plt.text(2.55, 28, r'$\dot{\theta}_{\rm tide} > 0$', c='b', fontsize=14,
                 rotation=20)

    plt.plot(tce2_s, dS_interp(tce2_s), mec='tab:green', mfc='none', marker='o',
             ms=15, mew=3)
    plt.text(tce2_s, dS_interp(tce2_s) + 10, 'tCE2', c='tab:green', va='bottom',
             ha='center')
    plt.plot(tce1_s, dS_interp(tce1_s), mec='orange', mfc='none',
             marker='o', ms=15, mew=3)
    plt.text(tce1_s * 1.1, dS_interp(tce1_s) + 3, 'tCE1', c='orange',
             va='bottom', ha='left')


    plt.tight_layout()
    plt.savefig(fn_str % s_c_str(s_c), dpi=300)
    plt.clf()

def plot_phop(I, s_c):
    s = np.linspace(2 * s_c, 10, 100)
    tops, bots = get_ps_anal(I, s_c, s)
    pc32 = (tops + bots) / bots
    p_caps32 = np.minimum(pc32, np.ones_like(pc32))
    pc12 = (tops + bots) / tops
    p_caps12 = np.minimum(pc12, np.ones_like(pc12))
    plt.ylim(bottom=0)
    plt.xlabel(r'$s / \Omega_1$')
    plt.ylabel('Probability')
    plt.plot(s, p_caps32, label=r'$III \to II$')
    plt.plot(s, p_caps12, label=r'$I \to II$')
    plt.legend()
    plt.savefig('6pc%s' % s_c_str(s_c), dpi=400)

def get_cross_dat(I, s_c, s0, eps, tf, mu0, phi0):
    [mu4] = get_mu4(I, s_c, np.array([s0]))

    H4_0 = get_H4(I, s_c, s0)
    H_0 = H(I, s_c, s0, mu0, phi0)
    if H_0 > H4_0:
        print('Inside separatrix for', mu0, phi0)
        return [-1, 0]

    # stop sim when H = H4
    print('Running for', mu0, phi0)
    init = [*to_cart(np.arccos(mu0), phi0), s0]
    def event(t, y):
        x, y, z, s = y
        _, phi = to_ang(x, y, z)
        H4 = get_H4(I, s_c, s)
        H_curr = H(I, s_c, s, z, phi)
        dH = H_curr - H4
        return dH
    event.terminal = True
    _, _, s, ret = solve_ic(I, s_c, eps, init, tf,
                               rtol=1e-4,
                               dense_output=True,
                               events=[event])
    if ret.t_events[0].size > 0:
        return [s[-1], mu0 - mu4]
    else:
        return [-s[-1], 0]

def plot_equil_dist_anal(I, s_c, s0, eps, tf=8000):
    pkl_fn = '6pc_dist%s.pkl' % s_c_str(s_c)
    n_mu = 501
    n_phi = 50
    mu_vals =  np.linspace(-0.99, 0.99, n_mu)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    if not os.path.exists(pkl_fn):
        # store tuple (s_cross, mu0 - mu4)
        # s_cross convention: -1 = inside separatrix (pcap = 1),
        # negative = - s_final (no encounter; s_f currently unused)
        cross_dat = np.zeros((n_mu, n_phi, 2), dtype=np.float64)

        # build arguments array up from scratch
        args = []
        for idx, mu0 in enumerate(mu_vals):
            for idx2, phi0 in enumerate(phi_vals):
                args.append((I, s_c, s0, eps, tf, mu0, phi0))
                # cross_dat[idx, idx2] = get_cross_dat(I, s_c, s0, eps, tf, mu0, phi0)
        p = Pool(POOL_SIZE)
        res = p.starmap(get_cross_dat, args)
        cross_dat = np.reshape(np.array(res), (n_mu, n_phi, 2))
        with open(pkl_fn, 'wb') as f:
            pickle.dump(cross_dat, f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            cross_dat = pickle.load(f)
    try:
        p_caps_anal = get_anal_caps(I, s_c, cross_dat, mu_vals)
        p_caps = get_num_caps(I, s_c, cross_dat, mu_vals)
        tot_probs_anal = np.sum(p_caps_anal / n_phi, axis=1)
        tot_probs = np.sum(p_caps / n_phi, axis=1)
        plt.plot(mu_vals, tot_probs_anal, 'ro', ms=2, label='Anal')
        plt.plot(mu_vals, tot_probs, 'bo', ms=2, label='Num')
        plt.ylim([0, 1])
        plt.xticks([-1, -0.5, 0, 0.5, 1])
        plt.legend()
        plt.tight_layout()
        plt.savefig('6pc_dist%s' % s_c_str(s_c), dpi=300)
        plt.clf()
    except Exception as e: # on remote, can't plot, just return
        print(e)
        return

if __name__ == '__main__':
    eps = 1e-3
    I = np.radians(20)
    plot_equils(I, 0.2)
    # plot_equils(I, 0.2, verbose=False, fn_str='6equils_half%s')
    # plot_equils(I, 0.06)
    # plot_equils(I, 0.6)
    # plot_phop(I, 0.2)
    # plot_equil_dist_anal(I, 0.06, 10, eps)
    # plot_equil_dist_anal(I, 0.2, 10, eps)
    # plot_equil_dist_anal(I, 0.7, 10, eps)

    # test cases
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0.9, np.pi)) # no cross
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0, np.pi)) # in sep
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0, 0)) # fast cross

    # mu = -0.91
    # phi = np.pi / 2
    # mu_vals = np.array([mu])
    # cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, mu, phi)
    # print(cross_dat)
    # cross_dat_re = np.reshape(np.array(cross_dat), (1, 1, 2))
    # print(get_num_caps(I, 0.7, cross_dat_re, mu_vals))

    # for mu in [0.4, 0.45, 0.5]:
    # for mu in [0.5]:
    #     cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, mu, np.pi)
    #     print(cross_dat)
    #     print(get_num_caps(I, 0.7, np.reshape(np.array(cross_dat), (1, 1, 2))))
    # cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0.468, 1.0471975511965976)
    # print(cross_dat)
    # print(get_num_caps(I, 0.7, np.reshape(np.array(cross_dat), (1, 1, 2))))
