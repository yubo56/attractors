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

POOL_SIZE = 50

from utils import roots, s_c_str, get_mu_equil, solve_ic, to_cart, to_ang,\
    get_H4, H, get_mu4, get_ps_anal, get_anal_caps, get_num_caps

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

def plot_equils(I, s_c):
    s_lt = np.linspace(s_c / 10, 1, 200) # max eta = 10
    s_gt = np.linspace(1, 3, 200) # other interesting part of the interval
    s_tot = np.concatenate((s_lt, s_gt))

    cs1_qs, cs2_qs = get_cs_val(I, s_c, s_tot)
    cs1_exist_idx = np.where(cs1_qs > -1)[0]
    plt.plot(s_tot, np.degrees(cs2_qs), 'r', label='CS2')
    plt.plot(s_tot[cs1_exist_idx], np.degrees(-cs1_qs[cs1_exist_idx]),
             'g', label='CS1')
    plt.plot(s_lt, np.degrees(np.arccos(get_mu_equil(s_lt))),
             'k:', label='ds=0')

    s_dq = np.linspace(2, 3, 200)
    plt.plot(s_dq, np.degrees(np.arccos(2 / s_dq)), 'b:', label='dq=0')

    plt.xlabel(r'$s / \Omega_1$')
    plt.ylabel(r'$\theta$')
    plt.ylim([0, 90])
    plt.legend(loc='upper right')
    plt.savefig('6equils%s' % s_c_str(s_c), dpi=400)
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
        # print(dH)
        return dH
    # event.terminal = True
    _, _, s, ret = solve_ic(I, s_c, eps, init, tf,
                               rtol=1e-4, dense_output=True,
                               events=[event])
    plt.plot(ret.y[2, :])
    plt.savefig('/tmp/foo.png')
    plt.clf()
    print(s[-1])
    if ret.t_events[0].size > 0:
        return [ret.sol(ret.t_events[0][0])[3], mu0 - mu4]
    else:
        return [-s[-1], 0]

def plot_equil_dist_anal(I, s_c, s0, eps, tf=8000):
    pkl_fn = '6pc_dist%s.pkl' % s_c_str(s_c)
    n_mu = 101
    n_phi = 60
    mu_vals =  np.linspace(-0.9, 0.9, n_mu)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    if not os.path.exists(pkl_fn):
        # store tuple (s_cross, mu0 - mu4)
        # s_cross convention: -1 = inside separatrix (pcap = 1),
        # negative = - s_final (see which equilibrium ends up at)
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
        # print(cross_dat[76, 10])
        # return
        # for idx, (i, j) in enumerate(zip(p_caps[76, :], phi_vals)):
        #     print(mu_vals[76], idx, i, j)
        # 76, 77, 82, 83 seem to be funky
        # for idx, (i, j) in enumerate(zip(mu_vals, tot_probs)):
        #     print(idx, i, j)
        plt.plot(mu_vals, tot_probs_anal, 'ro', ms=2, label='Anal')
        plt.plot(mu_vals, tot_probs, 'bo', ms=2, label='Num')
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig('6pc_dist%s' % s_c_str(s_c), dpi=400)
        plt.clf()
    except Exception as e: # on remote, can't plot, just return
        print(e)
        return

if __name__ == '__main__':
    eps = 1e-3
    I = np.radians(5)
    # plot_equils(I, 0.06)
    # plot_equils(I, 0.2)
    # plot_equils(I, 0.6)
    # plot_phop(I, 0.2)
    plot_equil_dist_anal(I, 0.06, 10, eps)
    plot_equil_dist_anal(I, 0.2, 10, eps)
    plot_equil_dist_anal(I, 0.7, 10, eps)

    # test cases
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0.9, np.pi)) # no cross
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0, np.pi)) # in sep
    # print(get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0, 0)) # fast cross
    # for mu in [0.4, 0.45, 0.5]:
    # for mu in [0.5]:
    #     cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, mu, np.pi)
    #     print(cross_dat)
    #     print(get_num_caps(I, 0.7, np.reshape(np.array(cross_dat), (1, 1, 2))))
    # cross_dat = get_cross_dat(I, 0.7, 10, 1e-3, 8000, 0.468, 1.0471975511965976)
    # print(cross_dat)
    # print(get_num_caps(I, 0.7, np.reshape(np.array(cross_dat), (1, 1, 2))))
