'''
misc little plots
'''
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

from utils import roots, s_c_str, get_mu_equil, solve_ic, to_cart, to_ang,\
    get_H4, H, get_mu4

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

def get_top(I, s_c, s):
    eta = s_c / s
    return s_c / s**2 * (
        -2 * np.cos(I) * (
            2 * np.pi * eta * np.cos(I)
            + (8 * np.sqrt(eta * np.sin(I)))
        )
        + s * np.cos(I) * 2 * np.pi

        + (eta * np.cos(I)) * (
            -8 * np.sqrt(np.sin(I) / eta)
        )
        + (s / 2) * (8 * np.sqrt(np.sin(I) / eta))

        - 4 * np.pi * np.sin(I)
    ) + 2 / s * ( # second term
        -2 * np.pi * (1 - 2 * eta * np.sin(I))
            - (16 * np.cos(I) * eta) * np.sqrt(eta * np.sin(I))
    ) + (
        8 * np.sqrt(eta * np.sin(I))
        + 2 * np.pi * eta * np.cos(I)
        - 64/3 * (eta * np.sin(I))**(3/2)
    )
def get_bot(I, s_c, s):
    eta = s_c / s
    return s_c / s**2 * (
        -2 * np.cos(I) * (
            -2 * np.pi * eta * np.cos(I)
            + (8 * np.sqrt(eta * np.sin(I)))
        )
        - s * np.cos(I) * 2 * np.pi

        + (eta * np.cos(I)) * (
            -8 * np.sqrt(np.sin(I) / eta)
        )
        + (s / 2) * (8 * np.sqrt(np.sin(I) / eta))

        + 4 * np.pi * np.sin(I)
    ) + 2 / s * ( # second term
        +2 * np.pi * (1 - 2 * eta * np.sin(I))
            - (16 * np.cos(I) * eta) * np.sqrt(eta * np.sin(I))
    ) + (
        8 * np.sqrt(eta * np.sin(I))
        - 2 * np.pi * eta * np.cos(I)
        - 64/3 * (eta * np.sin(I))**(3/2)
    )

def plot_phop(I, s_c):
    s = np.linspace(2 * s_c, 10, 100)
    tops = get_top(I, s_c, s)
    bots = get_bot(I, s_c, s)
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

def plot_equil_dist_anal(I, s_c, s0, eps, tf=3000):
    pkl_fn = '6pc_dist%s.pkl' % s_c_str(s_c)
    n_mu = 59
    n_phi = 100
    mu_vals =  np.linspace(-0.9, 0.9, n_mu)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi)

    if not os.path.exists(pkl_fn):
        [mu4] = get_mu4(I, s_c, np.array([s0]))
        def get_s_cross(mu0, phi0):
            init = [*to_cart(np.arccos(mu0), phi0), s0]

            # stop sim when H = H4
            def event(t, y):
                x, y, z, s = y
                _, phi = to_ang(x, y, z)
                H4 = get_H4(I, s_c, s)
                H_curr = H(I, s_c, s, z, phi)
                return H_curr - H4
            event.terminal = True
            _, _, s, ret = solve_ic(I, s_c, eps, init, tf,
                                       rtol=1e-4,
                                       events=[event])
            if ret.t_events[0].size > 0:
                return s[-1]
            else:
                return None

        p_caps = np.zeros((n_mu, n_phi))
        for idx, mu0 in enumerate(mu_vals):
            for idx2, phi0 in enumerate(phi_vals):
                # if inside separatrix, guaranteed
                H4_0 = get_H4(I, s_c, s0)
                H_0 = H(I, s_c, s0, mu0, phi0)
                if H_0 > H4_0:
                    print('Inside separatrix for', mu0, phi0)
                    p_caps[idx, idx2] = 1
                    continue

                print('Running for', mu0, phi0)
                s_cross = get_s_cross(mu0, phi0)
                if s_cross is None:
                    p_caps[idx, idx2] = 0
                    continue
                top = get_top(I, s_c, s_cross)
                bot = get_bot(I, s_c, s_cross)
                if mu0 < mu4:
                    pc = (top + bot) / bot
                else:
                    pc = (top + bot) / top
                p_caps[idx, idx2] = pc
        with open(pkl_fn, 'wb') as f:
            pickle.dump(p_caps, f)
    else:
        print('Loading %s' % pkl_fn)
        with open(pkl_fn, 'rb') as f:
            p_caps = pickle.load(f)
    p_caps = np.minimum(np.maximum(p_caps, np.zeros_like(p_caps)),
                        np.ones_like(p_caps))
    tot_probs = np.sum(p_caps / n_phi, axis=1)
    plt.plot(mu_vals, tot_probs, 'bo', ms=2)
    plt.ylim([0, 1])
    plt.savefig('6pc_dist%s' % s_c_str(s_c), dpi=400)
    plt.clf()

if __name__ == '__main__':
    eps = 1e-3
    I = np.radians(5)
    # plot_equils(I, 0.06)
    # plot_equils(I, 0.2)
    # plot_equils(I, 0.6)
    # plot_phop(I, 0.2)
    plot_equil_dist_anal(I, 0.2, 10, eps)
    plot_equil_dist_anal(I, 0.7, 10, eps)
