'''
misc little plots
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

from utils import roots, s_c_str, get_mu_equil

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
    def get_top(s):
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
    def get_bot(s):
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
    s = np.linspace(2 * s_c, 10, 100)
    tops = get_top(s)
    bots = get_bot(s)
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

if __name__ == '__main__':
    I = np.radians(5)
    plot_equils(I, 0.06)
    plot_equils(I, 0.2)
    plot_equils(I, 0.6)
    # plot_phop(I, 0.2)
