'''
plot some areas
'''
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('lines', lw=3.0)
plt.rc('font', family='serif', size=20)

def plot_A_crit():
    '''
    Plot area enclosed by separatrix at eta_crit
    '''
    I = np.linspace(0, 5, 21)**2
    area_frac = 1 - (1 + np.tan(np.radians(I))**(2/3))**(-3/2)
    fig, ax1 = plt.subplots(1, 1)
    l1 = ax1.plot(I, area_frac, 'r', label='Areas')
    ax1.set_xlabel(r'$I$ (deg)')
    ax1.set_ylabel(r'$A_{crit}/4\pi$')
    ax2 = ax1.twinx()

    # @ cross, area is (A_3 - 2pi) / 2pi, while A_3 = 4pi * (1 - area_frac)
    q_f = ((1 - area_frac) * 4 * np.pi - 2 * np.pi) / (2 * np.pi)
    l2 = ax2.plot(I, np.degrees(np.arccos(q_f)), 'b',
                  label=r'$\theta_{f,\max}$') # TODO
    ax2.set_ylabel(r'$\theta$')

    lns = l1 + l2
    ax2.legend(lns, [l.get_label() for l in lns], loc='upper left', fontsize=8)
    plt.savefig('1_Acrit.png', dpi=400)
    plt.clf()

def get_areas_ward(eta, I):
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

def plot_areas(I=np.radians(5), filename='1_areas'):
    ''' plot exact area + my old approx '''
    eta_c = (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)
    scale_pow = 3
    eta = np.linspace(0, eta_c**(1/scale_pow), 1001)**scale_pow

    A1w, A2w, A3w = get_areas_ward(eta, I)
    A_sep_crit = 1 - (1 + (np.tan(I))**(2/3))**(-3/2)
    # print(A2w.max(), A_sep_crit * 4 * np.pi)
    A2ys = 16 * np.sqrt(eta * np.sin(I))
    A1ys = 2 * np.pi * (1 - eta * np.cos(I)) - A2ys / 2
    A3ys = 2 * np.pi * (1 + eta * np.cos(I)) - A2ys / 2
    fig, ax1 = plt.subplots(1, 1)
    ax3 = ax1.twinx()
    ax4 = ax1.twiny()
    ax1.plot(eta, A1ys / (4 * np.pi), 'g:')
    ax1.plot(eta, A1w / (4 * np.pi), 'g', label=r'$A_{\rm I}$')
    ax1.plot(eta, A2ys / (4 * np.pi), 'b:')
    ax1.plot(eta, A2w / (4 * np.pi), 'b', label=r'$A_{\rm II}$')
    ax1.plot(eta, A3ys / (4 * np.pi), 'r:')
    ax1.plot(eta, A3w / (4 * np.pi), 'r', label=r'$A_{\rm III}$')

    # dotted line continuation to show "analytic" continuation
    eta_cont = np.linspace(eta_c, 1.2 * eta_c, 21)
    ax1.plot(eta_cont, np.full_like(eta_cont, A1w[-1] / (4 * np.pi)), 'g--')
    ax1.plot(eta_cont, np.full_like(eta_cont, A2w[-1] / (4 * np.pi)), 'b--')
    ax1.plot(eta_cont, np.full_like(eta_cont, A3w[-1] / (4 * np.pi)), 'r--')
    xlims = ax1.get_xlim()
    ax1.legend(loc='upper left', fontsize=14,
               bbox_to_anchor=(0.05, xlims[1] + 0.035))
    ax1.set_xlabel(r'$\eta$')
    ax1.set_ylabel(r'$A / 4\pi$')
    ax1.set_xticks([0, 0.3, 0.6, 0.9])
    ax1.set_xticklabels([r'$0$', r'$0.3$', r'$0.6$', r'$0.9$'])
    old_ylims = ax1.get_ylim()
    ax1.set_ylim([0, old_ylims[1]])
    ax3.set_ylim([0, old_ylims[1]])
    ax1.set_xlim([0, 1.2 * eta_c])
    ax4.set_xlim([0, 1.2 * eta_c])

    # plot critical values
    crit_vals_y = [A_sep_crit]
    # for val in crit_vals_y:
    #     ax3.axhline(val, c='k', lw=2.0, ls='dashed')
    ax3.set_yticks(crit_vals_y)
    ax3.set_yticklabels([r'$A_{\rm II}(\eta_c)$'])
    eta_2_max = eta[np.argmax(A2w)]
    eta_3_min = eta[np.argmin(A3w)]
    crit_vals_x = [eta_c, eta_2_max, eta_3_min]
    for val in crit_vals_x:
        ax4.axvline(val, c='k', lw=2.0, ls='dashed')
    ax4.set_xticks(crit_vals_x)
    ax4.set_xticklabels([r'$\eta_c$',
                         r'$\eta_{\max,II}$',
                         r'$\eta_{\min,III}$'])

    plt.title(r'$I = %d^\circ$' % np.degrees(I))
    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.clf()

if __name__ == '__main__':
    # plot_A_crit()
    plot_areas()
    # plot_areas(I=np.radians(20), filename='1_areas20')
