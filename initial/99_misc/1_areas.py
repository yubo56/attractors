'''
plot some areas
'''
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

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
    ax2.legend(lns, [l.get_label() for l in lns],
               loc='upper left', fontsize=8)
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
    print(A2w.max(), A_sep_crit * 4 * np.pi)
    A2ys = 16 * np.sqrt(eta * np.sin(I))
    A1ys = 2 * np.pi * (1 - eta * np.cos(I)) - A2ys / 2
    A3ys = 2 * np.pi * (1 + eta * np.cos(I)) - A2ys / 2
    plt.plot(eta, A1ys / (4 * np.pi), 'g:')
    plt.plot(eta, A1w / (4 * np.pi), 'g', label=r'$A_{I}$')
    plt.plot(eta, A2ys / (4 * np.pi), 'k:')
    plt.plot(eta, A2w / (4 * np.pi), 'k', label=r'$A_{II}$')
    plt.plot(eta, A3ys / (4 * np.pi), 'r:')
    plt.plot(eta, A3w / (4 * np.pi), 'r', label=r'$A_{III}$')
    # plt.axhline(A_sep_crit, c='b', lw=0.6)

    # dotted line continuation to show "analytic" continuation
    eta_cont = np.linspace(eta_c, 1.2 * eta_c, 21)
    plt.plot(eta_cont, np.full_like(eta_cont, A1w[-1] / (4 * np.pi)), 'g--')
    plt.plot(eta_cont, np.full_like(eta_cont, A2w[-1] / (4 * np.pi)), 'k--')
    plt.plot(eta_cont, np.full_like(eta_cont, A3w[-1] / (4 * np.pi)), 'r--')
    plt.legend(loc='upper left')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$A_{sep} / 4\pi$')
    plt.xticks([0, 0.3, 0.6, eta_c, 0.9],
               [r'$0$', r'$0.3$', r'$0.6$', r'$\eta_c$', r'$0.9$'])
    old_ylims = plt.ylim()
    plt.ylim([0, old_ylims[1]])
    plt.xlim([0, 1.2 * eta_c])
    plt.title(r'$I = %d^\circ, \eta_c = %.3f$' % (np.degrees(I), eta_c))
    plt.savefig(filename, dpi=400)
    plt.clf()

if __name__ == '__main__':
    # plot_A_crit()
    plot_areas()
    plot_areas(I=np.radians(20), filename='1_areas20')
