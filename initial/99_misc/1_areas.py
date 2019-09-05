'''
plot some areas
'''
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

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

def get_area_ward(eta, I):
    mu4 = eta * np.cos(I) / (1 - eta * np.sin(I))
    q4 = -np.arccos(mu4)

    # WH2004 eq 11-13
    z0 = eta * np.cos(I)
    chi = np.sqrt(-np.tan(q4)**3 / np.tan(I) - 1)
    rho = chi * np.sin(q4)**2 * np.cos(q4) / (
        chi**2 * np.cos(q4)**2 + 1)
    T = 2 * chi * np.cos(q4) / (
        chi**2 * np.cos(q4)**2 - 1)
    A2_ward = 8 * rho + 4 * np.arctan(T) - 8 * z0 * np.arctan(1 / chi)
    return A2_ward

def plot_areas():
    ''' plot exact area + my old approx '''
    eta = np.linspace(0, 0.5, 101)
    I = np.radians(5)

    A2_ward = get_area_ward(eta, I)
    A_mine = 16 * np.sqrt(eta * np.sin(I))
    plt.plot(eta, A_mine / (4 * np.pi), 'g:', label='Mine')
    plt.plot(eta, A2_ward / (4 * np.pi), 'k', label='Ward')
    plt.legend()
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$A_{sep} / 4\pi$')
    plt.savefig('1_areas.png', dpi=400)
    plt.clf()

def plot_eps_dist():
    '''
    plot final theta as function of initial mutual inclination
    '''
    I = np.radians(5)
    eps_init = np.linspace(0, 20, 101)
    j_init = 2 * np.pi * (1 - np.cos(np.radians(eps_init)))

    # set j_init = A2 to find eta @ crossing
    # TODO use ward's formula instead
    eta_c = (j_init / 16)**2 / np.sin(I)
    A3 = 2 * np.pi * (1 + eta_c * np.cos(I)) - j_init / 2

    q_f_1 = (j_init + A3 - 2 * np.pi) / (2 * np.pi)
    q_f_2 = (A3 - 2 * np.pi) / (2 * np.pi)
    plt.plot(eps_init, np.degrees(np.arccos(q_f_1)), 'r')
    plt.plot(eps_init, np.degrees(np.arccos(q_f_2)), 'r')
    plt.xlabel(r'Initial $\epsilon$')
    plt.ylabel(r'Final $\theta$')
    plt.savefig('1_eps_dist.png', dpi=400)
    plt.clf()

if __name__ == '__main__':
    plot_A_crit()
    plot_areas()
    plot_eps_dist()
