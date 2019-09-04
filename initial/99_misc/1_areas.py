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
    l2 = ax2.plot(I, q_f, 'b', label=r'$\cos\theta_f$') # TODO
    ax2.set_ylabel(r'$\cos\theta$')

    lns = l1 + l2
    ax2.legend(lns, [l.get_label() for l in lns],
               loc='upper left', fontsize=8)
    plt.savefig('1_Acrit.png')

if __name__ == '__main__':
    plot_A_crit()
