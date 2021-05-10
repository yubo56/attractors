import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

# (3 * (8 Mearth) / (2.4 Msun) * (1.5)^3) / (3/2 * (0.6 Msun) / (8 Mearth) * ((2 Rearth) / (0.035 AU))^3 * 0.5)
eta = 0.38 / 2 # kq / k?

def plot_actives():
    ain = np.linspace(0.02, 0.05, 50)
    fig = plt.figure(figsize=(6, 5))
    # heuristic (TODO), fixed spin = n / 2
    eta_1 = 1 / (eta * (ain / 0.035)**(3))
    plt.plot(ain, eta_1, 'b', label=r'$\eta_{\rm sync} = 1$')
    eta_3 = 1/3 / (eta * (ain / 0.035)**(3))
    plt.plot(ain, eta_3, 'g', label=r'$\eta_{\rm sync} = 1/3$')
    # 2 / (3 * 1e-3 * Msun / (8 Mearth) * ((2 Rearth) / (0.045 AU))^5 * 2 * pi / ((0.045)^(3/2)))
    adot_gyr = 3 / (ain / 0.045)**(13/2)
    plt.plot(ain, adot_gyr, 'b--', label=r'$\dot{a} = \mathrm{Gyr}$')

    fill_blue = np.where(adot_gyr > eta_1)[0]
    plt.fill_between(ain[fill_blue], eta_1[fill_blue], adot_gyr[fill_blue],
                     facecolor='b', alpha=0.2)
    fill_green = np.where(adot_gyr > eta_3)[0]
    plt.fill_between(ain[fill_green],
                     np.maximum(eta_1[fill_green], adot_gyr[fill_green]),
                     eta_3[fill_green],
                     facecolor='g', alpha=0.2)

    plt.ylim(1, 15)
    plt.xlim(0.02, 0.05)
    plt.legend(fontsize=14)
    plt.xlabel(r'$a$ [AU]')
    plt.ylabel(r'$M$ [$M_\oplus$]')
    plt.tight_layout()
    plt.savefig('5millholland_actives', dpi=300)
    plt.close()

def plot_eigens():
    # for j in range(3):
    #     for k in range(3):
    #         if j != k:
    #             q = 3 if k > j else 2
    #             print(j, k, ':', j + (2 * np.abs(k - j) * q / 3))
    # return
    gamma = 1 / np.linspace(1.1, 2.0, 20)
    eigs = []
    for g in gamma:
        eigvals = np.linalg.eigvals([
            [-g**2 - g**4, g**2, g**4],
            [g**(7/3), -g**(10/3) - g**3, g**3],
            [g**(14/3), g**(10/3), -g**(14/3) - g**(10/3)]
        ])
        eigs.append(np.max(np.abs(eigvals)))
    plt.plot(1 / gamma, np.array(eigs) / gamma**2)
    plt.xlabel(r'$P_{j + 1} / P_j$')
    plt.ylabel(r'Max Abs Eig $\times (P_{j + 1} / P_j)^2$')
    plt.tight_layout()
    plt.savefig('5eigs', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_eigens()
