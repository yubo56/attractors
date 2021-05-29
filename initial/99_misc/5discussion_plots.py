import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

def plot_sehj_region():
    a_vals = np.geomspace(0.2, 0.6, 101)
    ap_vals = np.geomspace(2, 20, 101)
    a_grid = np.outer(a_vals, np.ones_like(ap_vals))
    ap_grid = np.outer(np.ones_like(a_vals), ap_vals)

    fig = plt.figure(figsize=(6, 8))
    ax1 = plt.axes([0.12, 0.55, 0.65, 0.42])
    ax2 = plt.axes([0.12, 0.1, 0.65, 0.42])
    cax = plt.axes([0.79, 0.1, 0.03, 0.87])
    for ax, Id in zip([ax1, ax2], [20, 5]):
        I = np.radians(Id)
        etasync = 0.303 * np.cos(I) * (a_grid / 0.4)**6 / (ap_grid / 5)
        etasync_min = np.floor(np.log10(etasync[0, 0]))
        etasync_max = np.ceil(np.log10(etasync.max()))
        etasync_norm = np.minimum(
                         np.maximum(etasync_min - 0.05, np.log10(etasync)),
                         etasync_max,
                     )
        cf = ax.contourf(a_grid, ap_grid, etasync_norm)

        aout_tsc_crit = (
            (100 * (a_vals / 0.4)**(21/2) * np.sin(I) *
                 np.cos(I)**2)**(2/9) * 5)
        ax.plot(a_vals, aout_tsc_crit, 'k', lw=3)
        ax.fill_between(a_vals, aout_tsc_crit,
                         np.full_like(a_vals, ap_vals.max()),
                         color='w')
        ax.text(0.3, 12, 'tCE2 Unstable', ha='center', va='center',
                 fontsize=16)
        ax.text(a_vals.min() + 0.01, ap_vals.max() - 0.5, r'$I = %d^\circ$' % Id,
                ha='left', va='top')
        ax.set_ylabel(r'$a_{\rm p}$ [AU]')
        ax.set_ylim(ap_vals.min(), ap_vals.max())
        ax.set_xlim(a_vals.min(), a_vals.max())
        etac = get_etac(I)
        ax.contour(a_grid, ap_grid, etasync_norm, levels=[np.log10(etac)],
                   colors=['k'], linewidths=2.0)
    ax1.xaxis.set_visible(False)
    ax2.set_xlabel(r'$a$ [AU]')

    # use a shared colorbar, values are close enough
    ticks = np.arange(etasync_min, etasync_max + 0.001, 1)
    cb = plt.colorbar(cf, ax=ax, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels([r'$10^{%d}$' % f for f in ticks])
    cb.ax.set_ylabel(r'$\eta_{\rm sync}$')
    plt.savefig('5sehj_region', dpi=300)
    plt.close()

if __name__ == '__main__':
    # plot_eigens()

    plot_sehj_region()
