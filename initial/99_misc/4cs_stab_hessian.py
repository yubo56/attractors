import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
import scipy.optimize as opt


def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

def roots(I, s_c, s):
    '''
    returns theta roots from EOM, not phi (otherwise same as 0_eta func)
    '''
    eta_c = get_etac(I)
    eta = s_c / s

    # function to minimize and derivatives
    f = lambda q: -eta * np.sin(q - I) + np.sin(q) * np.cos(q)
    fp = lambda q: -eta * np.cos(q - I) + np.cos(2 * q)

    if eta < eta_c:
        roots = []
        inits = [0, np.pi / 2, -np.pi, -np.pi / 2]
        for qi in inits:
            roots.append(opt.newton(f, qi, fprime=fp))
        return np.array(roots)

    else:
        roots = []
        inits = [np.pi / 2 - I, -np.pi + I]
        for qi in inits:
            # newton doesn't seem to work very well here @ large eta
            # roots.append(opt.newton(f, qi, fprime=fp))
            dq = np.pi / 2
            roots.append(opt.bisect(f, qi - dq, qi + dq))
        return np.array(roots)

def plot_tCE_stab():
    # TODO unused since inaccurate, but maybe can plot tCE stability as a
    # function of s_c
    I = np.radians(5)
    s_c = 0.3

    num_etas = 100
    eigs = np.full((12, num_etas), -1000 + 0j, dtype=np.complex64)
    # CS2, CS3, CS1, CS4
    eps = 3e-3
    eta_c = get_etac(I)
    s_vals = np.geomspace(s_c / (1.3 * eta_c), s_c * 10, num_etas)

    # convention: phi = 0
    for idx0, s in enumerate(s_vals):
        _cs_qs = roots(I, s_c, s)
        if len(_cs_qs) == 2:
            cs_qs = _cs_qs
        else:
            cs_qs = [_cs_qs[1], _cs_qs[2], _cs_qs[0], _cs_qs[3]]
        for idx, q in enumerate(cs_qs):
            mat = [
                [
                    -eps * (- np.cos(q) + 2 / s) * np.cos(q) - eps * np.sin(q)**2,
                    -np.sin(I),
                    2 * eps * np.sin(q) / s**2,
                ],
                [
                    np.sin(I) / np.sin(q)**2 + s * np.sin(q) / s_c,
                    0,
                    -np.cos(q) / s_c,
                ],
                [
                    eps * (2 * s * np.sin(q) * np.cos(q) - 2 * np.sin(q)),
                    0,
                    -eps * (1 + np.cos(q)**2),
                ],
            ]
            eigs_lst = np.linalg.eigvals(mat)
            eigs[idx * 3: (idx + 1) * 3, idx0] = eigs_lst
    fig, ((ax3, ax1), (ax2, ax4)) = plt.subplots(
        2, 2,
        figsize=(10, 10),
        sharex=True)
    ax1.plot(s_c / s_vals, eigs[0].real, 'b')
    ax1.plot(s_c / s_vals, eigs[1].real, 'r--')
    ax1.plot(s_c / s_vals, eigs[2].real, 'g')
    ax1.set_ylabel('CS2 Eigenvalues Real Part')
    ax2.plot(s_c / s_vals, eigs[3].real, 'b')
    ax2.plot(s_c / s_vals, eigs[4].real, 'r--')
    ax2.plot(s_c / s_vals, eigs[5].real, 'g')
    ax2.set_ylabel('CS3 Eigenvalues Real Part')

    idxs = np.where(eigs[6].real > -999)[0]
    ax3.plot(s_c / s_vals[idxs], eigs[6][idxs].real, 'b')
    ax3.plot(s_c / s_vals[idxs], eigs[7][idxs].real, 'r--')
    ax3.plot(s_c / s_vals[idxs], eigs[8][idxs].real, 'g')
    ax3.set_ylabel('CS1 Eigenvalues Real Part')
    idxs = np.where(eigs[9].real > -999)[0]
    ax4.plot(s_c / s_vals[idxs], eigs[9][idxs].real, 'b')
    ax4.plot(s_c / s_vals[idxs], eigs[10][idxs].real, 'r--')
    ax4.plot(s_c / s_vals[idxs], eigs[11][idxs].real, 'g')
    ax4.set_ylabel('CS4 Eigenvalues Real Part')

    for ax in [ax2, ax4]:
        ax.set_xlabel(r'$\eta$')

    eta_c = get_etac(I)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(eta_c, c='k', lw=1.0, ls=':')
        ylims = ax.get_ylim()
        ax.set_ylim(top=np.abs(ylims).max(), bottom=-np.abs(ylims).max())

    for ax in [ax1, ax4]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('4_cs_stab_hessian', dpi=300)
    plt.close()

def plot_CS_stab():
    I = np.radians(5)
    s_c = 0.3

    num_etas = 100
    eigs = np.full((8, num_etas), -1000 + 0j, dtype=np.complex64)
    # CS2, CS3, CS1, CS4
    eps = 3e-3
    eta_c = get_etac(I)
    s_vals = np.geomspace(s_c / (1.3 * eta_c), s_c * 10, num_etas)

    # convention: phi = 0
    for idx0, s in enumerate(s_vals):
        _cs_qs = roots(I, s_c, s)
        if len(_cs_qs) == 2:
            cs_qs = _cs_qs
        else:
            cs_qs = [_cs_qs[1], _cs_qs[2], _cs_qs[0], _cs_qs[3]]
        for idx, q in enumerate(cs_qs):
            mat = [
                [
                    -eps * (- np.cos(q) + 2 / s) * np.cos(q) - eps * np.sin(q)**2,
                    -np.sin(I),
                ],
                [
                    np.sin(I) / np.sin(q)**2 + s * np.sin(q) / s_c,
                    0,
                ],
            ]
            eigs_lst = np.linalg.eigvals(mat)
            eigs[idx * 2: (idx + 1) * 2, idx0] = eigs_lst
    fig, ((ax3, ax1), (ax2, ax4)) = plt.subplots(
        2, 2,
        figsize=(10, 10),
        sharex=True)
    ax1.plot(s_c / s_vals, eigs[0].real, 'b')
    ax1.plot(s_c / s_vals, eigs[1].real, 'r--')
    ax1.set_ylabel('CS2 Eigenvalues Real Part')
    ax2.plot(s_c / s_vals, eigs[2].real, 'b')
    ax2.plot(s_c / s_vals, eigs[3].real, 'r--')
    ax2.set_ylabel('CS3 Eigenvalues Real Part')

    idxs = np.where(eigs[4].real > -999)[0]
    ax3.plot(s_c / s_vals[idxs], eigs[4][idxs].real, 'b')
    ax3.plot(s_c / s_vals[idxs], eigs[5][idxs].real, 'r--')
    ax3.set_ylabel('CS1 Eigenvalues Real Part')
    idxs = np.where(eigs[6].real > -999)[0]
    ax4.plot(s_c / s_vals[idxs], eigs[6][idxs].real, 'b')
    ax4.plot(s_c / s_vals[idxs], eigs[7][idxs].real, 'r--')
    ax4.set_ylabel('CS4 Eigenvalues Real Part')

    for ax in [ax2, ax4]:
        ax.set_xlabel(r'$\eta$')

    eta_c = get_etac(I)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(eta_c, c='k', lw=1.0, ls=':')
        ylims = ax.get_ylim()
        ax.set_ylim(top=np.abs(ylims).max(), bottom=-np.abs(ylims).max())

    for ax in [ax1, ax4]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    plt.savefig('4_cs_stab_CS', dpi=300)
    plt.close()

if __name__ == '__main__':
    # plot_tCE_stab()
    plot_CS_stab()
