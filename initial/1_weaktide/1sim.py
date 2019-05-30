import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from utils import solve_ic, to_ang, get_etac, get_mu4, stringify, H
PLOT_DIR = '1plots'
NUM_CASES = 4 # number of return values from traj_for_sc

def get_name(s_c, eps, mu0):
    return stringify(s_c, mu0, strf='%.3f').replace('-', 'n')

def traj_for_sc(I, s_c, eps, mu0, s0, tf=2500):
    '''
    return value dictates qualitatively what the final state is:

    0 - eta < eta_c, converging to CS2
    1 - eta < eta_c, above separatrix
    2 - eta < eta_c, below separatrix ("unconverged")
    3 - eta > eta_c (in infinite time, converges to CS2)
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1)
    init = [-np.sqrt(1 - mu0**2), 0, mu0, s0]

    t, svec, s, _ = solve_ic(I, s_c, eps, init, tf)
    mu4 = get_mu4(I, s_c, s)

    q, phi = to_ang(*svec)

    # plot evolution of mu (at all times) and mu4
    ax1.plot(t, svec[2, :], 'go', label=r'$\mu$', markersize=0.3)
    base_slice = np.where(mu4 > 0)[0]
    ax1.plot(t[base_slice], mu4[base_slice], 'r', label=r'$\mu_4$',
             linewidth=0.7)

    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$\mu$')
    ax1.legend(loc='upper left')

    # scatter plot w/ color gradient + colorbar
    c = t
    ax2.plot(phi[-1], np.cos(q)[-1], 'ro', markersize=1.5)
    scat = ax2.scatter(phi, np.cos(q), c=c, s=0.3**2)
    ax2.set_xlabel(r'$\phi$')
    ax2.set_ylabel(r'$\cos\theta$')
    fig.colorbar(scat, ax=ax2)

    plt.suptitle(r'$(s_c, s_0, \cos \theta_0) = (%.2f, %d, %.2f)$' %
                  (s_c, s0, mu0))

    plt.savefig('%s/%s.png' % (PLOT_DIR, get_name(s_c, eps, mu0)), dpi=400)
    plt.close(fig)

    # get return value
    s_f = s[-1]
    eta = s_c / s[-1]
    mu_f, phi_f = np.cos(q[-1]), phi[-1]

    if eta > get_etac(I):
        return 3
    H_f = H(I, s_c, s_f, mu_f, phi_f)
    H_4 = H(I, s_c, s_f, mu4[-1], 0)
    if H_f > H_4:
        return 0
    if mu_f > mu4[-1]:
        return 1
    return 2

if __name__ == '__main__':
    resume_file = '1log_cum.log' # if this is None, run sims
    # resume_file = None

    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    s_c_arr = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    I = np.radians(20)
    eps = 1e-3
    s0 = 10
    s_c = 1.5
    nruns = 399
    mus = np.linspace(-1, 1, nruns + 2)[1: -1] # mus to iterate over
    labels = [
        r'$H > H_4$',
        r'$H < H_4, \mu > \mu_4$',
        r'$H < H_4, \mu < \mu_4$',
        r'$\eta > \eta_c$',
    ]

    # try to resume from log if available
    data_dict = {}
    if resume_file is not None:
        loglines = open(resume_file).readlines()
        for line in loglines:
            line = line.split()
            s_c = float(line[0])
            mu0 = float(line[1])
            outcome_idx = int(line[2])
            if s_c not in data_dict:
                data_dict[s_c] = [[] for i in range(NUM_CASES)]
            data_dict[s_c][outcome_idx].append(mu0)
    for s_c in s_c_arr:
        mu_arrs = [[] for i in range(NUM_CASES)]
        fig, ax = plt.subplots(1, 1)
        if resume_file is None:
            for mu0 in mus:
                ret = traj_for_sc(I, s_c, eps, mu0, s0)
                print(s_c, mu0, ret)
                mu_arrs[ret].append(mu0)
        else:
            mu_arrs = data_dict[s_c]
        ax.hist(mu_arrs, bins=40, label=labels, stacked=True)
        ax.legend(loc='upper left')
        ax.set_xlabel(r'$\mu_0$')
        ax.set_ylabel('Counts')
        ax.set_title('Total: %d' % nruns)
        plt.savefig('1hist_%s.png' % str(s_c).replace('.', '_'))
        plt.close(fig)
