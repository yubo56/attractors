'''
more random plots
'''
import numpy as np
import scipy.optimize as opt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
LW=3.5

def get_etac(I):
    return (np.sin(I)**(2/3) + np.cos(I)**(2/3))**(-3/2)

def roots(I, eta):
    ''' returns theta roots from EOM '''
    eta_c = get_etac(I)

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

def get_cs(I):
    ''' gets list of cassini states at an uneven grid of etas '''
    etac = get_etac(I)
    eps = 1e-5
    min_eta, max_eta = etac / 30, etac * 30
    etas = np.concatenate((
        np.log(np.linspace(np.exp(min_eta), np.exp(etac - eps), 100)),
        np.exp(np.linspace(np.log(etac + eps), np.log(max_eta), 100)),
    ))
    cs_vals = [[], [], [], []]

    for eta_val in etas:
        root_vals = roots(I, eta_val)
        if eta_val > etac:
            cs_vals[1].append(root_vals[0])
            cs_vals[2].append(root_vals[1])
        else:
            for cs_lst, root_val in zip(cs_vals, root_vals):
                cs_lst.append(root_val)
    etas_four = etas[np.where(etas < etac)[0]]
    return etas, cs_vals, etas_four, etac

def plot_cs(I=np.radians(5)):
    etas, cs_vals, etas_four, etac = get_cs(I)

    plt.semilogx(etas_four, np.degrees(cs_vals[0]), 'y', lw=LW, label='1')
    plt.semilogx(etas, np.degrees(cs_vals[1]), 'r', lw=LW, label='2')
    plt.semilogx(etas, np.degrees(cs_vals[2]), 'm', lw=LW, label='3')
    plt.semilogx(etas_four, np.degrees(cs_vals[3]), 'c', lw=LW, label='4')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\theta$ (deg)')
    plt.xlim([min(etas), max(etas)])
    plt.yticks([-180 + np.degrees(I),
                -90,
                np.degrees(I),
                90],
               [r'$%d$' % (np.degrees(I) - 180),
                r'$-90$',
                r'$%d$' % np.degrees(I),
                r'$90$'])
    # place upper right corner of legend flush against eta -> infinity asymptote
    # (np.degrees(I))
    ymin, ymax = plt.ylim()
    y_perc = (np.degrees(I) - ymin) / (ymax - ymin)
    legend = plt.legend(fontsize=16, loc='upper right',
                        bbox_to_anchor=(1.0, y_perc))
    plt.axhline(np.degrees(I), lw=0.8, c='k', ls='dashed')
    plt.axhline(-180 + np.degrees(I), lw=0.8, c='k', ls='dashed')
    plt.axvline(etac, c='k', lw='0.8', ls='dashed')
    plt.title(r'$I = %d^\circ$' % np.degrees(I))
    plt.tight_layout()
    plt.savefig('2_cs_locs', dpi=400)

    legend.remove()
    # place upper right corner of legend flush against eta -> infinity asymptote
    # (np.degrees(I))
    legend = plt.legend(fontsize=16, loc='upper left',
                        bbox_to_anchor=(0.0, y_perc))
    xlims = plt.xlim()
    plt.xlim(xlims[1], xlims[0]) # flip
    plt.savefig('2_cs_locs_flip', dpi=400)
    plt.clf()

def plot_eigens(I=np.radians(5)):
    etas, cs_vals, etas_four, etac = get_cs(I)
    def lambda2(eta, q, sign):
        # note that the 4 CSs are 0, pi/2, -pi, -pi/2 by convention, which
        # correspond to sign choices of -1 for all of them.
        return (
            (np.sin(q) - sign * eta * np.sin(I) / (np.sin(q)**2)) *
            (sign * eta * np.sin(I))) / (1 + eta**2)
    plt.semilogx(etas_four,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas_four, cs_vals[0])],
                  'y', label='1')
    plt.semilogx(etas,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas, cs_vals[1])],
                  'r', label='2')
    plt.semilogx(etas,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas, cs_vals[2])],
                  'm', label='3')
    plt.semilogx(etas_four,
                 [lambda2(e, q, -1)
                  for e, q in zip(etas_four, cs_vals[3])],
                  'c', label='4')
    # plt.yscale('symlog', linthreshy=1e-5)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\lambda^2/ (1 + \eta^2)$')
    plt.legend(loc='lower right')
    plt.xlim([min(etas), max(etas)])
    plt.axhline(0, lw=0.8, c='k', ls='dashed')
    plt.axvline(etac, c='k', lw='0.8', ls='dashed')
    plt.title(r'$I = %d^\circ, \eta_c = %.3f$' % (np.degrees(I), etac))
    plt.tight_layout()
    plt.savefig('2_lambdas.png', dpi=400)
    plt.clf()

if __name__ == '__main__':
    plot_cs(np.radians(5))
    # plot_eigens(np.radians(5))
